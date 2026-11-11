#!/usr/bin/env python
"""Train the first PTrans model from random initialization.

This is a compact baseline, not a pretrained model.
It intentionally does NOT feed the provided molecular_weight column to the model.

Requirements:
    pip install torch pandas numpy rdkit

Example (Windows PowerShell or Ubuntu):
    python ptrans_train_baseline.py --config configs/rtx4090.json --skip-generation-eval

Small smoke test:
    python ptrans_train_baseline.py --device cpu --output-dir runs/smoke --dry-run --batch-size 2 --d-model 48 --nhead 4 --encoder-layers 1 --decoder-layers 1 --dim-feedforward 96 --max-train-samples 16 --max-val-samples 8 --skip-generation-eval
"""
import argparse
import gzip
import hashlib
import platform
import csv
from functools import partial
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

TOKEN_RE = re.compile(
    r"\[[^\]]+\]|%[0-9]{2}|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|Cu|Fe|Zn|Sn|Ag|Au|Hg|Pt|Pd|Mn|Co|Ni|Cr|As|"
    r"B|C|N|O|P|S|F|I|b|c|n|o|p|s|\(|\)|\.|=|#|-|\+|\\|/|:|~|\*|\$|@|[0-9]|."
)
ROOT = Path(__file__).resolve().parent

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
NUCLEUS_IDS = {"1H": 0, "13C": 1, "31P": 2}
NUCLEUS_SCALE = {"1H": 10.0, "13C": 200.0, "31P": 100.0}

def tokenize_smiles(smiles):
    tokens = TOKEN_RE.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"Tokenizer did not round-trip SMILES: {smiles}")
    return tokens

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clamp(x, low=-5.0, high=5.0):
    return max(low, min(high, float(x)))

def peak_to_features(peak):
    nucleus = peak.get("nucleus", "1H")
    lo = peak.get("shift_min_ppm")
    hi = peak.get("shift_max_ppm")
    if lo is None and hi is None:
        center, width = 0.0, 0.0
    else:
        lo = float(lo if lo is not None else hi)
        hi = float(hi if hi is not None else lo)
        center = (lo + hi) / 2.0
        width = abs(hi - lo)
    scale = NUCLEUS_SCALE.get(nucleus, 100.0)
    integral = peak.get("integral_protons")
    integral = 0.0 if integral is None else clamp(float(integral) / 20.0, 0.0, 5.0)
    couplings = peak.get("coupling_hz") or []
    vals = []
    for value in couplings[:3]:
        try:
            vals.append(clamp(float(value) / 100.0))
        except (TypeError, ValueError):
            vals.append(0.0)
    vals += [0.0] * (3 - len(vals))
    return [
        clamp(center / scale),
        clamp(width / scale, 0.0, 5.0),
        integral,
        min(len(couplings), 3) / 3.0,
        vals[0], vals[1], vals[2]
    ]

def record_source(record, multiplicity_vocab):
    features, nucleus_ids, mult_ids = [], [], []
    for nucleus in ("1H", "13C", "31P"):
        for peak in record["nmr"].get(nucleus, []):
            features.append(peak_to_features(peak))
            nucleus_ids.append(NUCLEUS_IDS[nucleus])
            mult = peak.get("multiplicity") or "<none>"
            mult_ids.append(multiplicity_vocab.get(mult, multiplicity_vocab["<unk>"]))
    if not features:
        features = [[0.0] * 7]
        nucleus_ids = [0]
        mult_ids = [multiplicity_vocab["<unk>"]]
    return features, nucleus_ids, mult_ids

class PTransDataset(Dataset):
    def __init__(self, records, vocab, multiplicity_vocab, max_target_tokens):
        self.records = records
        self.vocab = vocab
        self.multiplicity_vocab = multiplicity_vocab
        self.max_target_tokens = max_target_tokens
        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]
        self.unk_id = vocab["<unk>"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        features, nucleus_ids, mult_ids = record_source(record, self.multiplicity_vocab)
        tokens = tokenize_smiles(record["canonical_smiles"])
        ids = [self.bos_id] + [self.vocab.get(t, self.unk_id) for t in tokens] + [self.eos_id]
        if len(ids) > self.max_target_tokens:
            raise ValueError(
                f"{record['sample_id']} has {len(ids)} target tokens, above max_target_tokens="
                f"{self.max_target_tokens}; increase the setting rather than silently truncating."
            )
        return {
            "sample_id": record["sample_id"],
            "smiles": record["canonical_smiles"],
            "features": torch.tensor(features, dtype=torch.float32),
            "nucleus_ids": torch.tensor(nucleus_ids, dtype=torch.long),
            "mult_ids": torch.tensor(mult_ids, dtype=torch.long),
            "target_ids": torch.tensor(ids, dtype=torch.long),
        }

def collate_batch(batch, pad_id):
    batch_size = len(batch)
    max_src = max(x["features"].shape[0] for x in batch)
    max_tgt = max(x["target_ids"].shape[0] for x in batch)
    src_features = torch.zeros(batch_size, max_src, 7, dtype=torch.float32)
    src_nucleus = torch.zeros(batch_size, max_src, dtype=torch.long)
    src_mult = torch.zeros(batch_size, max_src, dtype=torch.long)
    src_pad_mask = torch.ones(batch_size, max_src, dtype=torch.bool)
    target = torch.full((batch_size, max_tgt), pad_id, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        t = item["target_ids"].shape[0]
        src_features[i, :n] = item["features"]
        src_nucleus[i, :n] = item["nucleus_ids"]
        src_mult[i, :n] = item["mult_ids"]
        src_pad_mask[i, :n] = False
        target[i, :t] = item["target_ids"]
    return {
        "sample_ids": [x["sample_id"] for x in batch],
        "smiles": [x["smiles"] for x in batch],
        "src_features": src_features,
        "src_nucleus": src_nucleus,
        "src_mult": src_mult,
        "src_pad_mask": src_pad_mask,
        "tgt_in": target[:, :-1],
        "tgt_out": target[:, 1:],
        "tgt_pad_mask": target[:, :-1].eq(pad_id),
    }

def make_collate(pad_id):
    return partial(collate_batch, pad_id=pad_id)

class PTransModel(nn.Module):
    def __init__(self, vocab_size, mult_vocab_size, d_model=192, nhead=6,
                 encoder_layers=3, decoder_layers=4, dim_feedforward=768,
                 dropout=0.1, max_target_tokens=384, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.peak_projection = nn.Linear(7, d_model)
        self.nucleus_embedding = nn.Embedding(3, d_model)
        self.mult_embedding = nn.Embedding(mult_vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.target_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_target_tokens, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def encode(self, src_features, src_nucleus, src_mult, src_pad_mask):
        src = (self.peak_projection(src_features)
               + self.nucleus_embedding(src_nucleus)
               + self.mult_embedding(src_mult))
        return self.encoder(src, src_key_padding_mask=src_pad_mask)

    def decode(self, memory, src_pad_mask, tgt_ids, tgt_pad_mask=None):
        length = tgt_ids.size(1)
        if length > self.position_embedding.num_embeddings:
            raise ValueError("Target sequence exceeds positional embedding capacity.")
        positions = torch.arange(length, device=tgt_ids.device).unsqueeze(0)
        tgt = self.target_embedding(tgt_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, device=tgt_ids.device, dtype=torch.bool), diagonal=1
        )
        decoded = self.decoder(
            tgt, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.output_projection(decoded)

    def forward(self, src_features, src_nucleus, src_mult, src_pad_mask, tgt_in, tgt_pad_mask):
        memory = self.encode(src_features, src_nucleus, src_mult, src_pad_mask)
        return self.decode(memory, src_pad_mask, tgt_in, tgt_pad_mask)

def move_batch(batch, device):
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

def batch_loss(model, batch, criterion):
    logits = model(
        batch["src_features"], batch["src_nucleus"], batch["src_mult"],
        batch["src_pad_mask"], batch["tgt_in"], batch["tgt_pad_mask"]
    )
    return criterion(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1))

@torch.no_grad()
def beam_search_one(model, src_features, src_nucleus, src_mult, src_pad_mask,
                    vocab, inverse_vocab, max_new_tokens, beam_size=10, length_penalty=0.7):
    """Beam search for one spectrum; returns unique candidate SMILES and scores."""
    pad_id, bos_id, eos_id, unk_id = (vocab[x] for x in SPECIALS)
    memory = model.encode(src_features, src_nucleus, src_mult, src_pad_mask)
    beams = [([bos_id], 0.0, False)]

    def rank_score(sequence, score):
        length = max(1, len(sequence) - 1)  # exclude BOS
        penalty = ((5.0 + length) / 6.0) ** length_penalty
        return score / penalty

    for _ in range(max_new_tokens):
        active = [b for b in beams if not b[2]]
        finished = [b for b in beams if b[2]]
        if not active:
            break

        tgt = torch.tensor([b[0] for b in active], dtype=torch.long, device=src_features.device)
        expanded_memory = memory.expand(len(active), -1, -1)
        expanded_mask = src_pad_mask.expand(len(active), -1)
        logits = model.decode(expanded_memory, expanded_mask, tgt)[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs[:, pad_id] = -torch.inf
        log_probs[:, bos_id] = -torch.inf
        log_probs[:, unk_id] = -torch.inf

        candidates = list(finished)
        for row_idx, (sequence, score, _) in enumerate(active):
            k = min(beam_size, log_probs.size(-1))
            values, token_ids = torch.topk(log_probs[row_idx], k=k)
            for value, token_id in zip(values.tolist(), token_ids.tolist()):
                new_sequence = sequence + [int(token_id)]
                new_score = score + float(value)
                candidates.append((new_sequence, new_score, int(token_id) == eos_id))

        candidates.sort(key=lambda b: rank_score(b[0], b[1]), reverse=True)
        beams = candidates[:beam_size]
        if all(b[2] for b in beams):
            break

    beams.sort(key=lambda b: rank_score(b[0], b[1]), reverse=True)
    unique = []
    seen = set()
    for sequence, score, finished in beams:
        if not finished:
            continue
        smiles = ids_to_smiles(sequence, inverse_vocab, bos_id, eos_id, pad_id)
        try:
            mol = Chem.MolFromSmiles(smiles)
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None and mol.GetNumAtoms() else None
        except Exception:
            canonical = None
        key = canonical if canonical is not None else "INVALID::" + smiles
        if key in seen:
            continue
        seen.add(key)
        unique.append((smiles, float(rank_score(sequence, score))))
        if len(unique) >= beam_size:
            break
    return unique

def ids_to_smiles(ids, inverse_vocab, bos_id, eos_id, pad_id):
    tokens = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == eos_id:
            break
        if token_id in (bos_id, pad_id):
            continue
        tokens.append(inverse_vocab[token_id])
    return "".join(tokens)

def structure_match(predicted, truth):
    try:
        pred_mol = Chem.MolFromSmiles(predicted)
        true_mol = Chem.MolFromSmiles(truth)
        if pred_mol is None or true_mol is None or not pred_mol.GetNumAtoms():
            return False, False
        pred_can = Chem.MolToSmiles(pred_mol, canonical=True, isomericSmiles=True)
        true_can = Chem.MolToSmiles(true_mol, canonical=True, isomericSmiles=True)
        return True, pred_can == true_can
    except Exception:
        return False, False

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PTrans baseline; relative paths use the package directory")
    parser.add_argument('--config', default='configs/rtx4090.json')
    parser.add_argument('--data', default='data/ptrans_normalized_dataset.jsonl.gz')
    parser.add_argument('--manifest', default='data/ptrans_split_manifest.csv')
    parser.add_argument('--output-dir', default='runs/ptrans_v0')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='cuda')
    for name, value in [('epochs',60),('batch-size',16),('d-model',192),('nhead',6),
                        ('encoder-layers',3),('decoder-layers',4),('dim-feedforward',768),
                        ('max-target-tokens',384),('patience',8),('seed',42),('num-workers',0),
                        ('cpu-threads',2),('max-train-samples',0),('max-val-samples',0),
                        ('eval-samples',32),('beam-size',10),('generation-max-tokens',0)]:
        parser.add_argument('--'+name, type=int, default=value)
    for name, value in [('dropout',0.1),('learning-rate',3e-4),('weight-decay',1e-4),
                        ('label-smoothing',0.05),('length-penalty',0.7)]:
        parser.add_argument('--'+name, type=float, default=value)
    parser.add_argument('--skip-generation-eval', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true', help='Held-out test, only with --eval-only')
    parser.add_argument('--dry-run', action='store_true', help='One forward/backward batch; no optimizer update')
    parser.add_argument('--eval-only', action='store_true', help='Load --checkpoint and evaluate without training')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--resume', default=None, help='Resume from this script\'s last.pt, saving in the same output directory')
    preliminary, _ = parser.parse_known_args(argv)
    config_path = ROOT / preliminary.config
    config = json.loads(config_path.read_text(encoding='utf-8'))
    allowed = {a.dest for a in parser._actions} - {'help','config'}
    unknown = set(config)-allowed
    if unknown:
        parser.error(f'Unknown config keys: {sorted(unknown)}')
    for action in parser._actions:
        if action.dest in config:
            v = config[action.dest]
            if action.type is not None and (isinstance(v, bool) or not isinstance(v, (int,float) if action.type is float else action.type)):
                parser.error(f'Invalid config type for {action.dest}')
            if action.choices and v not in action.choices:
                parser.error(f'Invalid config choice for {action.dest}')
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for field in ['epochs','batch_size','d_model','nhead','encoder_layers','decoder_layers',
                  'dim_feedforward','patience','cpu_threads','beam_size']:
        if getattr(args,field)<1: parser.error(field+' must be positive')
    for field in ['num_workers','max_train_samples','max_val_samples','eval_samples','generation_max_tokens']:
        if getattr(args,field)<0: parser.error(field+' must be nonnegative')
    if args.d_model % args.nhead or args.max_target_tokens<3:
        parser.error('d-model must be divisible by nhead; max-target-tokens must be >=3')
    if not (0<=args.dropout<1 and 0<=args.label_smoothing<1) or args.learning_rate<=0 or args.weight_decay<0:
        parser.error('Invalid optimizer/dropout values')
    if args.evaluate_test and not args.eval_only:
        parser.error('--evaluate-test requires --eval-only and a frozen checkpoint')
    if args.eval_only and (not args.checkpoint or args.resume or args.dry_run or args.skip_generation_eval):
        parser.error('--eval-only requires --checkpoint and cannot combine with resume/dry-run/skip-generation-eval')
    if args.checkpoint and not args.eval_only:
        parser.error('--checkpoint requires --eval-only; use --resume for training')
    for field in ['data','manifest','output_dir','config','checkpoint','resume']:
        v=getattr(args,field)
        if v is not None: setattr(args,field,str((ROOT/v).resolve()))
    return args

def load_records(data, manifest):
    opener = gzip.open if Path(data).suffix == '.gz' else open
    with opener(data, 'rt', encoding='utf-8') as f:
        records=[json.loads(line) for line in f if line.strip()]
    with open(manifest,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    by_id={r['sample_id']:r for r in rows}
    if len(by_id)!=len(rows) or len(records)!=len(rows):
        raise ValueError('Manifest/data size mismatch or duplicate sample IDs')
    seen=set()
    groups={}
    structures={}
    for r in records:
        sid=r['sample_id']; m=by_id.get(sid)
        if sid in seen or m is None: raise ValueError('Duplicate or missing sample: '+sid)
        seen.add(sid)
        for field in ['canonical_smiles','split_scaffold','split_random_structure']:
            if r[field]!=m[field]: raise ValueError(f'Manifest mismatch: {sid} {field}')
        split=r['split_scaffold']
        if split not in ['train','val','test']: raise ValueError('Unknown split: '+split)
        for d,key in [(groups,m['scaffold_group_key']),(structures,m['canonical_smiles'])]:
            if key in d and d[key]!=split: raise ValueError('Cross-split group leakage')
            d[key]=split
    return records

def build_vocabs(records):
    tokens=set(); multiplicities={'<none>'}
    for r in records:
        tokens.update(tokenize_smiles(r['canonical_smiles']))
        for peaks in r['nmr'].values():
            multiplicities.update(p.get('multiplicity') or '<none>' for p in peaks)
    inverse=SPECIALS+sorted(tokens-set(SPECIALS))
    mult=['<pad>','<unk>']+sorted(multiplicities-{'<pad>','<unk>'})
    return {t:i for i,t in enumerate(inverse)}, inverse, {t:i for i,t in enumerate(mult)}

def atomic_save(obj,path):
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,temp)
    temp.replace(path)

def fingerprint(records):
    h=hashlib.sha256()
    for r in records:
        h.update(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()

def evaluate(model, records_by_split, vocab, inverse, mult_vocab, model_args, args, device, output_dir):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    model.eval()
    predictions=[]; metrics={}
    for split_name in (['val','test'] if args.evaluate_test else ['val']):
        records=records_by_split[split_name]
        if args.eval_samples: records=records[:args.eval_samples]
        hits=[0,0,0]; valid=total=0
        for r in records:
            ds=PTransDataset([r],vocab,mult_vocab,model_args['max_target_tokens'])
            batch=move_batch(make_collate(vocab['<pad>'])([ds[0]]),device)
            candidates=beam_search_one(model,batch['src_features'],batch['src_nucleus'],batch['src_mult'],
                batch['src_pad_mask'],vocab,inverse,
                max_new_tokens=min(args.generation_max_tokens or model_args['max_target_tokens']-1, model_args['max_target_tokens']-1),
                beam_size=args.beam_size,length_penalty=args.length_penalty)
            matches=[]
            for rank,(pred,score) in enumerate(candidates,1):
                is_valid,is_match=structure_match(pred,r['canonical_smiles'])
                valid+=int(is_valid); total+=1; matches.append(is_match)
                predictions.append({'sample_id':r['sample_id'],'split':split_name,'rank':rank,
                    'true_canonical_smiles':r['canonical_smiles'],'predicted_smiles':pred,
                    'beam_score_length_normalized':score,'valid_smiles':is_valid,'structure_exact_match':is_match})
            if not candidates:
                predictions.append({'sample_id':r['sample_id'],'split':split_name,'rank':0,
                    'true_canonical_smiles':r['canonical_smiles'],'predicted_smiles':'',
                    'beam_score_length_normalized':None,'valid_smiles':False,'structure_exact_match':False})
            for i,k in enumerate([1,5,10]): hits[i]+=int(any(matches[:k]))
        n=len(records)
        metrics[split_name]={'samples':n,'total_split_samples':len(records_by_split[split_name]),
            'pilot_subset':n<len(records_by_split[split_name]), 'beam_size':args.beam_size,
            'generation_limit':args.generation_max_tokens or model_args['max_target_tokens']-1,
            'top1':hits[0]/max(n,1),'top5':hits[1]/max(n,1) if args.beam_size>=5 else None,
            'top10':hits[2]/max(n,1) if args.beam_size>=10 else None,
            'candidate_valid_rate':valid/total if total else 0.0, 'completed_candidates':total}
        print(split_name, json.dumps(metrics[split_name]),flush=True)
    keys=['sample_id','split','rank','true_canonical_smiles','predicted_smiles','beam_score_length_normalized','valid_smiles','structure_exact_match']
    with (output_dir/'beam_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader();w.writerows(predictions)
    (output_dir/'evaluation_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')

def main(argv=None):
    args=parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this config but unavailable. Install CUDA PyTorch/check driver, or use --device cpu for a local smoke test.')
    device=torch.device('cuda' if args.device=='cuda' or args.device=='auto' and torch.cuda.is_available() else 'cpu')
    print('Device:',device, '; torch:',torch.__version__,flush=True)
    if device.type=='cuda': print('GPU:',torch.cuda.get_device_name(0),flush=True)
    records=load_records(args.data,args.manifest)
    data_fingerprint=fingerprint(records)
    by_split={s:[r for r in records if r['split_scaffold']==s] for s in ['train','val','test']}
    if any(not v for v in by_split.values()): raise ValueError('Each split must be nonempty')
    checkpoint=None
    if args.eval_only or args.resume:
        checkpoint=torch.load(args.checkpoint or args.resume,map_location=device,weights_only=False)
        if checkpoint['dataset_fingerprint']!=data_fingerprint: raise ValueError('Checkpoint/data fingerprint mismatch')
        vocab=checkpoint['vocab']; inverse=checkpoint['inverse_vocab'];mult_vocab=checkpoint['multiplicity_vocab']
        model_args=checkpoint['model_args']
    else:
        vocab,inverse,mult_vocab=build_vocabs(by_split['train'])
        model_args={k:getattr(args,k) for k in ['d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_target_tokens']}
    if args.resume:
        for k,v in model_args.items():
            if getattr(args,k)!=v: raise ValueError('Resume architecture differs: '+k)
        for k in ['batch_size','seed','learning_rate','weight_decay','label_smoothing','max_train_samples','max_val_samples']:
            if vars(args)[k]!=checkpoint['args'][k]: raise ValueError('Resume training settings differ: '+k)
        if Path(args.output_dir)!=Path(args.resume).resolve().parent:
            raise ValueError('Resume must use original output directory (which also holds best.pt)')
        if not (Path(args.output_dir)/'best.pt').is_file(): raise ValueError('Resume requires best.pt beside last.pt')
    stats={}
    for split,rs in by_split.items():
        unknown=set(); unknown_records=0
        for r in rs:
            ts=tokenize_smiles(r['canonical_smiles'])
            if len(ts)+2>model_args['max_target_tokens']: raise ValueError('Target capacity too small: '+r['sample_id'])
            unseen=set(ts)-vocab.keys();unknown.update(unseen);unknown_records+=bool(unseen)
            features,_,_=record_source(r,mult_vocab)
            if not np.isfinite(np.array(features)).all(): raise ValueError('Non-finite features: '+r['sample_id'])
        stats[split]={'records':len(rs),'unknown_target_tokens':sorted(unknown),'records_with_unknown_tokens':unknown_records}
    print('Data/vocabulary:',json.dumps(stats),flush=True)
    model=PTransModel(len(vocab),len(mult_vocab),**model_args,pad_id=vocab['<pad>']).to(device)
    if checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    output_dir=Path(args.output_dir)
    if not args.eval_only and not args.resume and (output_dir/'best.pt').exists():
        raise ValueError('Output already contains best.pt; choose a new output directory or --resume last.pt')
    output_dir.mkdir(parents=True,exist_ok=True)
    environment={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
        'rdkit':__import__('rdkit').rdBase.rdkitVersion,'device':str(device),'cuda_runtime':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None}
    run_config={'arguments':vars(args),'model_args':model_args,'environment':environment,
        'dataset_fingerprint':data_fingerprint,'split_stats':stats,'vocabulary_source':'full train split only',
        'checkpoint':args.checkpoint or args.resume,'molecular_weight_input':False,'precision':'float32'}
    (output_dir/'run_config.json').write_text(json.dumps(run_config,indent=2)+'\n',encoding='utf-8')
    if args.eval_only:
        evaluate(model,by_split,vocab,inverse,mult_vocab,model_args,args,device,output_dir)
        return
    train=by_split['train'][:args.max_train_samples or None]
    val=by_split['val'][:args.max_val_samples or None]
    collate=make_collate(vocab['<pad>'])
    train_ds=PTransDataset(train,vocab,mult_vocab,model_args['max_target_tokens'])
    val_ds=PTransDataset(val,vocab,mult_vocab,model_args['max_target_tokens'])
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    criterion=nn.CrossEntropyLoss(ignore_index=vocab['<pad>'],label_smoothing=args.label_smoothing)
    best_val=float('inf'); best_epoch=0; stale=0; history=[]; start=1
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val=checkpoint['best_val_loss'];best_epoch=checkpoint['best_epoch'];stale=checkpoint['stale_epochs']
        start=checkpoint['epoch']+1;history=checkpoint['history']
        if args.epochs<start: raise ValueError('epochs must exceed last completed epoch')
        random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type=='cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
    print(f'Records used: train={len(train_ds)}, val={len(val_ds)}; parameters={sum(p.numel() for p in model.parameters()):,}',flush=True)
    if args.dry_run:
        model.train();optimizer.zero_grad(set_to_none=True)
        batch=move_batch(next(iter(train_loader)),device)
        loss=batch_loss(model,batch,criterion)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite dry-run loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Non-finite dry-run gradient')
        print(f'Dry run passed: loss={loss.item():.6f}; forward/backward only, no weight update',flush=True)
        return
    for epoch in range(start,args.epochs+1):
        model.train();train_sum=0.;train_n=0
        for batch in train_loader:
            batch=move_batch(batch,device);optimizer.zero_grad(set_to_none=True)
            loss=batch_loss(model,batch,criterion)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            n=int(batch['tgt_out'].ne(vocab['<pad>']).sum());train_sum+=loss.item()*n;train_n+=n
        model.eval();val_sum=0.;val_n=0
        with torch.no_grad():
            for batch in val_loader:
                batch=move_batch(batch,device);loss=batch_loss(model,batch,criterion)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite validation loss')
                n=int(batch['tgt_out'].ne(vocab['<pad>']).sum());val_sum+=loss.item()*n;val_n+=n
        train_loss=train_sum/train_n;val_loss=val_sum/val_n
        history.append({'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss})
        print(f'Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}',flush=True)
        improved=val_loss<best_val
        if improved:best_val=val_loss;best_epoch=epoch;stale=0
        else:stale+=1
        state={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
            'vocab':vocab,'inverse_vocab':inverse,'multiplicity_vocab':mult_vocab,'model_args':model_args,
            'args':vars(args),'dataset_fingerprint':data_fingerprint,'epoch':epoch,'history':history,
            'best_epoch':best_epoch,'best_val_loss':best_val,'stale_epochs':stale,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        if improved: atomic_save(state,output_dir/'best.pt')
        atomic_save(state,output_dir/'last.pt')
        (output_dir/'training_history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
        if stale>=args.patience:break
    print(f'Best checkpoint: {output_dir/"best.pt"} (epoch {best_epoch})',flush=True)
    if not args.skip_generation_eval:
        best=torch.load(output_dir/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(best['model_state_dict'])
        evaluate(model,by_split,vocab,inverse,mult_vocab,model_args,args,device,output_dir)

if __name__=='__main__':
    main()
