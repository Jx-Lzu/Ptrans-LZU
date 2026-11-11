#!/usr/bin/env python
"""PTrans peak embedding architecture controls; requires the original deployment.

The original data, tokenizer, feature transform, Transformer blocks and decoding
are imported from ptrans_train_baseline.py. No old project files are replaced.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import ptrans_train_baseline as base

ROOT = Path(__file__).resolve().parent
EXPECTED_BASE_SHA256 = 'a9ed2625b53c19d7299a9b3cb59768fab16b9946b278d3d6678d04d0553e4b07'
FORMAT = 'ptrans_peak_arch_v1'

# Common read-only API used by the forward-only conditioning diagnostic.
set_seed = base.set_seed
load_records = base.load_records
fingerprint = base.fingerprint
PTransDataset = base.PTransDataset
make_collate = base.make_collate
move_batch = base.move_batch
batch_loss = base.batch_loss
tokenize_smiles = base.tokenize_smiles


class PeakResidual(nn.Module):
    def __init__(self, hidden, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class PTransModel(base.PTransModel):
    def __init__(self, vocab_size, mult_vocab_size, architecture='linear', peak_hidden=64, **kwargs):
        if architecture not in ['linear','shared_mlp','nucleus_mlp']:
            raise ValueError('Unknown architecture: '+architecture)
        super().__init__(vocab_size, mult_vocab_size, **kwargs)
        self.architecture = architecture
        # Preserve both the baseline's initial output and its subsequent RNG state.
        with torch.random.fork_rng(devices=[]):
            count = 0 if architecture == 'linear' else 1 if architecture == 'shared_mlp' else 3
            self.peak_residuals = nn.ModuleList([PeakResidual(peak_hidden,self.d_model) for _ in range(count)])

    def encode(self, src_features, src_nucleus, src_mult, src_pad_mask):
        projected = self.peak_projection(src_features)
        if self.architecture == 'shared_mlp':
            projected = projected + self.peak_residuals[0](src_features)
        elif self.architecture == 'nucleus_mlp':
            for nucleus_id, branch in enumerate(self.peak_residuals):
                projected = projected + branch(src_features) * src_nucleus.eq(nucleus_id).unsqueeze(-1)
        src = projected + self.nucleus_embedding(src_nucleus) + self.mult_embedding(src_mult)
        return self.encoder(src, src_key_padding_mask=src_pad_mask)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PTrans baseline; relative paths use the package directory")
    parser.add_argument('--config', default='configs/arch_shared_mlp.json')
    parser.add_argument('--data', default='data/ptrans_normalized_dataset.jsonl.gz')
    parser.add_argument('--manifest', default='data/ptrans_split_manifest.csv')
    parser.add_argument('--output-dir', default='runs/arch_shared_mlp')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='cuda')
    for name, value in [('epochs',60),('batch-size',16),('d-model',192),('nhead',6),
                        ('encoder-layers',3),('decoder-layers',4),('dim-feedforward',768),
                        ('max-target-tokens',384),('patience',8),('seed',42),('num-workers',0),
                        ('cpu-threads',2),('max-train-samples',0),('max-val-samples',0),
                        ('eval-samples',0),('beam-size',10),('generation-max-tokens',0)]:
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
    parser.add_argument('--architecture', choices=['linear','shared_mlp','nucleus_mlp'], default='shared_mlp')
    parser.add_argument('--peak-hidden', type=int, default=64)
    parser.add_argument('--eval-id-file', default='configs/arch_eval256_seed42.json')
    parser.add_argument('--eval-scope', choices=['fixed','full'], default='fixed')
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
                  'dim_feedforward','patience','cpu_threads','beam_size','peak_hidden']:
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
    if args.resume and args.dry_run:
        parser.error('Use dry-run for new experiments, not with resume')
    for field in ['data','manifest','output_dir','config','checkpoint','resume','eval_id_file']:
        v=getattr(args,field)
        if v is not None: setattr(args,field,str((ROOT/v).resolve()))
    return args


def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_eval_ids(path, records, data_fingerprint):
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest['dataset_fingerprint']!=data_fingerprint:
        raise ValueError('Frozen evaluation IDs do not belong to this dataset')
    ids=manifest['sample_ids']
    if not ids or len(set(ids))!=len(ids) or manifest['samples']!=len(ids):
        raise ValueError('Invalid evaluation ID manifest')
    by_id={r['sample_id']:r for r in records}
    if any(s not in by_id or by_id[s]['split_scaffold']!='val' for s in ids):
        raise ValueError('Evaluation IDs must all be in validation split')
    return ids, manifest


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def evaluate(model, by_split, vocab, inverse, mult_vocab, model_args, args, device, out, eval_ids, checkpoint_sha):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    model.eval()
    all_predictions=[];metrics={}
    for split in (['val','test'] if args.evaluate_test else ['val']):
        whole=by_split[split]
        if split=='val' and args.eval_scope=='fixed':
            by_id={r['sample_id']:r for r in whole}
            selected=[by_id[s] for s in eval_ids]
            sampling='frozen_uniform_random_validation_manifest'
        else:
            selected=list(whole)
            sampling='full_split' if not args.eval_samples else 'split_prefix'
        if args.eval_samples:selected=selected[:args.eval_samples]
        hits=[0,0,0];valid=total=0
        sync(device);started=time.perf_counter()
        for j,r in enumerate(selected,1):
            ds=base.PTransDataset([r],vocab,mult_vocab,model_args['max_target_tokens'])
            batch=base.move_batch(base.make_collate(vocab['<pad>'])([ds[0]]),device)
            candidates=base.beam_search_one(model,batch['src_features'],batch['src_nucleus'],batch['src_mult'],
                batch['src_pad_mask'],vocab,inverse,
                max_new_tokens=min(args.generation_max_tokens or model_args['max_target_tokens']-1,model_args['max_target_tokens']-1),
                beam_size=args.beam_size,length_penalty=args.length_penalty)
            matches=[]
            for rank,(pred,score) in enumerate(candidates,1):
                ok,match=base.structure_match(pred,r['canonical_smiles'])
                valid+=int(ok);total+=1;matches.append(match)
                all_predictions.append({'sample_id':r['sample_id'],'split':split,'rank':rank,
                    'true_canonical_smiles':r['canonical_smiles'],'predicted_smiles':pred,
                    'beam_score_length_normalized':score,'valid_smiles':ok,'structure_exact_match':match})
            if not candidates:
                all_predictions.append({'sample_id':r['sample_id'],'split':split,'rank':0,
                    'true_canonical_smiles':r['canonical_smiles'],'predicted_smiles':'',
                    'beam_score_length_normalized':None,'valid_smiles':False,'structure_exact_match':False})
            for i,k in enumerate([1,5,10]):hits[i]+=int(any(matches[:k]))
            if j%16==0 or j==len(selected):print(f'Evaluation {split}: {j}/{len(selected)}',flush=True)
        sync(device)
        n=len(selected)
        metrics[split]={'samples':n,'total_split_samples':len(whole),'pilot_subset':n<len(whole),
            'sampling':sampling,'sample_ids':[r['sample_id'] for r in selected],
            'eval_manifest_sha256':file_sha256(Path(args.eval_id_file)) if split=='val' and args.eval_scope=='fixed' else None,
            'checkpoint_sha256':checkpoint_sha,'architecture':model_args.get('architecture','linear'),
            'beam_size':args.beam_size,'generation_limit':args.generation_max_tokens or model_args['max_target_tokens']-1,
            'top1':hits[0]/max(n,1),'top5':hits[1]/max(n,1) if args.beam_size>=5 else None,
            'top10':hits[2]/max(n,1) if args.beam_size>=10 else None,
            'candidate_valid_rate':valid/total if total else 0.,'completed_candidates':total,
            'evaluation_seconds':time.perf_counter()-started}
        display={k:v for k,v in metrics[split].items() if k not in ['sample_ids','checkpoint_sha256','eval_manifest_sha256']}
        print(split,json.dumps(display),flush=True)
    columns=['sample_id','split','rank','true_canonical_smiles','predicted_smiles','beam_score_length_normalized','valid_smiles','structure_exact_match']
    with (out/'beam_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(all_predictions)
    (out/'evaluation_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')


def run_loss(model, loader, criterion, optimizer, device, pad_id):
    training=optimizer is not None
    model.train(training)
    total=0.;tokens=0;samples=0;steps=0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch=base.move_batch(batch,device)
            if training:optimizer.zero_grad(set_to_none=True)
            loss=base.batch_loss(model,batch,criterion)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training/validation loss')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            n=int(batch['tgt_out'].ne(pad_id).sum())
            total+=float(loss.item())*n;tokens+=n;samples+=len(batch['sample_ids']);steps+=1
    return total/tokens,samples,steps


def main(argv=None):
    args=parse_args(argv)
    base_file=ROOT/'ptrans_train_baseline.py'
    if file_sha256(base_file)!=EXPECTED_BASE_SHA256:
        raise RuntimeError('Original baseline script differs from the validated deployment. Keep the original baseline file unchanged.')
    torch.set_num_threads(args.cpu_threads);base.set_seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; use --device cpu only for local CPU checks')
    device=torch.device('cuda' if args.device=='cuda' or args.device=='auto' and torch.cuda.is_available() else 'cpu')
    print('Device:',device,'; torch:',torch.__version__,flush=True)
    if device.type=='cuda':print('GPU:',torch.cuda.get_device_name(0),flush=True)
    records=base.load_records(args.data,args.manifest)
    data_fingerprint=base.fingerprint(records)
    by_split={s:[r for r in records if r['split_scaffold']==s] for s in ['train','val','test']}
    if any(not rs for rs in by_split.values()):raise ValueError('All splits must be nonempty')
    eval_ids,eval_manifest=load_eval_ids(Path(args.eval_id_file),records,data_fingerprint)
    checkpoint=None;checkpoint_sha=None
    if args.checkpoint or args.resume:
        path=Path(args.checkpoint or args.resume)
        checkpoint_sha=file_sha256(path)
        checkpoint=torch.load(path,map_location=device,weights_only=False)
        if checkpoint['dataset_fingerprint']!=data_fingerprint:raise ValueError('Checkpoint/data fingerprint mismatch')
        vocab=checkpoint['vocab'];inverse=checkpoint['inverse_vocab'];mult=checkpoint['multiplicity_vocab']
        model_args=dict(checkpoint['model_args'])
        model_args.setdefault('architecture','linear');model_args.setdefault('peak_hidden',64)
    else:
        vocab,inverse,mult=base.build_vocabs(by_split['train'])
        model_args={k:getattr(args,k) for k in ['d_model','nhead','encoder_layers','decoder_layers',
            'dim_feedforward','dropout','max_target_tokens','architecture','peak_hidden']}
    out=Path(args.output_dir)
    if args.eval_only:
        if out.resolve()==Path(args.checkpoint).resolve().parent or (out/'best.pt').exists() or (out/'last.pt').exists():
            raise ValueError('Choose a separate evaluation output directory, not a training directory')
        if (out/'evaluation_metrics.json').exists():raise FileExistsError('Evaluation output exists; choose a new --output-dir')
    elif not args.resume and any((out/f).exists() for f in ['best.pt','last.pt','training_history.json']):
        raise FileExistsError('Training output exists; choose a new directory or --resume its last.pt')
    if args.resume:
        if checkpoint.get('format_version')!=FORMAT:
            raise ValueError('Legacy baseline checkpoints are supported for evaluation only; start architecture experiments from scratch')
        for k,v in model_args.items():
            if getattr(args,k)!=v:raise ValueError('Resume model setting differs: '+k)
        for k in ['batch_size','seed','learning_rate','weight_decay','label_smoothing','max_train_samples','max_val_samples','num_workers','cpu_threads']:
            if getattr(args,k)!=checkpoint['args'][k]:raise ValueError('Resume training setting differs: '+k)
        if out.resolve()!=Path(args.resume).resolve().parent or not (out/'best.pt').is_file():
            raise ValueError('Resume in the original output directory with best.pt beside last.pt')
        if args.epochs<=checkpoint['epoch']:raise ValueError('epochs must exceed last completed epoch')
        if checkpoint['stale_epochs']>=args.patience:
            raise ValueError('This run already early-stopped; increasing epochs alone does not reset early stopping')
    stats={}
    for split,rs in by_split.items():
        unknown=set();unknown_samples=0
        for r in rs:
            ts=base.tokenize_smiles(r['canonical_smiles'])
            if len(ts)+2>model_args['max_target_tokens']:raise ValueError('Target capacity too small: '+r['sample_id'])
            unseen=set(ts)-vocab.keys();unknown.update(unseen);unknown_samples+=bool(unseen)
            features,_,_=base.record_source(r,mult)
            if not np.isfinite(np.array(features)).all():raise ValueError('Non-finite features: '+r['sample_id'])
        stats[split]={'records':len(rs),'unknown_target_tokens':sorted(unknown),'records_with_unknown_tokens':unknown_samples}
    model=PTransModel(len(vocab),len(mult),**model_args,pad_id=vocab['<pad>']).to(device)
    if checkpoint:model.load_state_dict(checkpoint['model_state_dict'])
    count=sum(p.numel() for p in model.parameters())
    print('Data/vocabulary:',json.dumps(stats),flush=True)
    print(f'Architecture={model_args["architecture"]}; parameters={count:,}; fixed random validation IDs={len(eval_ids)}',flush=True)
    environment={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
        'rdkit':__import__('rdkit').rdBase.rdkitVersion,'device':str(device),'cuda_runtime':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None}
    run_config={'arguments':vars(args),'model_args':model_args,'parameter_count':count,'environment':environment,
        'dataset_fingerprint':data_fingerprint,'split_stats':stats,'vocabulary_source':'full train split only',
        'precision':'float32','molecular_weight_input':False,'baseline_script_sha256':EXPECTED_BASE_SHA256,
        'driver_sha256':file_sha256(Path(__file__)),'evaluation_manifest':eval_manifest,
        'timing_note':'Epoch times include training/validation and data loading, exclude checkpoint writing and beam evaluation; concurrent jobs share hardware.'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'run_config.json').write_text(json.dumps(run_config,indent=2)+'\n',encoding='utf-8')
    if args.eval_only:
        evaluate(model,by_split,vocab,inverse,mult,model_args,args,device,out,eval_ids,checkpoint_sha)
        return
    train=by_split['train'][:args.max_train_samples or None];val=by_split['val'][:args.max_val_samples or None]
    collate=base.make_collate(vocab['<pad>'])
    train_ds=base.PTransDataset(train,vocab,mult,model_args['max_target_tokens'])
    val_ds=base.PTransDataset(val,vocab,mult,model_args['max_target_tokens'])
    train_loader=base.DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate)
    val_loader=base.DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    criterion=nn.CrossEntropyLoss(ignore_index=vocab['<pad>'],label_smoothing=args.label_smoothing)
    best=float('inf');best_epoch=0;stale=0;history=[];start=1;total_steps=0
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best=checkpoint['best_val_loss'];best_epoch=checkpoint['best_epoch'];stale=checkpoint['stale_epochs']
        history=checkpoint['history'];start=checkpoint['epoch']+1;total_steps=checkpoint['optimizer_steps_total']
        random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type=='cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in checkpoint['cuda_rng']])
    print(f'Records used: train={len(train)}, val={len(val)}; batch={args.batch_size}; workers={args.num_workers}',flush=True)
    if args.dry_run:
        model.train();optimizer.zero_grad(set_to_none=True)
        batch=base.move_batch(next(iter(train_loader)),device)
        loss=base.batch_loss(model,batch,criterion)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite dry-run loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Non-finite gradient')
        print(f'Dry run passed: loss={loss.item():.6f}; no optimizer update',flush=True)
        return
    for epoch in range(start,args.epochs+1):
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        sync(device);t0=time.perf_counter()
        train_loss,n,steps=run_loss(model,train_loader,criterion,optimizer,device,vocab['<pad>'])
        sync(device);t1=time.perf_counter()
        val_loss,_,_=run_loss(model,val_loader,criterion,None,device,vocab['<pad>'])
        sync(device);t2=time.perf_counter()
        total_steps+=steps
        entry={'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss,
            'train_seconds':t1-t0,'validation_seconds':t2-t1,'epoch_compute_seconds':t2-t0,
            'train_samples_per_second':n/max(t1-t0,1e-12),'optimizer_steps_epoch':steps,
            'optimizer_steps_total':total_steps,'learning_rate':optimizer.param_groups[0]['lr'],
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}
        history.append(entry)
        print(f'Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} '
              f'seconds={t2-t0:.1f} samples/s={entry["train_samples_per_second"]:.1f} updates={total_steps}',flush=True)
        improved=val_loss<best
        if improved:best=val_loss;best_epoch=epoch;stale=0
        else:stale+=1
        state={'format_version':FORMAT,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
            'vocab':vocab,'inverse_vocab':inverse,'multiplicity_vocab':mult,'model_args':model_args,
            'args':vars(args),'dataset_fingerprint':data_fingerprint,'epoch':epoch,'history':history,
            'best_epoch':best_epoch,'best_val_loss':best,'stale_epochs':stale,'optimizer_steps_total':total_steps,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        if improved:base.atomic_save(state,out/'best.pt')
        base.atomic_save(state,out/'last.pt')
        (out/'training_history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
        if stale>=args.patience:
            print(f'Early stopping: {stale} epochs without improvement',flush=True)
            break
    print(f'Best checkpoint: {out/"best.pt"} (epoch {best_epoch}, val_loss={best:.6f})',flush=True)
    if not args.skip_generation_eval:
        best_state=torch.load(out/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(best_state['model_state_dict'])
        evaluate(model,by_split,vocab,inverse,mult,model_args,args,device,out,eval_ids,file_sha256(out/'best.pt'))


if __name__=='__main__':main()
