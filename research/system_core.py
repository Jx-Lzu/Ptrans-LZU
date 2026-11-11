"""Frozen original three-generator assembly. NMR masking applies to every learned stage."""
from pathlib import Path
from functools import lru_cache
import hashlib,json,math
import numpy as np
import torch
from torch.nn import functional as F
from rdkit import Chem,rdBase,RDLogger
from rdkit.Chem import rdFingerprintGenerator,Descriptors
import ptrans_rbf as rbf
from ptrans_score_tokens_v2 import compatible_tokens
from frozen_heads import CompatibilityRanker,ElementCountHead,candidate_features,standardize,count_prediction,count_penalty,element_counts,peak_features

ROOT=Path(__file__).resolve().parent
GROUPS={'hcp':['1H','13C','31P'],'hc':['1H','13C'],'hp':['1H','31P'],'cp':['13C','31P'],'c':['13C'],'h':['1H']}
ELEMENTS=['C','N','O','S','P','F','Cl','Br','I']
RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable_json(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def object_sha(x):return hashlib.sha256(stable_json(x).encode()).hexdigest()
def weights_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        v=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()
def masked_nmr(record,group):
    if group not in GROUPS:raise ValueError('Unsupported nuclear input group')
    nmr={n:record['nmr'].get(n,[]) for n in GROUPS[group]}
    if not any(nmr.values()):raise ValueError('Retained nuclear input contains no peaks')
    return nmr
@lru_cache(maxsize=100000)
def canonical(sm):
    mol=Chem.MolFromSmiles(sm) if sm else None
    return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True) if mol is not None and mol.GetNumAtoms() else None
def valid_proposals(proposals,source):
    pool=[];seen=set()
    for i,(sm,score) in enumerate(proposals,1):
        key=canonical(sm)
        if key is None or key in seen:continue
        if not math.isfinite(score):raise ValueError('Nonfinite generation score')
        seen.add(key);pool.append({'predicted_smiles':sm,'canonical_smiles':key,'source':source,'source_rank':i})
    return pool
def interleave(primary,secondary,prefer_primary=True):
    representatives={c['canonical_smiles']:c for c in primary};out=[];seen=set()
    for i in range(max(len(primary),len(secondary))):
        for pool in [primary,secondary]:
            if i>=len(pool):continue
            c=pool[i];key=c['canonical_smiles']
            if key in seen:continue
            seen.add(key);out.append(dict(representatives.get(key,c) if prefer_primary else c))
    return out
def expanded_pool(a10,b10,a30,b30):
    narrow=interleave(valid_proposals(a10,'A10'),valid_proposals(b10,'B10'),True)
    # Wide expansion interleaves RAW beam positions before filtering, exactly as
    # the original proposed_union helper. Narrow merging instead filters each
    # source first and prefers the primary spelling, as its original helper did.
    wide=[];wide_seen=set()
    for i in range(max(len(a30),len(b30))):
        for source,proposals in [('A30',a30),('B30',b30)]:
            if i>=len(proposals):continue
            sm,score=proposals[i]
            if not math.isfinite(score):raise ValueError('Nonfinite wide-beam score')
            key=canonical(sm)
            if key is None or key in wide_seen:continue
            wide_seen.add(key);wide.append({'predicted_smiles':sm,'canonical_smiles':key,'source':source,'source_rank':i+1})
    pool=[dict(c,retained_from_narrow=True) for c in narrow];seen={c['canonical_smiles'] for c in narrow}
    for c in wide:
        if c['canonical_smiles'] not in seen:seen.add(c['canonical_smiles']);pool.append(dict(c,retained_from_narrow=False))
    for i,c in enumerate(pool,1):c['original_rank']=i
    return pool
def mass_priority(pool,mass,ppm):
    if not math.isfinite(mass) or mass<=0:raise ValueError('Need a finite positive known theoretical mass')
    scored=[]
    for i,c in enumerate(pool,1):
        mol=Chem.MolFromSmiles(c['predicted_smiles']);assert mol is not None and mol.GetNumAtoms()
        value=float(Descriptors.ExactMolWt(mol));error=(value-mass)/mass*1e6
        scored.append({**c,'pre_mass_rank':i,'candidate_exact_mass':value,'mass_error_ppm':error,'mass_match':abs(error)<=ppm})
    scored.sort(key=lambda c:not c['mass_match'])
    for i,c in enumerate(scored,1):c['final_rank']=i
    return scored

class Generator:
    def __init__(self,arch,path,device,expected):
        if sha(path)!=expected['compact_sha256']:raise ValueError('Generator weight file differs')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['source_checkpoint_sha256']!=expected['source_sha256'] or weights_sha(state['model_state_dict'])!=expected['tensor_sha256']:raise ValueError('Generator tensor provenance differs')
        self.base=arch.base;self.device=device;self.vocab=state['vocab'];self.inverse=state['inverse_vocab'];self.mult=state['multiplicity_vocab'];self.args=state['model_args']
        args={**self.args,'rbf_centers':state['rbf_encoding']['centers'],'rbf_widths':state['rbf_encoding']['widths']}
        self.model=rbf.model_class(arch)(len(self.vocab),len(self.mult),**args,pad_id=self.vocab['<pad>']).to(device).float().eval().requires_grad_(False)
        self.model.load_state_dict(state['model_state_dict'],strict=True)
    def tensors(self,nmr):
        f,n,m=self.base.record_source({'nmr':nmr},self.mult)
        ft=torch.tensor([f],device=self.device,dtype=torch.float32);nt=torch.tensor([n],device=self.device);mt=torch.tensor([m],device=self.device)
        return ft,nt,mt,torch.zeros(ft.shape[:2],device=self.device,dtype=torch.bool)
    @torch.no_grad()
    def generate(self,nmr,beam,limit=383):
        return self.base.beam_search_one(self.model,*self.tensors(nmr),self.vocab,self.inverse,limit,beam,.7)
    @torch.no_grad()
    def score(self,nmr,smiles,batch_size=16):
        if not smiles:return []
        sequences=[]
        for sm in smiles:
            tokens,_=compatible_tokens(sm,self.vocab,self.base.tokenize_smiles)
            ids=[self.vocab['<bos>']]+[self.vocab[t] for t in tokens]+[self.vocab['<eos>']]
            if len(ids)>self.args['max_target_tokens']:raise ValueError('Scoring would truncate a candidate')
            sequences.append(ids)
        ft,nt,mt,mask=self.tensors(nmr);memory=self.model.encode(ft,nt,mt,mask);result=[];pad=self.vocab['<pad>']
        for start in range(0,len(sequences),batch_size):
            ss=sequences[start:start+batch_size];tt=torch.full((len(ss),max(map(len,ss))),pad,dtype=torch.long,device=self.device)
            for i,x in enumerate(ss):tt[i,:len(x)]=torch.tensor(x,device=self.device)
            ti,to=tt[:,:-1],tt[:,1:];keep=to.ne(pad)
            logits=self.model.decode(memory.expand(len(ss),-1,-1),mask.expand(len(ss),-1),ti,ti.eq(pad))
            selected=F.log_softmax(logits,-1).gather(-1,to.unsqueeze(-1)).squeeze(-1)
            score=selected.masked_fill(~keep,0.).sum(1)/((5.+keep.sum(1).float())/6.).pow(.7)
            if not torch.isfinite(score).all():raise ValueError('Nonfinite teacher-forced score')
            result.extend(score.cpu().tolist())
        return result

class FrozenAssembly:
    def __init__(self,device='cuda',root=ROOT,allow_unpinned=False):
        self.root=Path(root);self.config=read(self.root/'config.json');self.device=device
        if [self.config[k] for k in ['alpha','beta','gamma','mass_tolerance_ppm','generation_limit','length_penalty']]!=[.6,.25,.25,5.,383,.7]:raise ValueError('Keep the frozen scoring/generation policy')
        if rdBase.rdkitVersion!='2025.09.4':raise ValueError('RDKit2025.09.4 required')
        if not allow_unpinned and str(torch.__version__)!='2.6.0+cu124':raise ValueError('Use torch2.6.0+cu124')
        if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.arch=rbf.load_arch(self.root,'frozen_complete_arch');spec=self.config['models']
        self.a=Generator(self.arch,self.root/spec['primary']['path'],device,spec['primary'])
        self.b=Generator(self.arch,self.root/spec['secondary']['path'],device,spec['secondary'])
        self.extra=Generator(self.arch,self.root/spec['extra']['path'],device,spec['extra'])
        assert self.a.vocab==self.b.vocab==self.extra.vocab
        self.ranker,self.ranker_state=self.load_head(spec['ranker'],CompatibilityRanker)
        self.element,self.element_state=self.load_head(spec['element'],ElementCountHead)
        self.fp_generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
        self.features= lru_cache(maxsize=30000)(lambda sm:candidate_features(sm,self.fp_generator))
        self.counts=lru_cache(maxsize=30000)(lambda sm:element_counts(sm,ELEMENTS))
    def load_head(self,spec,cls):
        p=self.root/spec['path']
        if sha(p)!=spec['compact_sha256']:raise ValueError('Head file differs')
        s=torch.load(p,map_location='cpu',weights_only=False)
        if s['source_checkpoint_sha256']!=spec['source_sha256'] or weights_sha(s['model_state_dict'])!=spec['tensor_sha256']:raise ValueError('Head tensor provenance differs')
        m=cls(**s['model_args']).to(self.device).float().eval().requires_grad_(False);m.load_state_dict(s['model_state_dict'],strict=True)
        return m,s
    @torch.no_grad()
    def spectral_features(self,records,group,batch_size=64):
        result=[];base=self.a.base
        for start in range(0,len(records),batch_size):
            items=[]
            for r in records[start:start+batch_size]:
                f,n,m=base.record_source({'nmr':masked_nmr(r,group)},self.a.mult)
                items.append({'sample_id':r['sample_id'],'smiles':'','features':torch.tensor(f,dtype=torch.float32),'nucleus_ids':torch.tensor(n),'mult_ids':torch.tensor(m),'target_ids':torch.tensor([1,2])})
            batch=base.move_batch(base.make_collate(0)(items),self.device)
            mem=self.a.model.encode(batch['src_features'],batch['src_nucleus'],batch['src_mult'],batch['src_pad_mask']);keep=~batch['src_pad_mask'];parts=[]
            for idx in [None,0,1,2]:
                mask=keep if idx is None else keep&batch['src_nucleus'].eq(idx)
                parts.append(mem.masked_fill(~mask.unsqueeze(-1),0.).sum(1)/mask.sum(1).clamp_min(1).unsqueeze(-1))
            v=torch.cat(parts,-1)
            if not torch.isfinite(v).all():raise ValueError('Nonfinite frozen spectral features')
            result.append(v.cpu().numpy())
        return np.concatenate(result)
    @torch.no_grad()
    def score_pool(self,nmr,pool,spectrum):
        if not pool:return []
        smiles=[x['predicted_smiles'] for x in pool]
        narrow=sum(bool(c['retained_from_narrow']) for c in pool)
        assert all(c['retained_from_narrow'] for c in pool[:narrow]) and not any(c['retained_from_narrow'] for c in pool[narrow:])
        # Preserve the historical narrow-pool score batching; score newly added
        # wide-beam candidates separately, as in the cached expansion workflow.
        pa=np.asarray(self.a.score(nmr,smiles[:narrow],self.config['score_batch_size'])+self.a.score(nmr,smiles[narrow:],self.config['score_batch_size']),dtype=np.float64)
        pb=np.asarray(self.b.score(nmr,smiles[:narrow],self.config['score_batch_size'])+self.b.score(nmr,smiles[narrow:],self.config['score_batch_size']),dtype=np.float64)
        base=.6*pa+.4*pb
        data=[self.features(sm) for sm in smiles]
        for c,v in zip(pool,data):assert c['canonical_smiles']==v[0]
        fp=torch.tensor(np.stack([v[1] for v in data])[None],device=self.device,dtype=torch.float32)
        desc=torch.tensor(np.stack([v[2] for v in data])[None],device=self.device,dtype=torch.float32)
        spectral=torch.tensor(spectrum[None],device=self.device,dtype=torch.float32)
        compat=self.ranker(spectral,fp,desc).squeeze(0).cpu().numpy()
        peaks=peak_features({'nmr':nmr});peak_tensor=torch.tensor(peaks[None],device=self.device,dtype=torch.float32)
        predicted=count_prediction(self.element,spectral,peak_tensor,self.element_state['normalization'])
        sigma=torch.tensor(self.element_state['residual_log_scales'],device=self.device,dtype=torch.float32)
        active=torch.tensor(self.element_state['normalization']['active'],device=self.device,dtype=torch.float32)
        counts=torch.tensor(np.stack([self.counts(sm) for sm in smiles]),device=self.device,dtype=torch.float32)
        penalty=count_penalty(predicted,counts,sigma,active).cpu().numpy()
        frozen=standardize(base)+.25*standardize(compat);scores=frozen+.25*standardize(penalty)
        if not np.isfinite(scores).all():raise ValueError('Nonfinite assembly score')
        rows=[{**c,'primary_score':float(a),'secondary_score':float(b),'ranker_score':float(rr),'element_score':float(e),'assembled_score':float(s)} for c,a,b,rr,e,s in zip(pool,pa,pb,compat,penalty,scores)]
        return sorted(rows,key=lambda c:(-c['assembled_score'],c['original_rank']))
    @torch.no_grad()
    def predict(self,nmr,known_mass,spectrum,proposals=None):
        # nmr is already masked. No reference structure/formula or reference rank enters here.
        if proposals is None:
            proposals={'A10':self.a.generate(nmr,10),'B10':self.b.generate(nmr,10),'A30':self.a.generate(nmr,30),'B30':self.b.generate(nmr,30),'E10':self.extra.generate(nmr,10)}
        pool=expanded_pool(proposals['A10'],proposals['B10'],proposals['A30'],proposals['B30'])
        scored=self.score_pool(nmr,pool,spectrum)
        extra=valid_proposals(proposals['E10'],'encoder5')
        combined=interleave(scored,extra,True)
        ranked=mass_priority(combined,known_mass,5.)
        return {'current_pool':scored,'pre_mass_candidates':combined,'final_candidates':ranked,
                'observed_log_peak_counts':peak_features({'nmr':nmr}).tolist(),'raw_proposals':proposals}
