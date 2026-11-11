"""Shared protocol and train-only data utilities for six independently trained input groups."""
from pathlib import Path
from functools import lru_cache
import csv,gzip,json,hashlib,math,os,random,signal,sys,time
import numpy as np
import torch
import ptrans_review_worker_v1 as original
import ptrans_rbf as rbf
import ptrans_smiles_aug as aug
from system_core import GROUPS,masked_nmr,Generator,FrozenAssembly,ELEMENTS,canonical,weights_sha,object_sha,stable_json
from frozen_heads import CompatibilityRanker,ElementCountHead,candidate_features,element_counts,peak_features
from rdkit import rdBase
from rdkit.Chem import rdFingerprintGenerator
ROOT=Path(__file__).resolve().parent;STOP=False
read=original.read;dump=original.dump;save=original.save;sha=original.sha;cpu_state=original.cpu_state
def settings():
    c=read(ROOT/'config.json')
    if c['format']!='ptrans_retrain6_fullsystem_v1' or c['groups']!=['hcp','hc','hp','cp','c','h']:raise ValueError('Keep the declared six-input protocol')
    if c['component_seeds']!=[42,7] or c['selection']!='minimum_full_validation_loss':raise ValueError('Wrong seed/selection policy')
    if [s['epochs'] for s in c['stages']]!=[100,10,6] or c['batch_size']!=16:raise ValueError('Fixed production training budget changed')
    if [s['lr'] for s in c['stages']]!=[3e-4,3e-5,1e-5] or c['generation']!={'beam_size':10,'wide_beam_size':30,'length_penalty':.7,'max_new_tokens':383}:raise ValueError('Generation/training policy changed')
    if c['scoring']!={'alpha':.6,'beta':.25,'gamma':.25,'mass_tolerance_ppm':5.}:raise ValueError('Keep fixed score weights')
    return c
def out(c):return ROOT/c['output']
def groupdir(c,g):return out(c)/g
def stopped(c):
    if STOP or (out(c)/'STOP_REQUESTED').exists():print('PAUSED after durable checkpoint/progress.',flush=True);raise SystemExit(75)
def interrupted(*_):
    global STOP;STOP=True
def initialize_runtime(c,device):
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if str(torch.__version__)!=c['torch'] or np.__version__!=c['numpy'] or rdBase.rdkitVersion!=c['rdkit']:raise RuntimeError('Use the pinned torch/numpy/RDKit environment')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(c['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def verify_files():
    m=read(ROOT/'FILES.sha256.json')
    for name,x in m.items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise ValueError('Missing/changed package input: '+name)
    return {n:x['sha256'] for n,x in m.items()}
def context(c,load_cache=True):return original.context(c,load_cache)
def protocol_id(c):return object_sha(read(out(c)/'protocol.json'))
def common_cache(c,ctx):
    meta=read(ROOT/c['common_library_meta']);path=ROOT/c['common_library']
    if sha(path)!=meta['arrays_sha256'] or meta['dataset_fingerprint']!=c['dataset_fingerprint']:raise ValueError('Common TRAIN library changed')
    ids=[r['sample_id'] for r in ctx['by']['train']]
    if meta['sample_ids']!=ids:raise ValueError('Common library is not aligned with TRAIN')
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if set(a)!={'fingerprints','descriptors','positive','hard','element_counts','fit_indices','calibration_indices'}:raise ValueError('Do not reuse old NMR features')
    lib=meta['library_smiles'];n=len(ids)
    if a['fingerprints'].shape!=(len(lib),2048) or a['descriptors'].shape!=(len(lib),16) or a['positive'].shape!=(n,) or a['hard'].shape!=(n,8):raise ValueError('Invalid common library shapes')
    if any(lib[int(a['positive'][i])]!=r['canonical_smiles'] for i,r in enumerate(ctx['by']['train'])) or (a['hard']==a['positive'][:,None]).any():raise ValueError('False negative or shifted TRAIN positive')
    if any(not np.isfinite(x).all() for x in a.values()):raise ValueError('Nonfinite common library')
    if sorted(np.r_[a['fit_indices'],a['calibration_indices']].tolist())!=list(range(n)):raise ValueError('Count-head internal split differs')
    fit={lib[int(a['positive'][i])] for i in a['fit_indices']};cal={lib[int(a['positive'][i])] for i in a['calibration_indices']}
    if fit&cal:raise ValueError('Duplicate molecules cross count-head fit/calibration partitions')
    return a,meta
def prepare(c,device):
    sources=verify_files();initialize_runtime(c,device);ctx=context(c,False);common_cache(c,ctx)
    identity={'config':c,'source_hashes':sources,'environment':original.env(device),'training_records':len(ctx['by']['train']),
      'selected_checkpoint_policy':'Train the full allocated116epochs; choose the minimum full-VAL teacher-forced loss checkpoint, earlier on ties. Keep final separately.',
      'comparison_scope':'One complete assembly per nuclear group; component seeds42/7 are not independent whole-system repetitions.',
      'test_policy':'All configuration/selection rules frozen before runs; test starts only after all six systems have finished training and full validation.',
      'mass_policy':'Known theoretical mass used only in postprocessing; one generated candidate pool supplies both conditions.'}
    p=out(c)/'protocol.json'
    if p.exists() and read(p)!=identity:raise RuntimeError('Output belongs to another environment/configuration; preserve it and restore its environment')
    dump(p,identity);dump(out(c)/'prepared.json',{'model_args':ctx['model_args'],'vocabulary':ctx['inverse_vocab'],'multiplicity_vocab':ctx['multiplicity_vocab'],'training_RBF_ranges':ctx['ranges']})
    print('PREPARED six groups;18generators+6rankers+6count-heads; no old trained weights used.',flush=True)
def loader(ctx,rows,g,seed,epoch,train,c):
    base=ctx['base']
    class Data(base.PTransDataset):
        def __getitem__(self,i):
            r=self.records[i];sm=r['canonical_smiles']
            if train:sm,_=aug.choose_target(r['sample_id'],sm,ctx['cache'][r['sample_id']],epoch,.5,seed)
            ts=base.tokenize_smiles(sm)
            if train and set(ts)-self.vocab.keys():raise ValueError('TRAIN augmentation introduces an unknown token')
            ids=[self.bos_id]+[self.vocab.get(t,self.unk_id) for t in ts]+[self.eos_id]
            if len(ids)>self.max_target_tokens:raise ValueError('Target would be truncated')
            f,n,m=base.record_source({'nmr':masked_nmr(r,g)},self.multiplicity_vocab)
            return {'sample_id':r['sample_id'],'smiles':sm,'features':torch.tensor(f,dtype=torch.float32),'nucleus_ids':torch.tensor(n),'mult_ids':torch.tensor(m),'target_ids':torch.tensor(ids)}
    ds=Data(rows,ctx['vocab'],ctx['multiplicity_vocab'],ctx['model_args']['max_target_tokens'])
    rng=torch.Generator().manual_seed(aug.seeded_integer(f'retrain6-loader|{seed}|{epoch}|{train}'))
    return base.DataLoader(ds,batch_size=c['batch_size'],shuffle=train,drop_last=False,num_workers=0,collate_fn=base.make_collate(ctx['vocab']['<pad>']),generator=rng)
def rng_state(device):return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if device=='cuda' else None}
def restore_rng(s,device):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if device=='cuda':torch.cuda.set_rng_state_all(s['cuda'])
def compact_checkpoint(source,dest):
    s=torch.load(source,map_location='cpu',weights_only=False);state={k:s[k] for k in ['model_state_dict','model_args']}
    for k in ['vocab','inverse_vocab','multiplicity_vocab','rbf_encoding','normalization','residual_log_scales']:
        if k in s:state[k]=s[k]
    state['source_checkpoint_sha256']=sha(source);save(dest,state)
    return {'path':str(dest.relative_to(ROOT)).replace('\\','/'),'compact_sha256':sha(dest),'source_sha256':sha(source),'tensor_sha256':weights_sha(state['model_state_dict'])}
def generator_spec(c,g,role):
    folder=groupdir(c,g)/'generators'/role;receipt=read(folder/'complete.json')
    source=folder/'best.pt'
    if receipt['protocol_sha256']!=protocol_id(c) or receipt['group']!=g or receipt['role']!=role or sha(source)!=receipt['selected_sha256']:raise ValueError('Selected generator changed')
    return read(folder/'inference_spec.json')
def load_generator(c,g,role,device,arch=None):
    spec=generator_spec(c,g,role);arch=arch or rbf.load_arch(ROOT,'retrain6_generator_arch')
    return Generator(arch,ROOT/spec['path'],device,spec)
class TrainedAssembly(FrozenAssembly):
    def __init__(self,c,g,device,load_heads=True,load_extra=True):
        self.root=ROOT;self.config={'score_batch_size':c['score_batch_size']};self.device=device
        self.arch=rbf.load_arch(ROOT,'retrain6_assembly_arch')
        self.a=load_generator(c,g,'primary',device,self.arch);self.b=load_generator(c,g,'secondary',device,self.arch)
        if load_extra:self.extra=load_generator(c,g,'extra',device,self.arch)
        if load_heads:
            for role,cls in [('ranker',CompatibilityRanker),('element',ElementCountHead)]:
                folder=groupdir(c,g)/role;receipt=read(folder/'complete.json')
                if receipt['protocol_sha256']!=protocol_id(c) or receipt['group']!=g or receipt['kind']!=role or sha(folder/'selected.pt')!=receipt['selected_sha256']:raise ValueError('Head protocol/selected weight changed')
                spec=read(folder/'inference_spec.json');model,state=self.load_head(spec,cls);setattr(self,role,model);setattr(self,role+'_state',state)
        self.fp_generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
        self.features=lru_cache(maxsize=30000)(lambda sm:candidate_features(sm,self.fp_generator))
        self.counts=lru_cache(maxsize=30000)(lambda sm:element_counts(sm,ELEMENTS))
def append_progress(path,row):original.append_progress(path,row)
def read_progress(path,ids):return original.progress(path,ids)
def write_csv(p,rr):
    if not rr:return
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
