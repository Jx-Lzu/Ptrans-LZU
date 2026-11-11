"""Frozen HC/HCP equal-update experiment and resumable candidate generation.

Imports original data/model/beam code; never edits or resumes historical runs.
"""
import argparse, csv, gzip, hashlib, json, math, os, platform, random, signal, sys, time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import rdBase
import ptrans_rbf as rbf
import ptrans_smiles_aug as aug

ROOT=Path(__file__).resolve().parent
FORMAT='ptrans_review_equalupdate_v1'
STOP=False

def interrupted(*_):
    global STOP
    STOP=True

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return rbf.digest(p)
def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def objsha(obj): return hashlib.sha256(canonical(obj).encode()).hexdigest()
def inside(p):
    v=(ROOT/p).resolve()
    if not v.is_relative_to(ROOT): raise ValueError('Path leaves project: '+str(p))
    return v
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f: torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def weight_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        v=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()
def stopped(out):
    if STOP or (out/'STOP_REQUESTED').exists() or (out.parent/'STOP_REQUESTED').exists():
        print('PAUSED: durable checkpoint/progress saved; start again to resume.',flush=True)
        raise SystemExit(75)
def env(device):
    return {'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
            'rdkit':rdBase.rdkitVersion,'cuda':torch.version.cuda,'device':device,
            'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'precision':'float32'}
def config(path):
    c=read(inside(path))
    if c['format']!=FORMAT or c['seeds']!=[42,7] or c['groups']!=['hc','hcp']:
        raise ValueError('This package pins two paired seeds and HC/HCP inputs')
    if [s['epochs'] for s in c['stages']]!=[100,10,6] or [s['lr'] for s in c['stages']]!=[3e-4,3e-5,1e-5] or c['batch_size']!=16:
        raise ValueError('Fixed production update budget changed; create a new protocol version')
    return c
def verify_files(c):
    m=read(inside('REVIEW_AUTORUN_FILES_v1.json'))
    for name,value in {**m['addon_files'],**m['required_project_files']}.items():
        if not inside(name).is_file() or sha(inside(name))!=value: raise ValueError('Missing/changed file: '+name)
    return {**m['addon_files'],**m['required_project_files']}
def context(c,load_cache=True):
    arch=rbf.load_arch(ROOT,'review_autorun_arch');base=arch.base
    records=base.load_records(inside(c['data']),inside(c['manifest']))
    by={s:[r for r in records if r['split_scaffold']==s] for s in ['train','val','test']}
    counts={s:len(v) for s,v in by.items()}
    if counts!=c['split_counts']: raise ValueError('Unexpected split counts: '+str(counts))
    v,iv,m=base.build_vocabs(by['train']);fp=base.fingerprint(records)
    if fp!=c['dataset_fingerprint'] or len(v)!=109: raise ValueError('Dataset or TRAIN vocabulary changed')
    centers,widths,ranges=rbf.fit_bases(records,base,read(inside(c['rbf_config'])))
    cache={}
    if load_cache:
        # The common TRAIN-only bank is chemically verified once during prepare.
        cache,meta,cv=aug.load_cache(inside(c['cache_data']),inside(c['cache_meta']),records,aug.token_pattern(ROOT),384,verify_structures=False)
        if cv!=iv: raise ValueError('Variant bank vocabulary differs')
    return {'arch':arch,'base':base,'records':records,'by':by,'vocab':v,'inverse_vocab':iv,'multiplicity_vocab':m,
            'model_args':{**c['model'],'rbf_centers':centers,'rbf_widths':widths},'ranges':ranges,'cache':cache}
def prepare(c,device):
    out=inside(c['output']);out.mkdir(parents=True,exist_ok=True)
    print('PREPARE checking pinned project files and environment',flush=True);sources=verify_files(c)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in '+sys.executable)
    if rdBase.rdkitVersion!=c['rdkit_version']:
        raise RuntimeError('Existing SMILES cache requires RDKit '+c['rdkit_version']+'; current='+rdBase.rdkitVersion+'. Install only this dependency with project .venv/bin/python -m pip install rdkit==2025.9.4, then repeat start.')
    torch.set_num_threads(c['cpu_threads']);environment=env(device)
    pp=out/'protocol.json'
    identity={'format':FORMAT,'config':c,'source_hashes':sources,'environment':environment,
              'augmentation':'One shared pre-existing TRAIN-only variant bank; target choices use seed/global_epoch/sample_id',
              'primary_checkpoint':'fixed global epoch116; 128992 optimizer updates',
              'secondary_selection':'full VAL Top1, Top5, Top10, then earlier continuation epoch, candidates0..6',
              'test_policy':'historical frozen four checkpoints only; new-model test generation is not scheduled'}
    if pp.exists():
        if read(pp)!=identity: raise ValueError('Environment/code/protocol changed. Preserve this run; restore its environment or create a separately versioned experiment.')
        print('PREPARE reused verified frozen protocol',flush=True);return
    print('PREPARE loading split and TRAIN vocabulary',flush=True);ctx=context(c)
    print('PREPARE checking existing TRAIN-only SMILES bank once; no model training',flush=True)
    aug.load_cache(inside(c['cache_data']),inside(c['cache_meta']),ctx['records'],aug.token_pattern(ROOT),384,verify_structures=True)
    dump(out/'prepared.json',{'dataset_fingerprint':c['dataset_fingerprint'],'vocabulary':ctx['inverse_vocab'],
                             'multiplicity_vocab':ctx['multiplicity_vocab'],'model_args':ctx['model_args'],
                             'training_ranges':ctx['ranges'],'all_variants_checked':True})
    dump(pp,identity)
    print('PREPARE passed: splits17782/2222/2222; vocab109; common TRAIN variant bank verified. No smoke training.',flush=True)
def masked_source(base,group):
    original=base.record_source;kept=('1H','13C') if group=='hc' else ('1H','13C','31P')
    def source(record,mult):
        nmr={n:record['nmr'].get(n,[]) for n in kept}
        if not any(nmr.values()): return [[0.]*7],[0],[mult['<unk>']]
        return original({'nmr':nmr},mult)
    return source
def dataset(ctx,rows,group,seed,epoch,train):
    base=ctx['base'];source=masked_source(base,group)
    class Data(base.PTransDataset):
        def __getitem__(self,i):
            r=self.records[i];sm=r['canonical_smiles'];kind='canonical'
            if train: sm,kind=aug.choose_target(r['sample_id'],sm,ctx['cache'][r['sample_id']],epoch,.5,seed)
            ts=base.tokenize_smiles(sm)
            if train and set(ts)-self.vocab.keys(): raise ValueError('TRAIN target contains unknown token')
            ids=[self.bos_id]+[self.vocab.get(t,self.unk_id) for t in ts]+[self.eos_id]
            if len(ids)>self.max_target_tokens: raise ValueError('Overlength target; no truncation allowed')
            f,n,m=source(r,self.multiplicity_vocab)
            return {'sample_id':r['sample_id'],'smiles':sm,'features':torch.tensor(f,dtype=torch.float32),
                    'nucleus_ids':torch.tensor(n),'mult_ids':torch.tensor(m),'target_ids':torch.tensor(ids)}
    return Data(rows,ctx['vocab'],ctx['multiplicity_vocab'],ctx['model_args']['max_target_tokens'])
def loader(ctx,rows,group,seed,epoch,train,c):
    # Independent generator prevents network/dropout RNG from changing paired sample order.
    ds=dataset(ctx,rows,group,seed,epoch,train)
    gen=torch.Generator().manual_seed(aug.seeded_integer(f'review-loader|{seed}|{epoch}|{train}'))
    return ctx['base'].DataLoader(ds,batch_size=c['batch_size'],shuffle=train,drop_last=False,num_workers=0,
                                 collate_fn=ctx['base'].make_collate(ctx['vocab']['<pad>']),generator=gen)
def loss_epoch(ctx,model,loader,criterion,opt,device):
    base=ctx['base'];model.train(opt is not None);total=tokens=steps=0;order=hashlib.sha256();targets=hashlib.sha256()
    with torch.set_grad_enabled(opt is not None):
        for batch in loader:
            for sid,sm in zip(batch['sample_ids'],batch['smiles']):
                order.update((sid+'\n').encode());targets.update((sid+'|'+sm+'\n').encode())
            b=base.move_batch(batch,device)
            if opt is not None: opt.zero_grad(set_to_none=True)
            loss=base.batch_loss(model,b,criterion)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            if opt is not None:
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            n=int(b['tgt_out'].ne(ctx['vocab']['<pad>']).sum());total+=loss.item()*n;tokens+=n;steps+=1
    return total/tokens,steps,order.hexdigest(),targets.hexdigest()
def phase(c,epoch):
    start=0
    for i,s in enumerate(c['stages']):
        if epoch<=start+s['epochs']:return i,s,epoch-start
        start+=s['epochs']
    raise ValueError('Epoch outside schedule')
def train(c,group,seed,device):
    root=inside(c['output']);protocol=read(root/'protocol.json');pid=objsha(protocol)
    ctx=context(c);base=ctx['base'];base.set_seed(seed);Model=rbf.model_class(ctx['arch'])
    model=Model(len(ctx['vocab']),len(ctx['multiplicity_vocab']),**ctx['model_args'],pad_id=ctx['vocab']['<pad>']).to(device).float()
    initial=weight_sha(model.state_dict());folder=root/f'{group}_s{seed}';folder.mkdir(exist_ok=True)
    run={'protocol_sha256':pid,'group':group,'seed':seed,'initial_weights_sha256':initial}
    if (folder/'run.json').exists() and read(folder/'run.json')!=run: raise ValueError('Run identity changed')
    dump(folder/'run.json',run)
    criterion=nn.CrossEntropyLoss(ignore_index=ctx['vocab']['<pad>'],label_smoothing=c['label_smoothing'])
    total_epochs=sum(s['epochs'] for s in c['stages']);per_epoch=math.ceil(len(ctx['by']['train'])/c['batch_size'])
    history=[];updates=0;start=1;opt=None;opt_phase=None
    if (folder/'last.pt').exists():
        s=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if s['format_version']!=FORMAT or s['run']!=run or s['updates']!=s['epoch']*per_epoch:raise ValueError('Invalid own-run resume state')
        model.load_state_dict(s['model_state_dict'],strict=True);history=s['history'];updates=s['updates'];start=s['epoch']+1
        opt_phase=s['phase'];opt=torch.optim.AdamW(model.parameters(),lr=c['stages'][opt_phase]['lr'],weight_decay=c['weight_decay']);opt.load_state_dict(s['optimizer_state_dict'])
        random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
        dump(folder/'training_history.json',history)
        print(f'RESUME {group} seed={seed} after epoch={s["epoch"]} updates={updates}',flush=True)
    for epoch in range(start,total_epochs+1):
        stopped(root);pi,ph,ep=phase(c,epoch)
        if pi!=opt_phase:
            opt=torch.optim.AdamW(model.parameters(),lr=ph['lr'],weight_decay=c['weight_decay']);opt_phase=pi
        tick=time.perf_counter()
        tl,steps,order,target=loss_epoch(ctx,model,loader(ctx,ctx['by']['train'],group,seed,epoch,True,c),criterion,opt,device)
        vl,_,_,_=loss_epoch(ctx,model,loader(ctx,ctx['by']['val'],group,seed,epoch,False,c),criterion,None,device)
        if steps!=per_epoch:raise ValueError('Unexpected updates per epoch')
        updates+=steps
        row={'epoch':epoch,'phase':ph['name'],'phase_epoch':ep,'learning_rate':ph['lr'],'train_loss':tl,'val_loss':vl,
             'updates':updates,'sample_order_sha256':order,'target_order_sha256':target,'seconds':time.perf_counter()-tick}
        history.append(row);weights=cpu_state(model)
        # Final raw, then all seven continuation candidates (fine final is epoch zero).
        raw_end=c['stages'][0]['epochs'];fine_end=raw_end+c['stages'][1]['epochs']
        if epoch==raw_end or epoch>=fine_end:
            name='raw_final.pt' if epoch==raw_end else f'continuation/epoch{epoch-fine_end:03d}.pt'
            save(folder/name,{'format_version':FORMAT,'run':run,'epoch':epoch,'updates':updates,'val_loss':vl,
                 'model_args':ctx['model_args'],'vocab':ctx['vocab'],'inverse_vocab':ctx['inverse_vocab'],
                 'multiplicity_vocab':ctx['multiplicity_vocab'],'model_state_dict':weights,'model_weights_sha256':weight_sha(weights)})
        state={'format_version':FORMAT,'run':run,'epoch':epoch,'phase':pi,'updates':updates,'history':history,
               'model_state_dict':weights,'optimizer_state_dict':opt.state_dict(),'python_rng':random.getstate(),
               'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None}
        save(folder/'last.pt',state);dump(folder/'training_history.json',history)
        print(f'TRAIN group={group} seed={seed} epoch={epoch:03d}/{total_epochs} phase={ph["name"]} lr={ph["lr"]} train={tl:.6f} val={vl:.6f} updates={updates} seconds={row["seconds"]:.1f}',flush=True)
        stopped(root)
    dump(folder/'train_complete.json',{'run':run,'epochs':total_epochs,'updates':updates,'primary_checkpoint':f'continuation/epoch{c["stages"][-1]["epochs"]:03d}.pt','primary_checkpoint_sha256':sha(folder/f'continuation/epoch{c["stages"][-1]["epochs"]:03d}.pt')})

def append_progress(path,row):
    row={**row,'sha256':objsha(row)}
    with path.open('a',encoding='utf-8',newline='\n') as f:f.write(canonical(row)+'\n');f.flush();os.fsync(f.fileno())
def progress(path,ids):
    if not path.exists():return []
    data=path.read_bytes()
    if data and not data.endswith(b'\n'):
        cut=data.rfind(b'\n')+1;tail=data[cut:]
        # Preserve only the incomplete final write before recovering its durable prefix.
        backup=path.with_name(path.name+f'.partial_{time.time_ns()}');backup.write_bytes(tail)
        with path.open('r+b') as f:f.truncate(cut)
        data=data[:cut]
    rows=[]
    for i,line in enumerate(data.decode().splitlines()):
        r=json.loads(line);digest=r.pop('sha256')
        if objsha(r)!=digest or i>=len(ids) or r['sample_id']!=ids[i]:raise ValueError('Candidate progress mismatch')
        rows.append(r)
    return rows
def generate(ctx,checkpoint,group,split,out,device,c,expected_sha=None):
    out.mkdir(parents=True,exist_ok=True);cp_sha=sha(checkpoint)
    if expected_sha and cp_sha!=expected_sha:raise ValueError('Frozen historical checkpoint changed')
    s=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if any(s[k]!=ctx[k] for k in ['vocab','inverse_vocab','multiplicity_vocab']):raise ValueError('Checkpoint vocabulary differs')
    if s['model_args']!=ctx['model_args']:raise ValueError('Checkpoint architecture differs')
    if 'nucleus_policy' in s and s['nucleus_policy']['group']!=group:raise ValueError('Wrong historical input group')
    rows=ctx['by'][split];ids=[r['sample_id'] for r in rows]
    pin={'checkpoint_sha256':cp_sha,'group':group,'split':split,'sample_ids':ids,'generation':c['generation'],
         'protocol_sha256':objsha(read(inside(c['output'])/'protocol.json'))}
    pp=out/'generation_protocol.json'
    if pp.exists() and read(pp)!=pin:raise ValueError('Generation output belongs to another protocol')
    dump(pp,pin)
    old=progress(out/'candidate_progress.jsonl',ids)
    model=None
    if len(old)<len(rows):
        Model=rbf.model_class(ctx['arch']);model=Model(len(ctx['vocab']),len(ctx['multiplicity_vocab']),**s['model_args'],pad_id=ctx['vocab']['<pad>']).to(device).float()
        model.load_state_dict(s['model_state_dict'],strict=True);model.eval()
    source=masked_source(ctx['base'],group);base=ctx['base'];tick=time.perf_counter()
    with torch.no_grad():
        for i in range(len(old),len(rows)):
            stopped(inside(c['output']));r=rows[i];f,n,m=source(r,ctx['multiplicity_vocab'])
            ft=torch.tensor([f],device=device,dtype=torch.float32);nt=torch.tensor([n],device=device);mt=torch.tensor([m],device=device);mask=torch.zeros(ft.shape[:2],device=device,dtype=torch.bool)
            candidates=base.beam_search_one(model,ft,nt,mt,mask,ctx['vocab'],ctx['inverse_vocab'],c['generation']['max_new_tokens'],c['generation']['beam_size'],c['generation']['length_penalty'])
            record={'sample_id':r['sample_id'],'canonical_smiles':r['canonical_smiles'],'candidates':[]}
            for sm,score in candidates:
                valid,match=base.structure_match(sm,r['canonical_smiles'])
                record['candidates'].append({'smiles':sm,'score':score,'valid':bool(valid),'exact':bool(match)})
            append_progress(out/'candidate_progress.jsonl',record);old.append(record)
            if (i+1)%64==0 or i+1==len(rows):print(f'GENERATE group={group} split={split} {i+1}/{len(rows)}',flush=True)
    hits={str(k):sum(any(x['exact'] for x in r['candidates'][:k]) for r in old) for k in [1,5,10]}
    n=len(old);total=sum(len(r['candidates']) for r in old);valid=sum(sum(x['valid'] for x in r['candidates']) for r in old)
    metrics={'samples':n,'split':split,'checkpoint_sha256':cp_sha,'hits':hits,**{'top'+k:v/n for k,v in hits.items()},
             'candidate_valid_rate':valid/total if total else 0.,'completed_candidates':total,'beam_size':c['generation']['beam_size'],
             'generation_seconds_this_invocation':time.perf_counter()-tick,'historical_test_previously_inspected':split=='test'}
    csvpath=out/'beam_predictions.csv';tmp=csvpath.with_name(csvpath.name+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','split','rank','true_canonical_smiles','predicted_smiles','beam_score_length_normalized','valid_smiles','structure_exact_match']);w.writeheader()
        for row in old:
            for rank,x in enumerate(row['candidates'] or [{'smiles':'','score':None,'valid':False,'exact':False}],1):
                w.writerow(dict(sample_id=row['sample_id'],split=split,rank=rank if row['candidates'] else 0,true_canonical_smiles=row['canonical_smiles'],predicted_smiles=x['smiles'],beam_score_length_normalized=x['score'],valid_smiles=x['valid'],structure_exact_match=x['exact']))
    os.replace(tmp,csvpath);metrics['predictions_sha256']=sha(csvpath);dump(out/'metrics.json',metrics)
    print('RESULT '+canonical(metrics),flush=True);return metrics
def evaluate_new(c,group,seed,device):
    root=inside(c['output']);folder=root/f'{group}_s{seed}';ctx=context(c,False)
    receipt=read(folder/'train_complete.json')
    if sha(folder/receipt['primary_checkpoint'])!=receipt['primary_checkpoint_sha256']:raise ValueError('Fixed final checkpoint changed')
    all_metrics=[]
    for e in range(c['stages'][-1]['epochs']+1):
        m=generate(ctx,folder/f'continuation/epoch{e:03d}.pt',group,'val',folder/f'val_epoch{e:03d}',device,c)
        all_metrics.append({'continuation_epoch':e,'checkpoint':f'continuation/epoch{e:03d}.pt','metrics':m})
    selected=max(all_metrics,key=lambda x:(*[x['metrics']['hits'][str(k)] for k in [1,5,10]],-x['continuation_epoch']))
    dump(folder/'validation_complete.json',{'primary_fixed_final':all_metrics[-1],'secondary_full_val_selected':selected,'all_epochs':all_metrics,'new_model_test_evaluated':False})
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['prepare','native','train'],required=True)
    p.add_argument('--group',choices=['hc','hcp']);p.add_argument('--seed',type=int);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args();c=config('configs/review_autorun_v1.json');torch.set_num_threads(c['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if a.task=='prepare':prepare(c,a.device);return
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if a.group not in c['groups'] or a.seed not in c['seeds']:raise ValueError('Unknown matrix run')
    # A surviving worker keeps this OS lock even if its scheduler is killed.
    # A second start cannot train/write the same experiment concurrently.
    lock=None
    if os.name=='posix':
        import fcntl
        locks=inside(c['output'])/'worker_locks';locks.mkdir(exist_ok=True)
        lock=(locks/f'{a.task}_{a.group}_s{a.seed}.lock').open('a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This worker is still active; do not launch a duplicate')
    try:
        if a.task=='native':
            key=f'{a.group}_s{a.seed}';item=c['historical'][key]
            generate(context(c,False),inside(item['checkpoint']),a.group,'test',inside(c['output'])/'historical_test'/key,a.device,c,item['sha256'])
        else:train(c,a.group,a.seed,a.device);evaluate_new(c,a.group,a.seed,a.device)
    finally:
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
