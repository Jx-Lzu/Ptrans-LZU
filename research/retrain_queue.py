"""Linux queue and portable status/export. Status needs only Python's standard library."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,signal,subprocess,sys,time,zipfile
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def objsha(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def config():return read(ROOT/'config.json')
def output():return ROOT/config()['output']
def jobs(c):
    rows=[{'id':'prepare','kind':'prepare','deps':[]}]
    for g in c['groups']:
        for role in c['roles']:rows.append({'id':f'{g}_gen_{role}','kind':'generator','group':g,'role':role,'deps':['prepare']})
        rows.append({'id':g+'_cache','kind':'cache','group':g,'deps':[g+'_gen_primary',g+'_gen_secondary']})
        for kind in ['ranker','element']:rows.append({'id':g+'_'+kind,'kind':kind,'group':g,'deps':[g+'_cache']})
        rows.append({'id':g+'_val','kind':'val','group':g,'deps':[g+'_gen_'+r for r in c['roles']]+[g+'_ranker',g+'_element']})
    rows.append({'id':'freeze_test','kind':'freeze','deps':[g+'_val' for g in c['groups']]})
    for g in c['groups']:rows.append({'id':g+'_test','kind':'test','group':g,'deps':['freeze_test']})
    return rows
def receipt(out,j):return out/'jobs'/(j['id']+'.json')
def finished(out,j):return receipt(out,j).exists()
def lease(path):
    import fcntl
    path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
    except BlockingIOError:f.close();return None
def busy(out,identifier):
    if os.name!='posix':return False
    f=lease(out/'locks'/(identifier+'.lock'))
    if f is None:return True
    f.close();return False
def validate_receipts(c,out):
    for p in (out/'jobs').glob('*.json'):
        r=read(p)
        if r['config_sha256']!=objsha(c):raise ValueError('Configuration changed since '+p.stem)
        if (out/'protocol.json').exists() and r['protocol_sha256']!=objsha(read(out/'protocol.json')):raise ValueError('Protocol mismatch: '+p.stem)
        for name,m in r['outputs'].items():
            q=(ROOT/name).resolve()
            if not q.is_relative_to(ROOT) or not q.is_file() or q.stat().st_size!=m['bytes'] or digest(q)!=m['sha256']:raise ValueError('Saved task output changed: '+name)
def verify_package():
    for name,m in read(ROOT/'FILES.sha256.json').items():
        q=(ROOT/name).resolve()
        if not q.is_relative_to(ROOT) or not q.is_file() or q.stat().st_size!=m['bytes'] or digest(q)!=m['sha256']:raise ValueError('Package input changed: '+name)
def stage_state(out,state,running=None,error=None):dump(out/'queue_state.json',{'state':state,'running':running or [],'error':error,'updated':datetime.datetime.now().isoformat(timespec='seconds')})

def run(parallel):
    if os.name!='posix':raise RuntimeError('Training queue targets Ubuntu/Linux; Windows can inspect/export results.')
    c=config();out=output();qlease=lease(out/'queue.lock')
    if qlease is None:print('Queue already active.');return
    running={};failed=None
    def stop(*_):(out/'STOP_REQUESTED').touch()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        verify_package();validate_receipts(c,out)
        available=json.loads(subprocess.check_output([sys.executable,'-c','import json,torch; print(json.dumps(torch.cuda.mem_get_info()[0]/1024**3))'],text=True))
        capacity=1 if available<12 else parallel
        print(f'QUEUE capacity={capacity}; free_GPU_GiB={available:.1f};18generators,12heads; minimum-full-VAL-loss selection.',flush=True)
        tasks=jobs(c)
        while True:
            for jid,(proc,log) in list(running.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del running[jid];print(f'JOB_EXIT {jid} code={code}',flush=True)
                if code not in [0,75,76]:failed=f'{jid} exit={code}; see logs/{jid}.log';stop()
                elif code==0 and not (out/'jobs'/(jid+'.json')).exists():failed=f'{jid} did not write its completion receipt';stop()
                elif code==0:
                    if jid.endswith('_val') or jid.endswith('_test'):summary(out)
            external=[j['id'] for j in tasks if j['id'] not in running and not finished(out,j) and busy(out,j['id'])]
            if (out/'STOP_REQUESTED').exists():
                stage_state(out,'pausing',list(running)+external,failed)
                if not running and not external:
                    stage_state(out,'failed' if failed else 'paused',error=failed);print(f'QUEUE_{"FAILED" if failed else "PAUSED"} {failed or "restart to resume"}',flush=True);return
            else:
                done={j['id'] for j in tasks if finished(out,j)}
                if len(done)==len(tasks):
                    subprocess.run([sys.executable,str(ROOT/'analyze_results.py')],cwd=ROOT,check=True)
                    summary(out);export(out);stage_state(out,'completed');print('RETRAIN6_COMPLETE '+str(out),flush=True);return
                # Earliest group first; any vacant slot is filled immediately if dependencies permit.
                for j in tasks:
                    if len(running)+len(external)>=capacity:break
                    if j['id'] in done or j['id'] in running or j['id'] in external or not set(j['deps'])<=done:continue
                    (out/'logs').mkdir(parents=True,exist_ok=True);log=(out/'logs'/(j['id']+'.log')).open('a',encoding='utf-8')
                    args=[sys.executable,'-u',str(ROOT/'retrain_worker.py'),j['kind'],'--job-id',j['id']]
                    for k in ['group','role']:
                        if k in j:args.extend(['--'+k,j[k]])
                    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(c['cpu_threads']),MKL_NUM_THREADS=str(c['cpu_threads']),PYTHONUNBUFFERED='1')
                    proc=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env);running[j['id']]=(proc,log)
                    print('JOB_START '+j['id']+' pid='+str(proc.pid),flush=True)
                stage_state(out,'running',list(running)+external)
            time.sleep(3)
    except BaseException as e:
        (out/'STOP_REQUESTED').touch();stage_state(out,'failed',list(running),str(e));raise
    finally:qlease.close()

def start(parallel):
    out=output();out.mkdir(parents=True,exist_ok=True)
    if os.name!='posix':raise RuntimeError('Use this start command on the cloud Ubuntu machine')
    if busy_queue(out):print('Queue already active.');status(out);return
    (out/'STOP_REQUESTED').unlink(missing_ok=True)
    with (out/'matrix.log').open('a',encoding='utf-8') as f:
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--parallel',str(parallel)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    dump(out/'launcher.json',{'pid':p.pid,'python':sys.executable,'parallel':parallel});print(f'Retraining queue launched PID={p.pid}\nLog: {out/"matrix.log"}\nRepeat start after interruption to resume; status shows all groups.')
def busy_queue(out):
    if os.name!='posix':return False
    f=lease(out/'queue.lock')
    if f is None:return True
    f.close();return False
def status(out):
    c=config();tasks=jobs(c);state=read(out/'queue_state.json') if (out/'queue_state.json').exists() else {'state':'not started','running':[]}
    active=busy_queue(out);label=state['state']
    if not active and label in ['running','pausing']:label='queue not active; inspect log then repeat start'
    print('Queue:',label,'| running:',', '.join(state['running']) or 'none')
    for kind in ['generator','cache','ranker','element','val','test']:
        jj=[j for j in tasks if j['kind']==kind];print(f'{kind}: {sum(finished(out,j) for j in jj)}/{len(jj)}')
    for g in c['groups']:
        parts=[]
        for role in c['roles']:
            folder=out/g/'generators'/role;p=folder/'history.json';h=read(p) if p.exists() else [];s=f'{role} {h[-1]["epoch"] if h else 0}/116'
            if (folder/'complete.json').exists():s+=' best='+str(read(folder/'complete.json')['selected_epoch'])
            parts.append(s)
        print(g+': '+'; '.join(parts))
        for split in ['val','test']:
            p=out/g/split/'metrics.json'
            if p.exists():
                r=read(p);print(f'  {split} no-mass Top1={r["before_mass"]["top1"]:.2%}; mass Top1={r["after_mass"]["top1"]:.2%}')
            elif (out/g/split/'progress.json').exists():
                r=read(out/g/split/'progress.json');print(f'  {split} {r["completed"]}/{r["total"]}')
    if state.get('error'):print('ERROR:',state['error'])
    print('Main log:',out/'matrix.log');print('Task logs:',out/'logs')
def summary(out):
    rows=[]
    for g in config()['groups']:
        for split in ['val','test']:
            p=out/g/split/'metrics.json'
            if not p.exists():continue
            r=read(p)
            for mode in ['before_mass','after_mass']:
                m=r[mode];rows.append({'group':g,'split':split,'condition':mode,'samples':m['samples'],'top1':m['top1'],'top5':m['top5'],'top10':m['top10'],'pool_coverage':m['pool_coverage'],'MRR10':m['MRR10'],'mean_pool_size':r['mean_pool_size']})
    if rows:
        with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(out/'summary.json',{'rows':rows,'independent_assemblies_per_group':1,'component_seeds_do_not_define_assembly_replicates':True,'test_has_previously_been_inspected_in_project':True})
    for r in rows:print(f'SUMMARY {r["group"]} {r["split"]} {r["condition"]}: Top1={r["top1"]:.2%} Top5={r["top5"]:.2%} Top10={r["top10"]:.2%}',flush=True)
def export(out):
    out.mkdir(parents=True,exist_ok=True);summary(out);stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');target=ROOT.parent/f'PTrans_Retrain6_results_{stamp}.zip';files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(x in {'.venv','__pycache__','data'} for x in rel.parts):continue
        if rel.parts[0]=='runs' and not p.is_relative_to(out):continue
        if p.suffix not in {'.py','.json','.jsonl','.csv','.log','.txt','.md','.sh'}:continue
        if '.partial_' in p.name or p.name.endswith('.lock'):continue
        files.append(p)
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in files:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
        z.writestr(ROOT.name+'/RESULTS.sha256.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as z:
        for name,m in manifest.items():
            h=hashlib.sha256();n=0
            with z.open(ROOT.name+'/'+name) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
            if h.hexdigest()!=m['sha256'] or n!=m['bytes']:raise ValueError('Result ZIP verification failed: '+name)
    info={'archive':str(target),'bytes':target.stat().st_size,'sha256':digest(target),'verified_files':len(manifest),'contains_weights':False,'contains_training_arrays':False}
    dump(out/'last_export.json',info);print('RESULTS_ZIP '+json.dumps(info),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','run','status','stop','summary','export']);p.add_argument('--parallel',type=int,choices=[1,2],default=2);a=p.parse_args();out=output()
    if a.action=='start':start(a.parallel)
    elif a.action=='run':run(a.parallel)
    elif a.action=='stop':out.mkdir(parents=True,exist_ok=True);(out/'STOP_REQUESTED').touch();print('Stop requested. Workers finish their current epoch/sample and save progress.')
    elif a.action=='status':status(out)
    elif a.action=='summary':summary(out)
    else:export(out)
if __name__=='__main__':main()
