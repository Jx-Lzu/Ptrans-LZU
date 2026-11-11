"""One durable DAG task. A lease prevents duplicate work after queue restarts."""
import argparse,os,json
from pathlib import Path
import retrain_common as u

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['prepare','generator','cache','ranker','element','val','freeze','test']);p.add_argument('--group');p.add_argument('--role');p.add_argument('--job-id',required=True);p.add_argument('--device',default='cuda',choices=['cuda','cpu']);a=p.parse_args()
    c=u.settings();u.initialize_runtime(c,a.device);lock=None
    if os.name=='posix':
        import fcntl
        lockpath=u.out(c)/'locks'/(a.job_id+'.lock');lockpath.parent.mkdir(parents=True,exist_ok=True);lock=lockpath.open('a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(76)
    try:
        u.stopped(c)
        if a.kind=='prepare':u.prepare(c,a.device);paths=[u.out(c)/'protocol.json',u.out(c)/'prepared.json']
        else:
            if u.original.env(a.device)!=u.read(u.out(c)/'protocol.json')['environment']:raise RuntimeError('Runtime differs from this matrix protocol')
            if a.kind=='generator':
                from train_generator import train
                train(c,a.group,a.role,a.device);folder=u.groupdir(c,a.group)/'generators'/a.role
                paths=[folder/x for x in ['complete.json','inference_spec.json','selected_inference.pt','best.pt','final.pt','history.json']]
            elif a.kind=='cache':
                from prepare_group_cache import prepare
                prepare(c,a.group,a.device);folder=u.groupdir(c,a.group)/'cache';paths=[folder/x for x in ['complete.json','contract.json','arrays.npz','metadata.json','negative_audit.csv']]
            elif a.kind in ['ranker','element']:
                from train_heads import train
                train(c,a.group,a.kind,a.device);folder=u.groupdir(c,a.group)/a.kind
                paths=[folder/x for x in ['complete.json','inference_spec.json','selected_inference.pt','selected.pt','history.json']]
            elif a.kind in ['val','test']:
                from evaluate_group import run
                run(c,a.group,a.kind,a.device);folder=u.groupdir(c,a.group)/a.kind
                paths=[folder/x for x in ['metrics.json','protocol.json','candidate_progress.jsonl','per_sample_results.csv']]
            else:
                from evaluate_group import freeze_test
                freeze_test(c);paths=[u.out(c)/'test_plan.json']
        receipt={'job_id':a.job_id,'protocol_sha256':u.protocol_id(c),'config_sha256':u.object_sha(c),
                 'outputs':{str(path.relative_to(u.ROOT)).replace('\\','/'):{'bytes':path.stat().st_size,'sha256':u.sha(path)} for path in paths}}
        u.dump(u.out(c)/'jobs'/(a.job_id+'.json'),receipt);print('JOB_COMPLETE '+a.job_id,flush=True)
    finally:
        if lock is not None:lock.close()
if __name__=='__main__':main()
