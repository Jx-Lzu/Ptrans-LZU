"""One candidate pool, paired mass/no-mass assessment; no TEST-based selection."""
import csv,gzip,json,time,math
import numpy as np
import retrain_common as u
from system_core import canonical,masked_nmr,GROUPS
from ptrans_pcenter import Environments

def component_manifest(c,g):
    result={'protocol_sha256':u.protocol_id(c),'group':g,'generators':{},'heads':{}}
    for role in c['roles']:
        spec=u.generator_spec(c,g,role);folder=u.groupdir(c,g)/'generators'/role
        if u.sha(u.ROOT/spec['path'])!=spec['compact_sha256']:raise ValueError('Changed generator export')
        result['generators'][role]={'spec':spec,'receipt':u.read(folder/'complete.json')}
    for kind in ['ranker','element']:
        folder=u.groupdir(c,g)/kind;receipt=u.read(folder/'complete.json');spec=u.read(folder/'inference_spec.json')
        if receipt['protocol_sha256']!=u.protocol_id(c) or receipt['group']!=g or receipt['kind']!=kind or u.sha(folder/'selected.pt')!=receipt['selected_sha256'] or u.sha(u.ROOT/spec['path'])!=spec['compact_sha256'] or receipt['inference_sha256']!=spec['compact_sha256']:raise ValueError('Changed head export')
        result['heads'][kind]={'spec':spec,'receipt':receipt}
    return result

def freeze_group(c,g):
    manifest=component_manifest(c,g);path=u.groupdir(c,g)/'assembly.json'
    if path.exists() and u.read(path)!=manifest:raise ValueError('Already frozen system changed')
    u.dump(path,manifest);return manifest

def freeze_test(c,ctx=None):
    groups={}
    for g in c['groups']:
        frozen=freeze_group(c,g);folder=u.groupdir(c,g)/'val';report=u.read(folder/'metrics.json')
        if report['assembly_sha256']!=u.object_sha(frozen) or u.sha(folder/'candidate_progress.jsonl')!=report['progress_sha256']:raise ValueError('Complete full validation required before TEST')
        groups[g]={'assembly':frozen,'validation_metrics_sha256':u.sha(folder/'metrics.json')}
    ctx=ctx or u.context(c,False)
    plan={'protocol_sha256':u.protocol_id(c),'groups':groups,'test_ids':[r['sample_id'] for r in ctx['by']['test']],
          'weights':{'alpha':.6,'beta':.25,'gamma':.25},'mass_tolerance_ppm':5.,
          'test_selection':False,'generator_selection':'minimum full-VAL loss; no beam or mass-dependent checkpoint selection'}
    path=u.out(c)/'test_plan.json'
    if path.exists() and u.read(path)!=plan:raise ValueError('Frozen TEST plan changed')
    u.dump(path,plan);print('TEST_PLAN_FROZEN all six complete systems; settings fixed before TEST.',flush=True)

def rank(keys,truth):return next((i for i,k in enumerate(keys,1) if k==truth),0)
def metrics(cases,key):
    n=len(cases);ranks=[r[key] for r in cases];hits={str(k):sum(0<r<=k for r in ranks) for k in [1,5,10]}
    return {'samples':n,'hits':hits,**{'top'+k:v/n for k,v in hits.items()},'pool_hits':sum(r>0 for r in ranks),
      'pool_coverage':sum(r>0 for r in ranks)/n,'MRR10':sum(1/r if 0<r<=10 else 0 for r in ranks)/n}

def assess(pred,refs,scaffolds):
    envs=Environments();cases=[]
    for p,r in zip(pred,refs):
        if p['sample_id']!=r['sample_id']:raise ValueError('Prediction/reference alignment differs')
        truth=canonical(r['canonical_smiles']);local=envs.get(r['canonical_smiles']);post=p['final_candidates'];pre=sorted(post,key=lambda x:x['pre_mass_rank'])
        row={'sample_id':r['sample_id'],'reference_smiles':r['canonical_smiles'],'scaffold_group_key':scaffolds[r['sample_id']],
          'p_count':r['p_atom_count'],'pool_size':len(post),'mass_matches':sum(x['mass_match'] for x in post)}
        for name,pool in [('before_mass',pre),('after_mass',post)]:
            top=pool[0]['predicted_smiles'] if pool else '';loc=envs.get(top) if top else None
            row['rank_'+name]=rank([x['canonical_smiles'] for x in pool],truth);row['top1_'+name]=top
            for radius in [1,2]:row[f'R{radius}_{name}']=bool(loc and local and loc[str(radius)]==local[str(radius)])
        cases.append(row)
    report={name:metrics(cases,'rank_'+name) for name in ['before_mass','after_mass']}
    for name in ['before_mass','after_mass']:
        report[name]['local_structure']={}
        for population in ['all','single_P']:
            rr=cases if population=='all' else [r for r in cases if r['p_count']==1]
            report[name]['local_structure'][population]={'samples':len(rr),**{f'R{k}_top1':sum(r[f'R{k}_{name}'] for r in rr)/len(rr) if rr else None for k in [1,2]}}
    report.update(mean_pool_size=float(np.mean([r['pool_size'] for r in cases])),max_pool_size=max(r['pool_size'] for r in cases),
      empty_pools=sum(r['pool_size']==0 for r in cases),no_mass_match_records=sum(r['mass_matches']==0 for r in cases))
    return report,cases

def run(c,g,split,device,ctx=None):
    assembly=freeze_group(c,g);ctx=ctx or u.context(c,False);rows=ctx['by'][split];ids=[r['sample_id'] for r in rows]
    if split=='test':
        plan=u.read(u.out(c)/'test_plan.json')
        if plan['protocol_sha256']!=u.protocol_id(c) or plan['groups'][g]['assembly']!=assembly or plan['test_ids']!=ids:raise ValueError('TEST does not match its frozen plan')
    folder=u.groupdir(c,g)/split;folder.mkdir(parents=True,exist_ok=True)
    policy={'protocol_sha256':u.protocol_id(c),'assembly_sha256':u.object_sha(assembly),'group':g,'split':split,'sample_ids':ids,
      'masked_inputs_sha256':u.object_sha([{'sample_id':r['sample_id'],'nmr':masked_nmr(r,g)} for r in rows]),
      'known_mass_sha256':u.object_sha([float(r['molecular_weight_field']) for r in rows]),'reference_structure_is_not_ranking_input':True}
    path=folder/'protocol.json'
    if path.exists() and u.read(path)!=policy:raise ValueError('Evaluation progress belongs to another model/input')
    u.dump(path,policy);progress=folder/'candidate_progress.jsonl';pred=u.read_progress(progress,ids);tick=time.perf_counter()
    if len(pred)<len(rows):
        engine=u.TrainedAssembly(c,g,device);npy=folder/'spectra.npy';meta=folder/'spectra.json'
        if npy.exists() and meta.exists():
            m=u.read(meta)
            if m['protocol_sha256']!=u.object_sha(policy) or u.sha(npy)!=m['sha256']:raise ValueError('Evaluation features changed')
            spectra=np.load(npy,allow_pickle=False)
        else:
            blocks=[]
            for start in range(0,len(rows),c['spectral_batch_size']):
                u.stopped(c);blocks.append(engine.spectral_features(rows[start:start+c['spectral_batch_size']],g,c['spectral_batch_size']))
            spectra=np.concatenate(blocks);tmp=npy.with_name(npy.name+'.tmp')
            with tmp.open('wb') as f:np.save(f,spectra,allow_pickle=False)
            tmp.replace(npy);u.dump(meta,{'protocol_sha256':u.object_sha(policy),'sha256':u.sha(npy)})
        if spectra.shape!=(len(rows),ctx['model_args']['d_model']*4) or not np.isfinite(spectra).all():raise ValueError('Wrong spectral cache shape')
        for i in range(len(pred),len(rows)):
            u.stopped(c);r=rows[i];nmr=masked_nmr(r,g);start=time.perf_counter();mass=float(r['molecular_weight_field'])
            if not math.isfinite(mass) or mass<=0:raise ValueError('Invalid known theoretical mass')
            result=engine.predict(nmr,mass,spectra[i])
            value={'sample_id':r['sample_id'],'masked_input_sha256':u.object_sha(nmr),'known_mass_da':mass,
              'observed_log_peak_counts':result['observed_log_peak_counts'],'pre_mass_order':[x['canonical_smiles'] for x in result['pre_mass_candidates']],
              'final_candidates':result['final_candidates'],'raw_proposals':result['raw_proposals'],'seconds':time.perf_counter()-start}
            if set(value['pre_mass_order'])!={x['canonical_smiles'] for x in value['final_candidates']}:raise ValueError('Mass reranking changed the candidate pool')
            for nucleus in set(['1H','13C','31P'])-set(GROUPS[g]):
                if value['observed_log_peak_counts'][['1H','13C','31P'].index(nucleus)]!=0:raise ValueError('Excluded modality leaked into count features')
            u.append_progress(progress,value);pred.append(value)
            if (i+1)%16==0 or i+1==len(rows):
                u.dump(folder/'progress.json',{'completed':i+1,'total':len(rows)});print(f'EVAL {g}/{split} {i+1}/{len(rows)} seconds={value["seconds"]:.1f}',flush=True)
    with (u.ROOT/c['manifest']).open(encoding='utf-8-sig',newline='') as f:scaffolds={r['sample_id']:r['scaffold_group_key'] for r in csv.DictReader(f)}
    report,cases=assess(pred,rows,scaffolds);u.write_csv(folder/'per_sample_results.csv',cases)
    flat=[{'sample_id':p['sample_id'],'split':split,**x} for p in pred for x in p['final_candidates']]
    if flat:
        fields=sorted({k for r in flat for k in r});u.write_csv(folder/'candidate_scores.csv',[{k:r.get(k,'') for k in fields} for r in flat])
    report.update(group=g,split=split,protocol_sha256=u.protocol_id(c),assembly_sha256=u.object_sha(assembly),
      progress_sha256=u.sha(progress),per_sample_sha256=u.sha(folder/'per_sample_results.csv'),
      seconds_this_invocation=time.perf_counter()-tick,recorded_inference_seconds=sum(p['seconds'] for p in pred),
      newly_trained_complete_system=True,independent_system_repetitions=1,internal_generator_seeds=[42,7],
      neural_mass_input=False,mass_condition='Known theoretical exact mass, 5 ppm stable priority; not measured HRMS',
      generator_selection='minimum full validation loss',test_used_for_selection=False)
    u.dump(folder/'metrics.json',report)
    for name in ['before_mass','after_mass']:print('RETRAIN6_RESULT '+u.stable_json({'group':g,'split':split,'condition':name,**report[name]}),flush=True)
