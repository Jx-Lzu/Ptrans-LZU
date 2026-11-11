"""TRAIN-only features and generator-error negatives for each newly trained nuclear group."""
from pathlib import Path
import json,time
import numpy as np
import torch
from rdkit.Chem import rdFingerprintGenerator
import retrain_common as u
from system_core import canonical,masked_nmr
from frozen_heads import candidate_features,peak_features

def prepare(c,g,device,ctx=None):
    ctx=ctx or u.context(c,False);train=ctx['by']['train'];ids=[r['sample_id'] for r in train];by={r['sample_id']:r for r in train}
    common,meta=u.common_cache(c,ctx);folder=u.groupdir(c,g)/'cache';folder.mkdir(parents=True,exist_ok=True)
    primary=u.generator_spec(c,g,'primary');secondary=u.generator_spec(c,g,'secondary')
    order=np.random.default_rng(c['mining']['seed']).permutation(len(train))[:min(c['mining']['samples'],len(train))]
    chosen=[ids[int(i)] for i in order]
    contract={'protocol_sha256':u.protocol_id(c),'group':g,'primary':primary,'secondary':secondary,'mining_ids':chosen,
      'all_train_ids':ids,'molecular_library_sha256':u.sha(u.ROOT/c['common_library']),'neural_input':'only retained nucleus peak lists',
      'negative_generation':'in-sample TRAIN beam errors, not out-of-fold','validation_or_test_negative_sources':0}
    pin=folder/'contract.json'
    if pin.exists() and u.read(pin)!=contract:raise ValueError('Group cache contract changed')
    u.dump(pin,contract);done=folder/'complete.json'
    if done.exists():
        info=u.read(done)
        if info['contract_sha256']!=u.object_sha(contract) or u.sha(folder/'arrays.npz')!=info['arrays_sha256']:raise ValueError('Completed group cache changed')
        return
    engine=u.TrainedAssembly(c,g,device,load_heads=False,load_extra=False)
    feature_path=folder/'spectra.npy';feature_meta=folder/'spectra.json'
    if feature_path.exists() and feature_meta.exists():
        m=u.read(feature_meta)
        if m['contract_sha256']!=u.object_sha(contract) or u.sha(feature_path)!=m['sha256']:raise ValueError('Wrong group spectral feature cache')
        spectra=np.load(feature_path,allow_pickle=False)
    else:
        blocks=[]
        for start in range(0,len(train),c['spectral_batch_size']):
            u.stopped(c);block=train[start:start+c['spectral_batch_size']]
            blocks.append(engine.spectral_features(block,g,c['spectral_batch_size']))
            if start%(c['spectral_batch_size']*20)==0:print(f'FEATURES {g} {min(start+len(block),len(train))}/{len(train)}',flush=True)
        spectra=np.concatenate(blocks)
        temp=feature_path.with_name(feature_path.name+'.tmp')
        with temp.open('wb') as f:np.save(f,spectra,allow_pickle=False)
        temp.replace(feature_path);u.dump(feature_meta,{'contract_sha256':u.object_sha(contract),'sha256':u.sha(feature_path)})
    if spectra.shape!=(len(train),ctx['model_args']['d_model']*4) or not np.isfinite(spectra).all():raise ValueError('Invalid group spectra')
    progress_path=folder/'mining_progress.jsonl';mined=u.read_progress(progress_path,chosen)
    for i in range(len(mined),len(chosen)):
        u.stopped(c);sid=chosen[i];nmr=masked_nmr(by[sid],g);tick=time.perf_counter()
        aa=engine.a.generate(nmr,c['mining']['beam_size'],c['generation']['max_new_tokens']);bb=engine.b.generate(nmr,c['mining']['beam_size'],c['generation']['max_new_tokens'])
        proposed=[];seen=set()
        # Same raw-rank-first union as the historical TRAIN mining procedure.
        for j in range(max(len(aa),len(bb))):
            for pool in [aa,bb]:
                if j>=len(pool):continue
                sm,score=pool[j];key=canonical(sm)
                if key is None or key in seen:continue
                seen.add(key);proposed.append({'predicted_smiles':sm,'canonical_smiles':key})
        row={'sample_id':sid,'candidates':proposed,'seconds':time.perf_counter()-tick};u.append_progress(progress_path,row);mined.append(row)
        if (i+1)%16==0 or i+1==len(chosen):
            u.dump(folder/'progress.json',{'spectral_records':len(train),'mined':i+1,'mining_total':len(chosen)});print(f'MINING {g} {i+1}/{len(chosen)}',flush=True)
    del engine
    if device=='cuda':torch.cuda.empty_cache()
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    library=list(meta['library_smiles']);keys=list(library);index={s:i for i,s in enumerate(keys)};nrandom=len(library)
    fps=list(common['fingerprints']);descs=list(common['descriptors']);hard=common['hard'].copy();locations={sid:i for i,sid in enumerate(ids)}
    audit=[];slots=positive_excluded=0
    for row in mined:
        sid=row['sample_id'];ii=locations[sid];selected=[]
        for candidate in row['candidates']:
            key=candidate['canonical_smiles']
            if key==by[sid]['canonical_smiles']:positive_excluded+=1;continue
            if key not in index:
                k,fp,desc,_=candidate_features(candidate['predicted_smiles'],generator)
                if k!=key:raise ValueError('Generated negative identity changed')
                index[key]=len(library);keys.append(key);library.append(candidate['predicted_smiles']);fps.append(fp);descs.append(desc)
            idx=index[key]
            if idx not in selected:selected.append(idx)
            if len(selected)==c['ranker']['hard_negatives']:break
        used=len(selected);fallback=[int(x) for x in hard[ii] if int(x) not in selected]
        chosen_neg=(selected+fallback)[:c['ranker']['hard_negatives']]
        if len(chosen_neg)!=c['ranker']['hard_negatives']:raise ValueError('Insufficient TRAIN negatives')
        hard[ii]=chosen_neg;slots+=used;audit.append({'sample_id':sid,'generated_slots':used,'fallback_slots':len(chosen_neg)-used})
    if (hard==common['positive'][:,None]).any():raise ValueError('Positive became a negative')
    if c['mining'].get('require_generated_negative',True) and slots==0:raise ValueError('No usable generated TRAIN errors; inspect mining output before continuing')
    peaks=np.stack([peak_features({'nmr':masked_nmr(r,g)}) for r in train])
    arrays={'spectra':spectra.astype(np.float32),'peak_log_counts':peaks,'fingerprints':np.asarray(fps,dtype=np.uint8),
       'descriptors':np.asarray(descs,dtype=np.float32),'positive':common['positive'],'hard':hard,
       'element_counts':common['element_counts'],'fit_indices':common['fit_indices'],'calibration_indices':common['calibration_indices']}
    tmp=folder/'arrays.npz.tmp'
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(folder/'arrays.npz')
    metadata={'contract_sha256':u.object_sha(contract),'group':g,'sample_ids':ids,'library_smiles':library,'library_canonical_keys':keys,
       'random_library_size':nrandom,'generated_negative_slots':slots,'positive_outputs_excluded':positive_excluded,
       'input_spectral_features_from_this_group':True,'validation_or_test_labels_in_cache':0,'mining_is_out_of_fold':False,
       'arrays_sha256':u.sha(folder/'arrays.npz'),'protocol_sha256':u.protocol_id(c)}
    u.dump(folder/'metadata.json',metadata);u.write_csv(folder/'negative_audit.csv',audit)
    u.dump(done,{'protocol_sha256':u.protocol_id(c),'contract_sha256':u.object_sha(contract),'arrays_sha256':metadata['arrays_sha256'],
       'metadata_sha256':u.sha(folder/'metadata.json'),'train_records':len(train),'generated_slots':slots})
    print(f'CACHE_COMPLETE {g}: TRAIN={len(train)}, generated_negative_slots={slots}',flush=True)

def load_cache(c,g,ctx=None):
    folder=u.groupdir(c,g)/'cache';done=u.read(folder/'complete.json');meta=u.read(folder/'metadata.json')
    if done['protocol_sha256']!=u.protocol_id(c) or u.sha(folder/'arrays.npz')!=done['arrays_sha256'] or u.sha(folder/'metadata.json')!=done['metadata_sha256']:raise ValueError('Training cache changed')
    contract=u.read(folder/'contract.json')
    if contract['primary']!=u.generator_spec(c,g,'primary') or contract['secondary']!=u.generator_spec(c,g,'secondary') or contract['group']!=g:raise ValueError('Cache uses another encoder/group')
    with np.load(folder/'arrays.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    if (arrays['hard']==arrays['positive'][:,None]).any() or meta['validation_or_test_labels_in_cache']!=0:raise ValueError('Invalid training labels')
    return arrays,meta,done
