import time
import numpy as np
import torch
from torch.nn import functional as F
import retrain_common as u
from prepare_group_cache import load_cache
from frozen_heads import CompatibilityRanker,ElementCountHead,count_prediction

def train(c,g,kind,device):
    arrays,meta,cache_receipt=load_cache(c,g);spec=c[kind];folder=u.groupdir(c,g)/kind;folder.mkdir(parents=True,exist_ok=True)
    if kind=='ranker':dims={'spectrum_dim':arrays['spectra'].shape[1],'bits':arrays['fingerprints'].shape[1],'hidden':spec['hidden'],'embedding':spec['embedding']};cls=CompatibilityRanker
    else:dims={'spectrum_dim':arrays['spectra'].shape[1],'elements':9,'hidden':spec['hidden']};cls=ElementCountHead
    seed=spec['seed'];u.original.env(device);u.original.random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    model=cls(**dims).to(device).float();optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay']);rng=torch.Generator().manual_seed(seed)
    identity={'protocol_sha256':u.protocol_id(c),'group':g,'kind':kind,'cache':cache_receipt,'settings':spec,'initial_weights_sha256':u.weights_sha(model.state_dict()),
       'validation_or_test_optimizer_records':0,'source_encoder':u.generator_spec(c,g,'primary')}
    tensors={k:torch.as_tensor(v,device=device) for k,v in arrays.items() if k not in ['fit_indices','calibration_indices']}
    for k in ['spectra','fingerprints','descriptors','peak_log_counts','element_counts']:tensors[k]=tensors[k].float()
    n=len(meta['sample_ids']);normalization=None;fit=None;cal=None
    if kind=='element':
        fit=arrays['fit_indices'];cal=arrays['calibration_indices'];logs=np.log1p(arrays['element_counts']).astype(np.float32)
        mean=logs[fit].mean(0);raw_scale=logs[fit].std(0);scale=np.maximum(raw_scale,spec['target_scale_floor']);active=(raw_scale>1e-8).astype(np.float32)
        if active.sum()<1:raise ValueError('No variable count targets')
        normalization={'target_mean':mean.tolist(),'target_scale':scale.tolist(),'active':active.tolist(),
         'fit_mean_counts':arrays['element_counts'][fit].mean(0).tolist(),'residual_scale_floor':spec['residual_scale_floor']}
        tensors['targets']=torch.tensor((logs-mean)/scale,device=device);mask=torch.tensor(active,device=device)
        identity['normalization']=normalization
        # The split is entirely inside original TRAIN and groups duplicate structures.
        identity['fit_ids']=[meta['sample_ids'][int(i)] for i in fit];identity['calibration_ids']=[meta['sample_ids'][int(i)] for i in cal]
    if (folder/'run.json').exists() and u.read(folder/'run.json')!=identity:raise ValueError('Head run identity changed')
    u.dump(folder/'run.json',identity)
    def loss(indices):
        if kind=='ranker':
            pos=tensors['positive'][indices];nlib=meta['random_library_size']
            random=torch.randint(nlib-1,(len(indices),spec['random_negatives']),generator=rng).to(device);random=random+(random>=pos[:,None])
            chosen=torch.cat([pos[:,None],tensors['hard'][indices],random],1)
            if chosen[:,1:].eq(pos[:,None]).any():raise ValueError('False negative sampled')
            scores=model(tensors['spectra'][indices],tensors['fingerprints'][chosen],tensors['descriptors'][chosen]);value=F.cross_entropy(scores,torch.zeros(len(indices),device=device,dtype=torch.long))
        else:
            prediction=model(tensors['spectra'][indices],tensors['peak_log_counts'][indices]);value=((prediction-tensors['targets'][indices]).square()*mask).sum(-1).div(mask.sum()).mean()
        if not torch.isfinite(value):raise ValueError('Nonfinite head loss')
        return value
    start=1;history=[];updates=0;best=float('inf');best_epoch=0;best_weights=None
    lastpath=folder/'last.pt'
    if lastpath.exists():
        last=torch.load(lastpath,map_location='cpu',weights_only=False)
        if last['run']!=identity:raise ValueError('Wrong own-head resume state')
        model.load_state_dict(last['model_state_dict']);optimizer.load_state_dict(last['optimizer_state_dict']);rng.set_state(last['sampling_rng']);u.restore_rng(last['rng'],device)
        history=last['history'];updates=last['updates'];start=last['epoch']+1;best=last['best_loss'];best_epoch=last['best_epoch'];best_weights=last['best_model_state_dict']
        print(f'RESUME {g}/{kind} after epoch={last["epoch"]}',flush=True)
    for epoch in range(start,spec['epochs']+1):
        u.stopped(c);tick=time.perf_counter();model.train();total=0.;indices=np.arange(n) if kind=='ranker' else fit
        order=torch.randperm(len(indices),generator=rng).numpy()
        for offset in range(0,len(indices),spec['batch_size']):
            ii=torch.as_tensor(indices[order[offset:offset+spec['batch_size']]],device=device,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);value=loss(ii);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=value.item()*len(ii);updates+=1
        train_loss=total/len(indices);criterion=train_loss
        if kind=='element':
            model.eval();total=0.
            with torch.no_grad():
                for offset in range(0,len(cal),spec['batch_size']):
                    ii=torch.as_tensor(cal[offset:offset+spec['batch_size']],device=device,dtype=torch.long);total+=loss(ii).item()*len(ii)
            criterion=total/len(cal)
        weights=u.cpu_state(model)
        if criterion<best:best=criterion;best_epoch=epoch;best_weights=weights
        row={'epoch':epoch,'train_loss':train_loss,'selection_loss':criterion,'selection_scope':'TRAIN-only internal calibration' if kind=='element' else 'diagnostic training loss; final ranker is used',
          'updates':updates,'seconds':time.perf_counter()-tick};history.append(row)
        u.save(lastpath,{'run':identity,'epoch':epoch,'model_state_dict':weights,'optimizer_state_dict':optimizer.state_dict(),'sampling_rng':rng.get_state(),'rng':u.rng_state(device),
          'history':history,'updates':updates,'best_loss':best,'best_epoch':best_epoch,'best_model_state_dict':best_weights})
        u.dump(folder/'history.json',history);print(f'HEAD {g}/{kind} epoch={epoch}/{spec["epochs"]} train={train_loss:.6f} selection={criterion:.6f} updates={updates}',flush=True)
        u.stopped(c)
    chosen=u.cpu_state(model) if kind=='ranker' else best_weights;chosen_epoch=spec['epochs'] if kind=='ranker' else best_epoch
    export={'format':'ptrans_retrained_'+kind+'_v1','run':identity,'model_state_dict':chosen,'model_args':dims,'selected_epoch':chosen_epoch,'trained_epochs':spec['epochs'],'optimizer_updates':updates,'training_complete':True}
    if kind=='element':
        model.load_state_dict(chosen);model.eval();residual=[]
        with torch.no_grad():
            for offset in range(0,len(cal),spec['batch_size']):
                idx=cal[offset:offset+spec['batch_size']];ii=torch.tensor(idx,device=device,dtype=torch.long)
                residual.append(count_prediction(model,tensors['spectra'][ii],tensors['peak_log_counts'][ii],normalization).cpu().numpy()-logs[idx])
        sigma=np.maximum(np.sqrt(np.mean(np.concatenate(residual)**2,axis=0)),spec['residual_scale_floor'])
        export.update(normalization=normalization,residual_log_scales=sigma.tolist())
    path=folder/'selected.pt';u.save(path,export);inference=u.compact_checkpoint(path,folder/'selected_inference.pt');u.dump(folder/'inference_spec.json',inference)
    u.dump(folder/'complete.json',{'protocol_sha256':u.protocol_id(c),'group':g,'kind':kind,'selected_epoch':chosen_epoch,'selected_sha256':u.sha(path),
       'trained_epochs':spec['epochs'],'updates':updates,'inference_sha256':inference['compact_sha256'],'test_used_for_selection':False})
    print(f'HEAD_COMPLETE {g}/{kind} selected={chosen_epoch}',flush=True)
