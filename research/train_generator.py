"""Full-budget training with atomic own-run resume and retained best-validation weights."""
import math,time
import torch
from torch import nn
import retrain_common as u

def train(c,g,role,device,ctx=None):
    ctx=ctx or u.context(c);ctx={**ctx,'model_args':{**ctx['model_args'],'encoder_layers':c['roles'][role]['encoder_layers']}}
    seed=c['roles'][role]['seed'];base=ctx['base'];base.set_seed(seed)
    model=u.rbf.model_class(ctx['arch'])(len(ctx['vocab']),len(ctx['multiplicity_vocab']),**ctx['model_args'],pad_id=ctx['vocab']['<pad>']).to(device).float()
    folder=u.groupdir(c,g)/'generators'/role;folder.mkdir(parents=True,exist_ok=True)
    identity={'protocol_sha256':u.protocol_id(c),'group':g,'role':role,'seed':seed,'model_args':ctx['model_args'],'initial_weights_sha256':u.weights_sha(model.state_dict())}
    if (folder/'run.json').exists() and u.read(folder/'run.json')!=identity:raise ValueError('Generator run identity changed')
    u.dump(folder/'run.json',identity)
    per_epoch=math.ceil(len(ctx['by']['train'])/c['batch_size']);epochs=sum(x['epochs'] for x in c['stages'])
    criterion=nn.CrossEntropyLoss(ignore_index=ctx['vocab']['<pad>'],label_smoothing=c['label_smoothing'])
    start=1;history=[];updates=0;best_loss=float('inf');best_epoch=0;best_weights=None;optimizer=None;opt_phase=None
    def snapshot(weights,epoch,loss):return {'format':'ptrans_retrained_generator_v1','run':identity,'epoch':epoch,'updates':epoch*per_epoch,'val_loss':loss,
        'model_args':ctx['model_args'],'vocab':ctx['vocab'],'inverse_vocab':ctx['inverse_vocab'],'multiplicity_vocab':ctx['multiplicity_vocab'],
        'rbf_encoding':{'centers':ctx['model_args']['rbf_centers'],'widths':ctx['model_args']['rbf_widths']},
        'model_state_dict':weights,'model_weights_sha256':u.weights_sha(weights)}
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if last['run']!=identity or last['updates']!=last['epoch']*per_epoch:raise ValueError('Invalid own-generator resume checkpoint')
        model.load_state_dict(last['model_state_dict'],strict=True);history=last['history'];updates=last['updates'];start=last['epoch']+1
        best_loss=last['best_val_loss'];best_epoch=last['best_epoch'];best_weights=last['best_model_state_dict'];opt_phase=last['phase']
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['stages'][opt_phase]['lr'],weight_decay=c['weight_decay']);optimizer.load_state_dict(last['optimizer_state_dict'])
        u.restore_rng(last['rng'],device)
        # last.pt is authoritative: repair sidecars after a crash between writes.
        u.save(folder/'best.pt',snapshot(best_weights,best_epoch,best_loss));u.dump(folder/'history.json',history)
        print(f'RESUME {g}/{role} after epoch={last["epoch"]}',flush=True)
    for epoch in range(start,epochs+1):
        u.stopped(c);phase,spec,pe=u.original.phase(c,epoch)
        if phase!=opt_phase:optimizer=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=c['weight_decay']);opt_phase=phase
        tick=time.perf_counter()
        tl,steps,order,targets=u.original.loss_epoch(ctx,model,u.loader(ctx,ctx['by']['train'],g,seed,epoch,True,c),criterion,optimizer,device)
        vl,_,_,_=u.original.loss_epoch(ctx,model,u.loader(ctx,ctx['by']['val'],g,seed,epoch,False,c),criterion,None,device)
        if steps!=per_epoch:raise ValueError('Update count differs')
        updates+=steps;weights=u.cpu_state(model)
        improved=vl<best_loss
        if improved:best_loss=vl;best_epoch=epoch;best_weights=weights
        row={'epoch':epoch,'phase':spec['name'],'phase_epoch':pe,'learning_rate':spec['lr'],'updates':updates,'train_loss':tl,'val_loss':vl,
             'sample_order_sha256':order,'target_order_sha256':targets,'best_epoch':best_epoch,'best_val_loss':best_loss,'seconds':time.perf_counter()-tick}
        history.append(row)
        state={'run':identity,'epoch':epoch,'phase':phase,'updates':updates,'history':history,'model_state_dict':weights,
          'best_model_state_dict':best_weights,'best_epoch':best_epoch,'best_val_loss':best_loss,'optimizer_state_dict':optimizer.state_dict(),'rng':u.rng_state(device)}
        u.save(folder/'last.pt',state)
        if improved:u.save(folder/'best.pt',snapshot(best_weights,best_epoch,best_loss))
        u.dump(folder/'history.json',history)
        print(f'GEN {g}/{role} seed={seed} epoch={epoch}/{epochs} lr={spec["lr"]} train={tl:.6f} val={vl:.6f} best={best_epoch}:{best_loss:.6f} updates={updates} seconds={row["seconds"]:.1f}',flush=True)
        u.stopped(c)
    if best_weights is None:raise ValueError('No selected generator')
    u.save(folder/'best.pt',snapshot(best_weights,best_epoch,best_loss))
    u.save(folder/'final.pt',snapshot(u.cpu_state(model),epochs,history[-1]['val_loss']))
    spec=u.compact_checkpoint(folder/'best.pt',folder/'selected_inference.pt');u.dump(folder/'inference_spec.json',spec)
    u.dump(folder/'complete.json',{'protocol_sha256':u.protocol_id(c),'group':g,'role':role,'trained_epochs':epochs,'trained_updates':updates,
        'selection':'minimum full validation teacher-forced loss, earlier on ties','selected_epoch':best_epoch,'selected_updates':best_epoch*per_epoch,
        'selected_val_loss':best_loss,'selected_sha256':u.sha(folder/'best.pt'),'final_sha256':u.sha(folder/'final.pt'),
        'inference_sha256':spec['compact_sha256'],'test_used_for_selection':False})
    print(f'GEN_COMPLETE {g}/{role}: trained={epochs} selected={best_epoch} val_loss={best_loss:.6f}',flush=True)
