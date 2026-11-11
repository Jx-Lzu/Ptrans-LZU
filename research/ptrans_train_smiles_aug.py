#!/usr/bin/env python
"""One-factor training-only SMILES augmentation using the unchanged shared MLP.

Starts a separate run with the original canonical-training vocabulary, NMR
features, model, optimizer and split. Evaluation uses ptrans_train_arch.py.
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import ptrans_smiles_aug as aug

FORMAT = 'ptrans_smiles_aug_v1'


def augmented_dataset_class(base,cache,probability,seed,state,counters):
    original=base.PTransDataset
    class AugmentedDataset(original):
        def __getitem__(self,index):
            item=super().__getitem__(index)
            record=self.records[index]
            if record['split_scaffold']!='train':return item
            canonical=record['canonical_smiles'];sid=record['sample_id']
            target,kind=aug.choose_target(sid,canonical,cache[sid],max(1,state['epoch']),probability,seed)
            ts=base.tokenize_smiles(target)
            if set(ts)-self.vocab.keys() or len(ts)+2>self.max_target_tokens:
                raise ValueError('Augmentation violates frozen vocabulary/target capacity')
            item['target_ids']=torch.tensor([self.bos_id]+[self.vocab[t] for t in ts]+[self.eos_id],dtype=torch.long)
            item['smiles']=target
            counters[kind]+=1
            return item
    return AugmentedDataset


def main(argv=None):
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parent))
    p.add_argument('--aug-config',default='configs/smiles_aug_v1.json')
    options,forwarded=p.parse_known_args(argv)
    root=Path(options.project_root).resolve()
    if aug.digest(root/'ptrans_train_baseline.py')!=aug.BASE_SHA256 or aug.digest(root/'ptrans_train_arch.py')!=aug.ARCH_SHA256:
        raise ValueError('Keep the original baseline and architecture drivers unchanged')
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('ptrans_training_smiles_arch',root/'ptrans_train_arch.py')
    arch=importlib.util.module_from_spec(spec);sys.modules[spec.name]=arch;spec.loader.exec_module(arch)
    if not any(a=='--config' or a.startswith('--config=') for a in forwarded):
        forwarded+=['--config','configs/arch_shared_smiles_aug_v1.json']
    args=arch.parse_args(forwarded)
    if args.eval_only or args.evaluate_test or args.checkpoint:
        raise ValueError('Use unchanged ptrans_train_arch.py --eval-only for evaluation; this wrapper is training only')
    if args.architecture!='shared_mlp':raise ValueError('This control keeps the shared_mlp architecture')
    if args.num_workers!=0:raise ValueError('Keep num_workers=0 for deterministic per-epoch target selection')
    out=Path(args.output_dir)
    if not args.resume and any((out/name).exists() for name in ['best.pt','last.pt','training_history.json']):
        raise FileExistsError('Use a new output directory or resume its augmented last.pt')
    cfg_path=root/options.aug_config
    cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
    if set(cfg)!={'probability','seed','cache_data','cache_meta'}:
        raise ValueError('Unknown or missing SMILES augmentation config keys')
    probability=cfg['probability'];seed=cfg['seed']
    if isinstance(probability,bool) or not isinstance(probability,(int,float)) or not 0<=probability<=1:
        raise ValueError('Invalid augmentation probability')
    if seed!=args.seed:raise ValueError('Keep augmentation selection seed equal to training seed')
    records=aug.load_records(args.data,args.manifest)
    cache,metadata,inverse=aug.load_cache(root/cfg['cache_data'],root/cfg['cache_meta'],records,
        aug.token_pattern(root),args.max_target_tokens,verify_structures=True)
    train=[r for r in records if r['split_scaffold']=='train']
    _,actual_inverse,_=arch.base.build_vocabs(train)
    if actual_inverse!=inverse:raise ValueError('Vocabulary differs from original canonical-only training vocabulary')
    if metadata['seed']!=seed:raise ValueError('Variant cache seed differs from augmentation config')
    provenance={'method':'training-only equivalent SMILES target augmentation',
        'probability':probability,'selection_seed':seed,'selection':'SHA256(seed, epoch, sample_id), independent of loader RNG',
        'data_fingerprint':metadata['dataset_fingerprint'],'cache_sha256':metadata['cache_sha256'],
        'cache_meta_sha256':aug.digest(root/cfg['cache_meta']),'aug_config_sha256':aug.digest(cfg_path),
        'wrapper_sha256':aug.digest(Path(__file__)),'helper_sha256':aug.digest(root/'ptrans_smiles_aug.py'),
        'vocabulary_sha256':metadata['vocabulary_sha256'],'vocabulary_unchanged':True,
        'training_records':len(train),'records_with_variants':metadata['records_with_variants'],
        'validation_targets_augmented':False,'test_targets_augmented':False,
        'molecular_structure_and_stereochemistry_verified':True,
        'fallback_policy':'If no eligible random spelling exists, use canonical; never truncate',
        'training_format':FORMAT}
    state={'epoch':0};counters=Counter();last_stats={}
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint.get('format_version')!=FORMAT or checkpoint.get('smiles_augmentation',{}).get('config')!=provenance:
            raise ValueError('Resume requires an augmented checkpoint with the same cache/config/code')
        state['epoch']=checkpoint['epoch']
        del checkpoint
    out.mkdir(parents=True,exist_ok=True)
    provenance_path=out/'augmentation_provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text(encoding='utf-8'))!=provenance:
        raise ValueError('Output directory has different augmentation provenance')
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    stats_path=out/'augmentation_epoch_stats.jsonl'
    if args.resume and stats_path.exists():
        saved=[json.loads(line) for line in stats_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        saved=[row for row in saved if row['epoch']<=state['epoch']]
        stats_path.write_text(''.join(json.dumps(row)+'\n' for row in saved),encoding='utf-8')
    original_run_loss=arch.run_loss
    original_atomic_save=arch.base.atomic_save

    def annotate_run_config():
        path=out/'run_config.json'
        if path.exists():
            config=json.loads(path.read_text(encoding='utf-8'))
            config['smiles_augmentation']=provenance
            path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')

    def run_loss(model,loader,criterion,optimizer,device,pad_id):
        if optimizer is not None:
            state['epoch']+=1;counters.clear()
        result=original_run_loss(model,loader,criterion,optimizer,device,pad_id)
        if optimizer is not None:
            n=sum(counters.values())
            last_stats.clear();last_stats.update({'epoch':state['epoch'],'records_seen':n,
                'canonical':counters['canonical'],'augmented':counters['augmented'],
                'fallback_no_variants':counters['fallback_no_variants'],
                'actual_augmented_fraction':counters['augmented']/n if n else 0.})
            with stats_path.open('a',encoding='utf-8') as f:f.write(json.dumps(last_stats)+'\n')
            print('SMILES augmentation:',json.dumps(last_stats),flush=True)
        return result

    def atomic_save(obj,path):
        obj={**obj,'smiles_augmentation':{'config':provenance,'epoch_stats':dict(last_stats)}}
        original_atomic_save(obj,path)
        annotate_run_config()

    arch.base.PTransDataset=augmented_dataset_class(arch.base,cache,probability,seed,state,counters)
    arch.run_loss=run_loss;arch.base.atomic_save=atomic_save;arch.FORMAT=FORMAT
    print(f'Training-only SMILES augmentation: requested={probability:.0%}; '
          f'eligible records={metadata["records_with_variants"]}/{len(train)}; unchanged vocabulary={len(inverse)}',flush=True)
    print('Shared MLP; original NMR inputs and split; validation targets remain canonical.',flush=True)
    arch.main(forwarded)
    annotate_run_config()
    if args.dry_run:
        print('Augmented dry-run samples:',json.dumps(dict(counters)),'; optimizer updates=0',flush=True)


if __name__=='__main__':main()
