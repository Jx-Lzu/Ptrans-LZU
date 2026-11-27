#!/usr/bin/env python
"""Rebuild training-only SMILES variants on CPU; does not train a model."""
import argparse
from pathlib import Path
import ptrans_smiles_aug as aug


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parent))
    p.add_argument('--data',default='data/ptrans_normalized_dataset.jsonl.gz')
    p.add_argument('--manifest',default='data/ptrans_split_manifest.csv')
    p.add_argument('--output-data',default='data/ptrans_smiles_aug_train_v1.jsonl.gz')
    p.add_argument('--output-meta',default='data/ptrans_smiles_aug_train_v1.meta.json')
    p.add_argument('--variants',type=int,default=4)
    p.add_argument('--attempts',type=int,default=24)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-target-tokens',type=int,default=384)
    args=p.parse_args();root=Path(args.project_root).resolve()
    if args.variants<1 or args.attempts<args.variants:raise ValueError('Invalid variant/attempt counts')
    records=aug.load_records(root/args.data,root/args.manifest)
    entries,metadata=aug.build_variants(records,aug.token_pattern(root),
        args.max_target_tokens,args.seed,args.variants,args.attempts)
    metadata=aug.save_cache(root/args.output_data,root/args.output_meta,entries,metadata)
    print('Training records:',metadata['training_records'])
    print('Records with equivalent variants:',metadata['records_with_variants'])
    print('Cache bytes:',metadata['cache_bytes'])


if __name__=='__main__':main()
