"""Verify the eight original training-data files; standard library only."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = read(root / 'TRAINING_DATA_SUPPLEMENT.sha256.json')
    checked = 0
    for relative, expected in manifest['files'].items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError('Missing or invalid file: ' + relative)
        if p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
            raise ValueError('File differs from package manifest: ' + relative)
        checked += 1
    data = root / 'research/data'
    records = []
    with gzip.open(data / 'ptrans_normalized_dataset.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    by_id = {r['sample_id']: r for r in records}
    if len(records) != 22226 or len(by_id) != len(records):
        raise ValueError('Dataset record count or IDs differ')
    counts = {s: sum(r['split_scaffold'] == s for r in records) for s in ['train', 'val', 'test']}
    if counts != {'train': 17782, 'val': 2222, 'test': 2222}:
        raise ValueError('Dataset splits differ')
    with (data / 'ptrans_split_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(records) or {r['sample_id'] for r in rows} != set(by_id):
        raise ValueError('Split manifest IDs differ')
    for row in rows:
        r = by_id[row['sample_id']]
        if any(row[k] != r[k] for k in ['canonical_smiles', 'split_scaffold', 'split_random_structure']):
            raise ValueError('Dataset/manifest mismatch: ' + row['sample_id'])
    train = [r for r in records if r['split_scaffold'] == 'train']
    train_ids = [r['sample_id'] for r in train]
    train_set = set(train_ids)
    augmentation_ids = []
    variants = 0
    with gzip.open(data / 'ptrans_smiles_aug_train_v1.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['sample_id'] not in train_set or row['canonical_smiles'] != by_id[row['sample_id']]['canonical_smiles']:
                raise ValueError('Augmentation record differs or is outside TRAIN')
            augmentation_ids.append(row['sample_id'])
            variants += len(row['variants'])
    if augmentation_ids != train_ids:
        raise ValueError('Augmentation IDs differ from TRAIN order')
    aug_meta = read(data / 'ptrans_smiles_aug_train_v1.meta.json')
    if variants != aug_meta['variant_strings_by_record'] or not aug_meta['training_only']:
        raise ValueError('Augmentation metadata differs')
    lib_meta = read(data / 'train_molecular_library.meta.json')
    if lib_meta['sample_ids'] != train_ids or lib_meta['validation_or_test_labels_in_cache'] != 0:
        raise ValueError('Molecular library TRAIN IDs differ')
    if set(lib_meta['library_smiles']) != {r['canonical_smiles'] for r in train}:
        raise ValueError('Molecular library structures differ from TRAIN')
    if lib_meta['arrays_sha256'] != sha(data / 'train_molecular_library.npz'):
        raise ValueError('Molecular library metadata checksum differs')
    expected_arrays = {n + '.npy' for n in ['fingerprints', 'descriptors', 'positive', 'hard', 'element_counts', 'fit_indices', 'calibration_indices']}
    with zipfile.ZipFile(data / 'train_molecular_library.npz') as z:
        if set(z.namelist()) != expected_arrays or z.testzip() is not None:
            raise ValueError('Molecular library archive members or CRC differ')
    print(json.dumps({'files_hash_verified': checked, 'original_training_data_files': 8,
                      'records': len(records), 'split_counts': counts,
                      'training_augmentation_records': len(augmentation_ids), 'variant_strings_by_record': variants,
                      'training_library_structures': len(lib_meta['library_smiles']),
                      'validation_or_test_records_in_training_caches': 0,
                      'both_gzip_files_fully_read': True, 'molecular_library_crc_passed': True,
                      'neural_training_or_inference_performed': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
