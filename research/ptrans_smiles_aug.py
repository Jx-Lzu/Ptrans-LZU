"""Training-only, deterministic equivalent-SMILES helpers; no Torch dependency."""
import ast
import csv
import gzip
import hashlib
import io
import json
import re
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase

RDLogger.DisableLog('rdApp.error')
FORMAT = 'ptrans_smiles_aug_train_variants_v1'
BASE_SHA256 = 'a9ed2625b53c19d7299a9b3cb59768fab16b9946b278d3d6678d04d0553e4b07'
ARCH_SHA256 = '97cee65688b77a636181b63831060f977e789d54499f3ba8d3a0445245587aca'
SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(records):
    h=hashlib.sha256()
    for r in records:
        h.update(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def load_records(data, manifest):
    opener=gzip.open if Path(data).suffix=='.gz' else open
    with opener(data,'rt',encoding='utf-8') as f:records=[json.loads(line) for line in f if line.strip()]
    with Path(manifest).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    by_id={r['sample_id']:r for r in rows}
    if len(rows)!=len(records) or len(by_id)!=len(rows):raise ValueError('Dataset/manifest count mismatch')
    seen=set()
    for r in records:
        sid=r['sample_id']
        if sid in seen or sid not in by_id:raise ValueError('Duplicate/missing dataset sample ID')
        seen.add(sid)
        for field in ['canonical_smiles','split_scaffold','split_random_structure']:
            if r[field]!=by_id[sid][field]:raise ValueError('Dataset/manifest mismatch: '+sid)
    return records


def token_pattern(project_root):
    path=Path(project_root)/'ptrans_train_baseline.py'
    if digest(path)!=BASE_SHA256:raise ValueError('Keep the original baseline script unchanged')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='TOKEN_RE' for t in n.targets))
    return re.compile(ast.literal_eval(node.value.args[0]))


def tokens(pattern, smiles):
    result=pattern.findall(smiles)
    if ''.join(result)!=smiles:raise ValueError('SMILES tokenizer did not round-trip')
    return result


def vocabulary(train,pattern):
    found={t for r in train for t in tokens(pattern,r['canonical_smiles'])}
    return SPECIALS+sorted(found-set(SPECIALS))


def vocabulary_hash(inverse):
    return hashlib.sha256(json.dumps(inverse,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def seeded_integer(text):
    return (int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')&0x7fffffff) or 1


def choose_target(sid, canonical, variants, epoch, probability, seed):
    """Independent per-record/epoch choice; no global RNG state is consumed."""
    h=hashlib.sha256(f'{seed}|{epoch}|{sid}'.encode()).digest()
    requested=int.from_bytes(h[:8],'little')/(2**64)<probability
    if requested and variants:
        return variants[int.from_bytes(h[8:16],'little')%len(variants)],'augmented'
    return canonical,'fallback_no_variants' if requested else 'canonical'


def build_variants(records,pattern,max_target_tokens=384,seed=42,count=4,attempts=24):
    train=[r for r in records if r['split_scaffold']=='train']
    if not train:raise ValueError('No training records')
    inverse=vocabulary(train,pattern);allowed=set(inverse)
    by_structure={};entries=[];rejections={'same_spelling':0,'non_equivalent':0,'unknown_token':0,'too_long':0}
    for index,r in enumerate(train,1):
        canonical=r['canonical_smiles']
        if canonical not in by_structure:
            mol=Chem.MolFromSmiles(canonical)
            if mol is None or Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)!=canonical:
                raise ValueError('Training reference is not canonical under this RDKit: '+r['sample_id'])
            candidates=Chem.MolToRandomSmilesVect(mol,attempts,
                randomSeed=seeded_integer(f'{seed}|{canonical}'),isomericSmiles=True)
            variants=[];seen={canonical}
            for smiles in candidates:
                if smiles in seen:rejections['same_spelling']+=1;continue
                seen.add(smiles)
                ts=tokens(pattern,smiles)
                if set(ts)-allowed:rejections['unknown_token']+=1;continue
                if len(ts)+2>max_target_tokens:rejections['too_long']+=1;continue
                parsed=Chem.MolFromSmiles(smiles)
                if parsed is None or Chem.MolToSmiles(parsed,canonical=True,isomericSmiles=True)!=canonical:
                    rejections['non_equivalent']+=1;continue
                variants.append(smiles)
                if len(variants)==count:break
            by_structure[canonical]=variants
        entries.append({'sample_id':r['sample_id'],'canonical_smiles':canonical,
                        'variants':by_structure[canonical]})
        if index%1000==0 or index==len(train):
            print(f'Prepare training SMILES: {index}/{len(train)}',flush=True)
    metadata={'format':FORMAT,'dataset_fingerprint':fingerprint(records),'seed':seed,
        'rdkit_version':rdBase.rdkitVersion,'training_records':len(train),
        'distinct_training_structures':len(by_structure),'max_target_tokens':max_target_tokens,
        'variants_requested_per_structure':count,'random_attempts_per_structure':attempts,
        'records_with_variants':sum(bool(r['variants']) for r in entries),
        'variant_strings_by_record':sum(len(r['variants']) for r in entries),
        'rejected_unique_structure_candidates':rejections,
        'vocabulary_sha256':vocabulary_hash(inverse),'vocabulary_size':len(inverse),
        'training_only':True,'stereochemistry_retained':True,'vocabulary_expanded':False,
        'target_truncation':False,'new_training_performed':False}
    return entries,metadata


def save_cache(data_path,meta_path,entries,metadata):
    data_path,meta_path=Path(data_path),Path(meta_path)
    if data_path.exists() or meta_path.exists():raise FileExistsError('Choose new augmentation cache paths')
    data_path.parent.mkdir(parents=True,exist_ok=True);meta_path.parent.mkdir(parents=True,exist_ok=True)
    with data_path.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='\n') as f:
                for item in entries:f.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
    metadata={**metadata,'cache_sha256':digest(data_path),'cache_bytes':data_path.stat().st_size}
    with meta_path.open('x',encoding='utf-8') as f:json.dump(metadata,f,indent=2);f.write('\n')
    return metadata


def load_cache(data_path,meta_path,records,pattern,max_target_tokens,verify_structures=True):
    metadata=json.loads(Path(meta_path).read_text(encoding='utf-8'))
    if metadata['format']!=FORMAT or metadata['dataset_fingerprint']!=fingerprint(records):
        raise ValueError('Augmentation cache format/dataset fingerprint mismatch')
    if metadata['max_target_tokens']!=max_target_tokens or digest(data_path)!=metadata['cache_sha256']:
        raise ValueError('Augmentation cache target capacity/hash mismatch')
    train={r['sample_id']:r for r in records if r['split_scaffold']=='train'}
    inverse=vocabulary(list(train.values()),pattern);allowed=set(inverse)
    if vocabulary_hash(inverse)!=metadata['vocabulary_sha256']:raise ValueError('Training vocabulary changed')
    by_id={};verified={};variant_count=0
    with gzip.open(data_path,'rt',encoding='utf-8') as f:
        for line in f:
            item=json.loads(line);sid=item['sample_id'];canonical=item['canonical_smiles']
            if sid not in train or sid in by_id or canonical!=train[sid]['canonical_smiles']:
                raise ValueError('Augmentation contains nontraining, duplicate or wrong reference: '+sid)
            variants=item['variants']
            if len(set(variants))!=len(variants):raise ValueError('Duplicate augmentation spelling')
            for smiles in variants:
                ts=tokens(pattern,smiles)
                if smiles==canonical or set(ts)-allowed or len(ts)+2>max_target_tokens:
                    raise ValueError('Invalid, unknown-token or overlength augmentation')
                if verify_structures:
                    if smiles not in verified:
                        mol=Chem.MolFromSmiles(smiles)
                        verified[smiles]=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True) if mol else None
                    if verified[smiles]!=canonical:raise ValueError('Augmentation changed molecular structure/stereochemistry')
            by_id[sid]=variants;variant_count+=len(variants)
    if set(by_id)!=set(train) or len(train)!=metadata['training_records']:
        raise ValueError('Augmentation cache does not cover exactly the training split')
    if variant_count!=metadata['variant_strings_by_record'] or sum(bool(v) for v in by_id.values())!=metadata['records_with_variants']:
        raise ValueError('Augmentation count mismatch')
    return by_id,metadata,inverse
