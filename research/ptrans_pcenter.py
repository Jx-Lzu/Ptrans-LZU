"""Root-marked phosphorus local graphs and strictly train-only label vocabulary."""
import argparse,csv,gzip,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit import Chem,rdBase,RDLogger

ROOT=Path(__file__).resolve().parent
FORMAT='ptrans_phosphorus_center_v1'
SPEC={'version':1,'radii':[1,2],'ball':'heavy-atom shortest-path distance <=radius, induced bonds among selected atoms','atom_attributes':'element,formal charge,heavy-atom isotope,aromaticity,original total attached-H count; radical-electron count ignored','root':'central P atom map1; all other atom maps0','stereochemistry':'ignored; stereo removed before isomeric serialization that retains heavy isotopes','hydrogen':'implicit/explicit/isotopic hydrogen neighbors reduced to total attached-H count; H isotope identity is not retained','serialization':'canonical RDKit SMILES, all atoms/H counts and bonds explicit, unsanitized capped local graph; opaque identity key, not necessarily a valid standalone molecule','multi_P':'sorted multiset, duplicates retained; no spectrum-to-atom assignments','unselected_components':'outside the graph ball, including disconnected counterions, excluded'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def inside(value):
 path=(ROOT/value).resolve()
 if not path.is_relative_to(ROOT.resolve()):raise ValueError('Path escapes project')
 return path

def environment(mol,center,radius):
 if radius not in [1,2] or mol.GetAtomWithIdx(center).GetAtomicNum()!=15:raise ValueError('Use P center and radius1/2')
 distances={center:0};front=[center]
 for depth in range(1,radius+1):
  following=[]
  for idx in front:
   for other in mol.GetAtomWithIdx(idx).GetNeighbors():
    j=other.GetIdx()
    if other.GetAtomicNum()>1 and j not in distances:distances[j]=depth;following.append(j)
  front=following
 selected=sorted(distances);mapping={};graph=Chem.RWMol()
 for idx in selected:
  original=mol.GetAtomWithIdx(idx);atom=Chem.Atom(original);atom.SetAtomMapNum(1 if idx==center else 0);atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED);atom.SetNoImplicit(True);atom.SetNumExplicitHs(original.GetTotalNumHs(includeNeighbors=True));atom.SetNumRadicalElectrons(0);mapping[idx]=graph.AddAtom(atom)
 for bond in mol.GetBonds():
  a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
  if a in mapping and b in mapping:
   graph.AddBond(mapping[a],mapping[b],bond.GetBondType());graph.GetBondBetweenAtoms(mapping[a],mapping[b]).SetIsAromatic(bond.GetIsAromatic())
 local=graph.GetMol();local.UpdatePropertyCache(strict=False);Chem.GetSymmSSSR(local);Chem.RemoveStereochemistry(local)
 return Chem.MolToSmiles(local,canonical=True,isomericSmiles=True,allHsExplicit=True,allBondsExplicit=True)

class Environments:
 def __init__(self):self.cache={}
 def get(self,smiles):
  if smiles not in self.cache:
   try:
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():self.cache[smiles]=None
    else:
     centers=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==15]
     self.cache[smiles]={str(radius):tuple(sorted(environment(mol,i,radius) for i in centers)) for radius in [1,2]}
   except Exception:self.cache[smiles]=None
  return self.cache[smiles]

def load_records(cfg):
 path=inside(cfg['data']);opener=gzip.open if path.suffix=='.gz' else open
 with opener(path,'rt',encoding='utf-8') as f:records=[json.loads(line) for line in f if line.strip()]
 with inside(cfg['manifest']).open(encoding='utf-8-sig',newline='') as f:manifest_rows=list(csv.DictReader(f))
 manifest={r['sample_id']:r for r in manifest_rows}
 if len(manifest)!=len(manifest_rows) or len(records)!=len(manifest) or len({r['sample_id'] for r in records})!=len(records):raise ValueError('Duplicate or missing data/manifest IDs')
 h=hashlib.sha256();structures={};groups={}
 for r in records:
  m=manifest.get(r['sample_id'])
  if m is None or any(r[k]!=m[k] for k in ['canonical_smiles','split_scaffold','split_random_structure']):raise ValueError('Data/manifest differ')
  for table,key in [(structures,r['canonical_smiles']),(groups,m['scaffold_group_key'])]:
   if key in table and table[key]!=r['split_scaffold']:raise ValueError('Cross-split structure/group overlap')
   table[key]=r['split_scaffold']
  h.update(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode());h.update(b'\n')
 return records,h.hexdigest()

def dictionaries(records,minimum,maximum,extractor):
 train=[r for r in records if r['split_scaffold']=='train'];unique=sorted({r['canonical_smiles'] for r in train});freq={str(r):Counter() for r in [1,2]}
 for sm in unique:
  env=extractor.get(sm)
  if env is None or not env['1']:raise ValueError('TRAIN must have valid phosphorus-containing structures')
  for radius in ['1','2']:freq[radius].update(set(env[radius]))
 vocab={}
 for radius in ['1','2']:
  chosen=sorted((s for s,count in freq[radius].items() if count>=minimum),key=lambda s:(-freq[radius][s],s))[:maximum]
  if not chosen:raise ValueError('No supported TRAIN environments; adjust minimum support before training')
  vocab[radius]={'keys':chosen,'support_unique_train_structures':[freq[radius][s] for s in chosen],'other_index':len(chosen),'all_unique_train_environments':len(freq[radius])}
 return vocab

def label_arrays(train,vocab,extractor):
 arrays={}
 for radius in ['1','2']:
  keys=vocab[radius]['keys'];lookup={s:i for i,s in enumerate(keys)};y=np.zeros((len(train),len(keys)+1),dtype=np.uint8)
  for i,r in enumerate(train):
   for value in set(extractor.get(r['canonical_smiles'])[radius]):y[i,lookup.get(value,len(keys))]=1
  counts=y.sum(0,dtype=np.int64);active=(counts>0)&(counts<len(train));weights=np.clip((len(train)-counts)/np.maximum(counts,1),1/20,20).astype(np.float32)
  arrays['labels_r'+radius]=y;arrays['active_r'+radius]=active;arrays['weights_r'+radius]=weights;arrays['counts_r'+radius]=counts
 return arrays

def prepare(cfg):
 data=inside(cfg['cache_data']);meta=inside(cfg['cache_meta'])
 if data.exists() or meta.exists():raise FileExistsError('Cache already exists; verify it instead of overwriting')
 records,fp=load_records(cfg);train=[r for r in records if r['split_scaffold']=='train'];extractor=Environments();vocab=dictionaries(records,cfg['minimum_unique_support'],cfg['maximum_labels_per_radius'],extractor);arrays=label_arrays(train,vocab,extractor);arrays['sample_ids']=np.array([r['sample_id'] for r in train]);data.parent.mkdir(parents=True,exist_ok=True)
 with data.open('wb') as f:np.savez_compressed(f,**arrays)
 info={'format':FORMAT,'specification':SPEC,'dataset_fingerprint':fp,'cache_sha256':digest(data),'helper_sha256':digest(__file__),'rdkit_version':rdBase.rdkitVersion,'training_records':len(train),'training_unique_structures':len({r['canonical_smiles'] for r in train}),'dictionary':vocab,'dictionary_source':'original TRAIN unique structures only; no val/test frequency selection','minimum_unique_support':cfg['minimum_unique_support'],'maximum_labels_per_radius':cfg['maximum_labels_per_radius'],'class_statistics':'original TRAIN record counts, positive weights clipped[1/20,20]','auxiliary_target':'multi-label environment presence +OTHER; not multiplicity or peak-to-atom assignment','validation_or_test_labels_in_cache':0,'source_hashes':{cfg['data']:digest(inside(cfg['data'])),cfg['manifest']:digest(inside(cfg['manifest']))}}
 dump(meta,info);return info

def load_cache(cfg):
 records,fp=load_records(cfg);info=read(inside(cfg['cache_meta']));path=inside(cfg['cache_data'])
 if info['format']!=FORMAT or info['specification']!=SPEC or info['dataset_fingerprint']!=fp or info['cache_sha256']!=digest(path) or info['helper_sha256']!=digest(__file__) or info['rdkit_version']!=rdBase.rdkitVersion or info['validation_or_test_labels_in_cache']!=0:raise ValueError('P-center cache/protocol identity differs')
 if info['minimum_unique_support']!=cfg['minimum_unique_support'] or info['maximum_labels_per_radius']!=cfg['maximum_labels_per_radius']:raise ValueError('Dictionary policy differs')
 for name,sha in info['source_hashes'].items():
  if digest(inside(name))!=sha:raise ValueError('Original data source differs')
 with np.load(path,allow_pickle=False) as z:arrays={name:z[name].copy() for name in z.files}
 train=[r for r in records if r['split_scaffold']=='train']
 if arrays['sample_ids'].tolist()!=[r['sample_id'] for r in train]:raise ValueError('P-center labels must be TRAIN-only and ordered')
 extractor=Environments();vocab=dictionaries(records,cfg['minimum_unique_support'],cfg['maximum_labels_per_radius'],extractor)
 if vocab!=info['dictionary']:raise ValueError('Recomputed TRAIN dictionary differs')
 expected=label_arrays(train,vocab,extractor)
 if set(arrays)!=set(expected)|{'sample_ids'} or any(not np.array_equal(arrays[k],v) for k,v in expected.items()):raise ValueError('Recomputed TRAIN local labels/statistics differ')
 return records,arrays,info

def head_scores(logits,weights):return 1/(1+np.exp(-np.clip(np.asarray(logits,dtype=float)-np.log(weights),-60,60)))

def args(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/pcenter_v1.json');p.add_argument('--prepare',action='store_true');a=p.parse_args(argv);return a,read(inside(a.config))

if __name__=='__main__':
 a,cfg=args()
 if a.prepare:info=prepare(cfg)
 else:_,_,info=load_cache(cfg)
 print(json.dumps({'training_records':info['training_records'],'R1_known_classes':len(info['dictionary']['1']['keys']),'R2_known_classes':len(info['dictionary']['2']['keys']),'train_labels_only':True,'GPU_training_or_scoring_performed':False},indent=2))
