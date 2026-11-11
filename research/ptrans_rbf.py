"""Nucleus-specific Gaussian shift bases added to the unchanged peak embedding."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

FORMAT='ptrans_rbf_shift_aug_v1'
ORIGINAL_HASHES={
 'ptrans_train_baseline.py':'a9ed2625b53c19d7299a9b3cb59768fab16b9946b278d3d6678d04d0553e4b07',
 'ptrans_train_arch.py':'97cee65688b77a636181b63831060f977e789d54499f3ba8d3a0445245587aca',
 'ptrans_smiles_aug.py':'e68498b82cb13344dd6e72001d35904d26ab08fc7297f2ff31c0b309fbdf74b1',
 'ptrans_train_smiles_aug.py':'720de709aa4cb3db8eda540b33ff91994b480c2643f68be35173010175b6a6c7'}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def load_module(root,filename,name):
 sys.path.insert(0,str(root))
 spec=importlib.util.spec_from_file_location(name,Path(root)/filename)
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return m

def check_originals(root):
 for name,sha in ORIGINAL_HASHES.items():
  if digest(Path(root)/name)!=sha:raise ValueError('Original source changed: '+name)

def load_arch(root,name='ptrans_rbf_arch'):
 check_originals(root)
 return load_module(root,'ptrans_train_arch.py',name)

def fit_bases(records,base,config):
 """Fit ranges on training peaks only; do not use targets or val/test shifts."""
 if set(config)!={'basis_count','bandwidth_multipliers','quantiles','range_margin'}:
  raise ValueError('Invalid RBF settings')
 k=config['basis_count'];qs=config['quantiles'];mult=config['bandwidth_multipliers'];margin=config['range_margin']
 if type(k)!=int or not 4<=k<=256 or len(qs)!=2 or not 0<=qs[0]<qs[1]<=1:
  raise ValueError('Invalid RBF range/count')
 if not mult or not all(np.isfinite(x) and x>0 for x in mult) or not np.isfinite(margin) or margin<0:
  raise ValueError('Invalid RBF widths/margin')
 values=[[],[],[]]
 for r in records:
  if r['split_scaffold']!='train':continue
  for name,idx in base.NUCLEUS_IDS.items():
   for peak in r['nmr'].get(name,[]):
    if peak.get('shift_min_ppm') is None and peak.get('shift_max_ppm') is None:continue
    values[idx].append(base.peak_to_features(peak)[0])
 centers=[];widths=[];ranges=[]
 for vs in values:
  if not vs or not np.isfinite(vs).all():raise ValueError('Need finite training shifts for all three nuclei')
  lo,hi=np.quantile(vs,qs);span=max(float(hi-lo),0.01)
  lo=float(lo-span*margin);hi=float(hi+span*margin)
  if hi-lo<0.01:lo-=0.005;hi+=0.005
  centers.append(np.linspace(lo,hi,k).tolist())
  widths.append([float((hi-lo)/(k-1)*m) for m in mult])
  ranges.append({'normalized_min':lo,'normalized_max':hi,'training_peaks':len(vs)})
 return centers,widths,ranges

def validate_bases(centers,widths):
 c=np.asarray(centers);w=np.asarray(widths)
 if c.ndim!=2 or c.shape[0]!=3 or c.shape[1]<4 or w.ndim!=2 or w.shape[0]!=3 or not w.shape[1]:
  raise ValueError('RBF centers/widths must have three nucleus rows')
 if not np.isfinite(c).all() or not np.isfinite(w).all() or (w<=0).any() or not (np.diff(c,axis=1)>0).all():
  raise ValueError('Invalid RBF basis values')

def model_class(arch,centers=None,widths=None):
 class RBFModel(arch.PTransModel):
  def __init__(self,*args,rbf_centers=centers,rbf_widths=widths,**kwargs):
   validate_bases(rbf_centers,rbf_widths)
   super().__init__(*args,**kwargs)
   self.register_buffer('rbf_centers',torch.tensor(rbf_centers,dtype=torch.float32))
   self.register_buffer('rbf_widths',torch.tensor(rbf_widths,dtype=torch.float32))
   n=self.rbf_centers.size(1)*self.rbf_widths.size(1)
   # Zero initialization is exact baseline initialization and consumes no RNG.
   self.rbf_projection=nn.Parameter(torch.zeros(3,n,self.d_model))

  def shift_embedding(self,features,nucleus):
   x=features[:,:,0,None,None]
   c=self.rbf_centers[nucleus].unsqueeze(-2)
   w=self.rbf_widths[nucleus].unsqueeze(-1)
   phi=torch.exp(-0.5*((x-c)/w).square()).flatten(-2)
   out=torch.zeros((*features.shape[:2],self.d_model),device=features.device,dtype=features.dtype)
   for n in range(3):out=out+(phi@self.rbf_projection[n])*(nucleus==n).unsqueeze(-1)
   return out

  def encode(self,features,nucleus,mult,padding):
   if self.architecture!='shared_mlp':raise ValueError('RBF experiment requires shared_mlp')
   source=self.peak_projection(features)+self.peak_residuals[0](features)
   source=source+self.shift_embedding(features,nucleus)+self.nucleus_embedding(nucleus)+self.mult_embedding(mult)
   return self.encoder(source,src_key_padding_mask=padding)
 return RBFModel

def validate_checkpoint(state):
 if state.get('format_version')!=FORMAT:raise ValueError('Not an RBF checkpoint')
 p=state['rbf_encoding'];a=state['model_args']
 validate_bases(a['rbf_centers'],a['rbf_widths'])
 if a['rbf_centers']!=p['centers'] or a['rbf_widths']!=p['widths'] or p['model_helper_sha256']!=digest(__file__):
  raise ValueError('RBF architecture/provenance differs')

