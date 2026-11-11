import numpy as np
import torch
from torch import nn
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors


DESCRIPTORS=['heavy_atoms','C','N','O','P','S','F','Cl','Br','I','rings','formal_charge','total_H','weight','chiral_centers','aromatic_atoms']

SCALES=np.asarray([50,50,10,15,5,5,10,10,10,5,10,5,100,1000,10,50],dtype=np.float32)

def candidate_features(smiles,generator):
 mol=Chem.MolFromSmiles(smiles)
 if mol is None or mol.GetNumAtoms()==0:raise ValueError('Candidate is not a nonempty valid molecule')
 canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
 fp=generator.GetFingerprint(mol);bits=np.zeros(generator.GetOptions().fpSize,dtype=np.uint8)
 DataStructs.ConvertToNumpyArray(fp,bits)
 counts={z:sum(a.GetAtomicNum()==z for a in mol.GetAtoms()) for z in [6,7,8,15,16,9,17,35,53]}
 desc=[mol.GetNumHeavyAtoms()]+[counts[z] for z in [6,7,8,15,16,9,17,35,53]]+[
  mol.GetRingInfo().NumRings(),Chem.GetFormalCharge(mol),sum(a.GetTotalNumHs() for a in mol.GetAtoms()),
  Descriptors.MolWt(mol),len(Chem.FindMolChiralCenters(mol,includeUnassigned=True)),sum(a.GetIsAromatic() for a in mol.GetAtoms())]
 desc=np.asarray(desc,dtype=np.float32)/SCALES
 if not np.isfinite(desc).all():raise ValueError('Nonfinite candidate descriptors')
 return canonical,bits,desc,fp

class CompatibilityRanker(nn.Module):
 def __init__(self,spectrum_dim,bits,hidden=256,embedding=128):
  super().__init__()
  self.spectral=nn.Sequential(nn.LayerNorm(spectrum_dim),nn.Linear(spectrum_dim,hidden),nn.GELU(),nn.Linear(hidden,embedding))
  self.molecular=nn.Sequential(nn.Linear(bits+len(DESCRIPTORS),hidden),nn.GELU(),nn.Linear(hidden,embedding))
  self.match=nn.Sequential(nn.Linear(embedding*4,hidden),nn.GELU(),nn.Linear(hidden,1))

 def forward(self,spectra,fp,desc):
  # spectra [batch, dims], candidates [batch, candidates, bits/features]
  s=self.spectral(spectra).unsqueeze(1).expand(-1,fp.size(1),-1)
  c=self.molecular(torch.cat([fp,desc],-1))
  return self.match(torch.cat([s,c,s*c,(s-c).abs()],-1)).squeeze(-1)

def standardize(values):
 x=np.asarray(values,dtype=np.float64)
 if not np.isfinite(x).all():raise ValueError('Nonfinite ranking score')
 return (x-x.mean())/max(float(x.std()),1e-6)

def peak_features(record):
 # Observed source peaks only. No target/formula/structure input.
 counts=np.asarray([len(record['nmr'].get(k,[])) for k in ['1H','13C','31P']],dtype=np.float32)
 return np.log1p(counts).astype(np.float32)

def element_counts(spelling,elements):
 mol=Chem.MolFromSmiles(spelling)
 if mol is None or not mol.GetNumAtoms():raise ValueError('Cannot count candidate/reference atoms: '+repr(spelling))
 counts={name:0 for name in elements}
 for atom in mol.GetAtoms():
  if atom.GetSymbol() in counts:counts[atom.GetSymbol()]+=1
 return np.asarray([counts[k] for k in elements],dtype=np.float32)

class ElementCountHead(nn.Module):
 def __init__(self,spectrum_dim,elements=9,hidden=256):
  super().__init__();self.norm=nn.LayerNorm(spectrum_dim)
  self.network=nn.Sequential(nn.Linear(spectrum_dim+3,hidden),nn.GELU(),nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,elements))
 def forward(self,spectra,peak_counts):return self.network(torch.cat([self.norm(spectra),peak_counts],-1))

def count_penalty(predicted_log_counts,candidate_counts,sigma,active):
 # Only predictions and proposed candidate counts enter this soft score.
 target=torch.log1p(candidate_counts);scaled=(target-predicted_log_counts)/sigma
 loss=torch.nn.functional.smooth_l1_loss(scaled,torch.zeros_like(scaled),reduction='none')
 return -(loss*active).sum(-1)/active.sum().clamp_min(1.)

def count_prediction(model,spectra,peaks,normalization):
 mean=torch.as_tensor(normalization['target_mean'],dtype=torch.float32,device=spectra.device)
 std=torch.as_tensor(normalization['target_scale'],dtype=torch.float32,device=spectra.device)
 active=torch.as_tensor(normalization['active'],dtype=torch.float32,device=spectra.device)
 return (model(spectra,peaks)*std*active+mean).clamp_min(0.)
