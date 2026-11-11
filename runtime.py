"""Load the released HCP safetensors and run the archived three-source scoring procedure."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase
from rdkit.Chem import rdFingerprintGenerator
from functools import lru_cache
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "research"
sys.path.insert(0, str(CORE))
import ptrans_rbf as rbf
from system_core import (
    FrozenAssembly, Generator, CompatibilityRanker, ElementCountHead, ELEMENTS,
    sha, weights_sha, candidate_features, element_counts,
)


def load_tensors(spec):
    path = (ROOT / spec['path']).resolve()
    if not path.is_relative_to(ROOT) or sha(path) != spec['sha256']:
        raise ValueError('Missing or changed release weights: '+spec['path'])
    state = load_file(str(path),device='cpu')
    if weights_sha(state) != spec['tensor_sha256']:
        raise ValueError('Tensor fingerprint mismatch: '+spec['path'])
    return state


class ReleasedGenerator(Generator):
    def __init__(self, arch, spec, vocabulary, device):
        self.base = arch.base
        self.device = device
        self.vocab = vocabulary['vocab']
        self.inverse = vocabulary['inverse_vocab']
        self.mult = vocabulary['multiplicity_vocab']
        self.args = spec['model_args']
        self.model = rbf.model_class(arch)(
            len(self.vocab),len(self.mult),pad_id=self.vocab['<pad>'],**self.args
        ).to(device).float().eval().requires_grad_(False)
        self.model.load_state_dict(load_tensors(spec),strict=True)


class HCPAssembly(FrozenAssembly):
    def __init__(self, device='auto'):
        config = json.loads((ROOT/'model_config.json').read_text(encoding='utf-8'))
        self.config = config['scoring']
        if [self.config[k] for k in ('alpha','beta','gamma','mass_tolerance_ppm')] != [.6,.25,.25,5.]:
            raise ValueError('Unsupported ranking configuration')
        if rdBase.rdkitVersion != '2025.09.4':
            raise ValueError('Use rdkit==2025.9.4 to match the published chemistry pipeline.')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        if self.device not in ('cpu','cuda') or self.device=='cuda' and not torch.cuda.is_available():
            raise ValueError('Requested device unavailable: '+self.device)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.arch = rbf.load_arch(CORE,'ptrans_released_arch')
        self.a,self.b,self.extra = [ReleasedGenerator(self.arch,config['components'][name],config['vocabulary'],self.device)
                                     for name in ['primary','secondary','extra']]
        for key,cls in [('ranker',CompatibilityRanker),('element',ElementCountHead)]:
            meta=config['components'][key]
            head=cls(**meta['model_args']).to(self.device).float().eval().requires_grad_(False)
            head.load_state_dict(load_tensors(meta),strict=True)
            setattr(self,key,head)
            setattr(self,key+'_state',meta)
        self.fp_generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
        self.features=lru_cache(maxsize=30000)(lambda sm:candidate_features(sm,self.fp_generator))
        self.counts=lru_cache(maxsize=30000)(lambda sm:element_counts(sm,ELEMENTS))
