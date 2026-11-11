"""Predict structures for experimental 1H, 13C and 31P peak lists."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from inputs import read_inputs
from runtime import HCPAssembly
import torch
from rdkit import rdBase


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='JSON record, JSON array or JSONL records')
    parser.add_argument('--output',type=Path,required=True,help='Output JSON path')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--top-k',type=int,default=10,help='Maximum number of final ranked candidates per record')
    args=parser.parse_args()
    if args.top_k<1 or args.input.resolve()==args.output.resolve():
        parser.error('Use a positive --top-k and a separate output path.')
    records=read_inputs(args.input)
    if any(r['known_mass'] is None for r in records):
        parser.error('Supply a neutral monoisotopic mass in Da for every record.')
    model=HCPAssembly(args.device)
    spectra=model.spectral_features(records,'hcp')
    rows=[]
    for r,vec in zip(records,spectra):
        mass=r['known_mass']['value_da']
        pred=model.predict(r['nmr'],mass,vec)['final_candidates']
        rows.append({'sample_id':r['sample_id'],'known_mass_da':mass,'pool_size':len(pred),
                     'mass_matches':sum(p['mass_match'] for p in pred),
                     'candidates':[{k:p[k] for k in ['final_rank','canonical_smiles','candidate_exact_mass','mass_error_ppm','mass_match']}
                                   for p in pred[:args.top_k]]})
    report={'system':'PTrans HCP complete system','device':model.device,'torch':str(torch.__version__),
            'rdkit':rdBase.rdkitVersion,'mass_tolerance_ppm':5,'records':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with NamedTemporaryFile(mode='w',encoding='utf8',delete=False,dir=args.output.parent) as f:
        f.write(data)
    try:Path(f.name).replace(args.output)
    finally:Path(f.name).unlink(missing_ok=True)
    print(f'Saved {len(rows)} prediction(s) to {args.output}',file=sys.stderr)


if __name__=='__main__':
    main()
