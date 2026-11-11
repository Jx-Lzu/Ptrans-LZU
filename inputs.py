"""Reference-free NMR and known-mass JSON input validation."""
import json, math
from pathlib import Path
def positive_number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be a finite positive number')
    return float(value)

def validate_mass(value):
    if value is None:return None
    if not isinstance(value,dict) or set(value)-{'value_da','type','source'}:raise ValueError('known_mass must contain value_da, type and optional source')
    if value.get('type')!='monoisotopic_mass':
        raise ValueError('known_mass.type must be monoisotopic_mass; raw m/z and average molecular weight are not accepted')
    source=value.get('source','user_supplied')
    if not isinstance(source,str):raise ValueError('known_mass.source must be text')
    return {'value_da':positive_number(value.get('value_da'),'known_mass.value_da'),'type':'monoisotopic_mass','source':source}

NUCLEI=('1H','13C','31P')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def finite(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError(name+' must be a finite number')
    return float(value)
def normalize_record(r,index=1):
    if not isinstance(r,dict) or set(r)-{'sample_id','nmr','known_mass'}:raise ValueError('Input accepts only sample_id, nmr and known_mass; no reference structure or formula')
    sid=r.get('sample_id',f'sample_{index:04d}')
    if not isinstance(sid,str) or not sid.strip():raise ValueError('sample_id must be nonempty text')
    nmr=r.get('nmr')
    if not isinstance(nmr,dict) or set(nmr)!=set(NUCLEI):raise ValueError('Provide all three nmr lists: 1H, 13C and 31P')
    clean={}
    allowed={'nucleus','shift_ppm','shift_min_ppm','shift_max_ppm','integral_protons','multiplicity','coupling_hz'}
    for nuc in NUCLEI:
        peaks=nmr[nuc]
        if not isinstance(peaks,list) or not peaks:raise ValueError(nuc+' must have a nonempty peak list for this HCP model')
        normalized=[]
        for i,peak in enumerate(peaks,1):
            label=f'{sid}/{nuc}/peak{i}'
            if isinstance(peak,(int,float)) and not isinstance(peak,bool):peak={'shift_ppm':peak}
            if not isinstance(peak,dict) or set(peak)-allowed:raise ValueError(label+': unsupported peak fields')
            if peak.get('nucleus',nuc)!=nuc:raise ValueError(label+': nucleus disagrees with its list')
            if 'shift_ppm' in peak:
                if 'shift_min_ppm' in peak or 'shift_max_ppm' in peak:raise ValueError(label+': use shift_ppm or min/max, not both')
                lo=hi=finite(peak['shift_ppm'],label+'/shift')
            else:
                if 'shift_min_ppm' not in peak or 'shift_max_ppm' not in peak:raise ValueError(label+': shift is required')
                lo=finite(peak['shift_min_ppm'],label+'/min');hi=finite(peak['shift_max_ppm'],label+'/max')
                if lo>hi:raise ValueError(label+': min exceeds max')
            integral=peak.get('integral_protons')
            if integral is not None:
                integral=finite(integral,label+'/integral')
                if integral<0:raise ValueError(label+': negative integral')
            mult=peak.get('multiplicity')
            if mult is not None and not isinstance(mult,str):raise ValueError(label+': multiplicity must be text or null')
            couplings=peak.get('coupling_hz',[])
            if couplings is None:couplings=[]
            if not isinstance(couplings,list):raise ValueError(label+': coupling_hz must be a number list')
            couplings=[finite(v,label+'/J') for v in couplings]
            normalized.append({'nucleus':nuc,'shift_min_ppm':lo,'shift_max_ppm':hi,'integral_protons':integral,'multiplicity':mult,'coupling_hz':couplings})
        clean[nuc]=normalized
    return {'sample_id':sid,'nmr':clean,'known_mass':validate_mass(r.get('known_mass'))}

def read_inputs(path):
    path=Path(path)
    if path.suffix.lower()=='.jsonl':raw=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    else:
        raw=read(path);raw=raw if isinstance(raw,list) else [raw]
    if not raw:raise ValueError('No input records')
    data=[normalize_record(r,i) for i,r in enumerate(raw,1)]
    if len({r['sample_id'] for r in data})!=len(data):raise ValueError('Duplicate sample_id')
    return data

