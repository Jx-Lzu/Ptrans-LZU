"""Paired scaffold bootstrap on cached outputs; never fits or selects a model."""
import csv,json
import numpy as np
import retrain_queue as q
def paired(a,b,ka,kb):
    if [r['sample_id'] for r in a]!=[r['sample_id'] for r in b]:raise ValueError('Paired IDs differ')
    groups={}
    for i,r in enumerate(a):groups.setdefault(r['scaffold_group_key'],[]).append(i)
    clusters=list(groups.values());sizes=np.array([len(x) for x in clusters]);rng=np.random.default_rng(42);result={}
    for k in [1,5,10]:
        aa=np.array([0<int(r[ka])<=k for r in a]);bb=np.array([0<int(r[kb])<=k for r in b]);delta=bb.astype(int)-aa.astype(int)
        sums=np.array([delta[ii].sum() for ii in clusters]);boots=[]
        for _ in range(2000):
            ii=rng.integers(len(clusters),size=len(clusters));boots.append(float(sums[ii].sum()/sizes[ii].sum()))
        result[str(k)]={'gained':int((~aa&bb).sum()),'lost':int((aa&~bb).sum()),'net':int(delta.sum()),'delta':float(delta.mean()),'CI95':np.quantile(boots,[.025,.975]).tolist()}
    return {'samples':len(a),'scaffold_clusters':len(clusters),'method':'paired cluster bootstrap; 2000 resamples of operational scaffold groups; not seed uncertainty','topk':result}
def main():
    c=q.config();out=q.output();result={}
    for split in ['val','test']:
        rows={}
        for g in c['groups']:
            p=out/g/split/'per_sample_results.csv'
            if not p.exists():continue
            with p.open(encoding='utf-8-sig',newline='') as f:rows[g]=list(csv.DictReader(f))
        for g,a in rows.items():result[f'{split}_{g}_mass']=paired(a,a,'rank_before_mass','rank_after_mass')
        for a,b in [('hc','hcp'),('c','cp'),('h','hp')]:
            if a not in rows or b not in rows:continue
            for mode in ['before_mass','after_mass']:result[f'{split}_{a}_to_{b}_{mode}']=paired(rows[a],rows[b],'rank_'+mode,'rank_'+mode)
    q.dump(out/'paired_comparisons.json',result);print('PAIRED_COMPARISONS '+str(out/'paired_comparisons.json'))
if __name__=='__main__':main()
