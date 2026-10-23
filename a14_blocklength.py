"""Step 14 (revision, item A7): sensitivity of the block-bootstrap intervals to the block length.
The reference attribution is held fixed (sampling uncertainty only; model-refit spread is in a4/a8)."""
import pandas as pd, numpy as np, json
H = pd.read_pickle('hosp2.pkl'); R = json.load(open('results.json'))
V = H.dropna(subset=['E','L','B','dLh']).copy()
E, L, d = V.E.values, V.L.values, V.dLh.values
def ind(i):
    e, l, x = E[i], L[i], d[i]; eu = np.minimum(e, l)
    return dict(heat_share=x.sum()/l.sum()*100, SHHR_pr=(eu*x/l).sum()/x.sum()*100,
                SHHR_hp=np.minimum(eu, x).sum()/x.sum()*100, SHHR_bl=np.minimum(np.clip(eu - (l - x), 0, None), x).sum()/x.sum()*100,
                HEOR=e.sum()/x.sum()*100)
pk = pd.read_csv('peaks_monthly.csv')            # monthly peak relief, resampled in blocks of months
OUT = {'point': ind(np.arange(len(V)))}
rng = np.random.default_rng(14); NBOOT = 2000   # same count as the primary design [rev A7]
t0 = V.index.min().normalize()
for days in [1, 7, 14, 28, 56]:
    blk = ((V.index.normalize() - t0).days // days).values
    ub = np.unique(blk); pos = {b: np.where(blk == b)[0] for b in ub}
    B = [ind(np.concatenate([pos[b] for b in rng.choice(ub, len(ub))])) for _ in range(NBOOT)]
    B = pd.DataFrame(B)
    OUT[f'{days}d'] = {c: [float(B[c].quantile(.025)), float(B[c].quantile(.975))] for c in B}
for mb in [1, 2, 3, 6]:                           # moving blocks of consecutive months for the peak relief
    x = pk.dP.values; n = len(x); st = np.arange(n - mb + 1)
    bs = [np.concatenate([x[s:s+mb] for s in rng.choice(st, int(np.ceil(n/mb)))])[:n].mean() for _ in range(NBOOT)]
    OUT[f'dP_{mb}mo'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
json.dump(OUT, open('results_a14.json', 'w'), indent=1)
T = pd.DataFrame({k: {c: f'{v[0]:.2f}-{v[1]:.2f}' for c, v in OUT[k].items()} for k in OUT if k.endswith('d')}).T
print(T.to_string()); print({k: np.round(v, 1).tolist() for k, v in OUT.items() if k.startswith('dP')})
