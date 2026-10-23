"""Step 18 (revision, item A7): block-length sensitivity under the SAME design as the primary analysis
(season-stratified block bootstrap of calendar days). The primary length is fixed in advance by the Politis-White
selector on the deseasonalised daily load (bootdraws.py); here only the block length changes, to half and twice the
primary length and to two neighbouring values. The reference attribution is held fixed (sampling only), so every row,
including the primary one, measures the same quantity. Block length is not chosen from these results."""
import pandas as pd, numpy as np, json
from bootdraws import T0, SEASON, NDRAW
H = pd.read_pickle('hosp2.pkl'); design = json.load(open('boot_design.json')); B0 = design['b_days']
V = H.dropna(subset=['E','L','B','dLh']).copy()
E, L, d = V.E.values, V.L.values, V.dLh.values; Eu = np.minimum(E, L)
dn = ((V.index.normalize() - T0).days).values; ndays = int(dn.max()) + 1
pk = pd.read_csv('peaks_monthly.csv'); pkm = pk.month.astype(str).values; dP = pk.dP.values
day_month = (T0 + pd.to_timedelta(np.arange(ndays), unit='D')).strftime('%Y-%m')
month_len = pd.Series(day_month).value_counts()
def ind(i):
    e, eu, l, x = E[i], Eu[i], L[i], d[i]
    return dict(heat_share=x.sum()/l.sum()*100, SHHR_pr=(eu*x/l).sum()/x.sum()*100,
                SHHR_hp=np.minimum(eu, x).sum()/x.sum()*100, HEOR=e.sum()/x.sum()*100)
def run(b, seed):
    block_of_day = np.arange(ndays) // b; nblk = block_of_day.max() + 1
    mid = {k: SEASON[(T0 + pd.Timedelta(days=int(np.median(np.where(block_of_day == k)[0])))).month] for k in range(nblk)}
    strata = {s: np.array([k for k in range(nblk) if mid[k] == s]) for s in sorted(set(mid.values()))}
    pos = pd.Series(np.arange(len(dn))).groupby(block_of_day[dn]).apply(np.array).to_dict(); empty = np.array([], int)
    days_of = {k: np.where(block_of_day == k)[0] for k in range(nblk)}
    rng = np.random.default_rng(seed); rows, dps = [], []
    for _ in range(NDRAW):
        draw = np.concatenate([rng.choice(ks, len(ks)) for ks in strata.values()])
        rows.append(ind(np.concatenate([pos.get(k, empty) for k in draw])))
        cnt = pd.Series(day_month[np.concatenate([days_of[k] for k in draw])]).value_counts()
        wt = np.array([cnt.get(m, 0)/month_len[m] for m in pkm]); dps.append((wt*dP).sum()/wt.sum())
    Bt = pd.DataFrame(rows); Bt['dP'] = dps
    return dict(b=b, n_blocks=int(nblk), strata={s: int(len(k)) for s, k in strata.items()},
                ci={c: [float(Bt[c].quantile(.025)), float(Bt[c].quantile(.975))] for c in Bt})
OUT = {'primary_b': B0, 'design': 'season-stratified block bootstrap, reference attribution fixed', 'rows': []}
for b in sorted({B0 // 2, 14, B0, 28, 2*B0}):
    r = run(b, 2026 + b); r['role'] = 'primary' if b == B0 else 'sensitivity'; OUT['rows'].append(r)
json.dump(OUT, open('results_a18.json', 'w'), indent=1)
T = pd.DataFrame([{**{'b (d)': r['b'], 'role': r['role'], 'blocks': r['n_blocks']},
                   **{c: f"{v[0]:.2f}–{v[1]:.2f}" for c, v in r['ci'].items()}} for r in OUT['rows']])
pd.set_option('display.width', 220); print(T.to_string(index=False))
