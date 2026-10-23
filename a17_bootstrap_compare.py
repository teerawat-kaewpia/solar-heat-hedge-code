"""Step 17 (revision, item A7): compare bootstrap schemes for the sampling intervals of the headline indicators,
with the reference attribution held fixed: 28-day blocks, calendar-month stratified, season-stratified 28-day blocks,
and Politis-White (2004, with the Patton-Politis-White 2009 correction) automatic block lengths."""
import pandas as pd, numpy as np, json
H = pd.read_pickle('hosp2.pkl')
V = H.dropna(subset=['E','L','B','dLh']).copy()
E, L, d = V.E.values, V.L.values, V.dLh.values; Eu = np.minimum(E, L)
day = V.index.normalize(); T0 = pd.Timestamp('2023-03-03')
dnum = ((day - T0).days).values
SEASON = {11: 'cool', 12: 'cool', 1: 'cool', 2: 'cool', 3: 'hot', 4: 'hot', 5: 'hot', 6: 'wet', 7: 'wet', 8: 'wet', 9: 'wet', 10: 'wet'}
def ind(i):
    e, eu, l, x = E[i], Eu[i], L[i], d[i]
    return dict(heat_share=x.sum()/l.sum()*100, SHHR_pr=(eu*x/l).sum()/x.sum()*100,
                SHHR_hp=np.minimum(eu, x).sum()/x.sum()*100, HEOR=e.sum()/x.sum()*100)
point = ind(np.arange(len(V)))

# ---------- Politis-White automatic block length ----------
def pw_block(x):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    def acov(k): return (x[:n-k]*x[k:]).sum()/n
    r0 = acov(0); KN = max(5, int(np.ceil(np.sqrt(np.log10(n))))); c = 2*np.sqrt(np.log10(n)/n)
    mmax = int(np.ceil(np.sqrt(n))) + KN
    rho = np.array([acov(k)/r0 for k in range(mmax + KN + 1)])
    mhat = next((m for m in range(1, mmax + 1) if np.all(np.abs(rho[m+1:m+KN+1]) < c)), mmax)
    M = min(2*mhat, mmax)
    lam = lambda t: 1.0 if abs(t) <= 0.5 else (2*(1 - abs(t)) if abs(t) <= 1 else 0.0)
    ks = np.arange(-M, M + 1); R = np.array([acov(abs(k)) for k in ks]); w = np.array([lam(k/M) for k in ks])
    G = (w*np.abs(ks)*R).sum(); g0 = (w*R).sum()
    b_cb = (2*G**2/((4/3)*g0**2))**(1/3)*n**(1/3); b_sb = (2*G**2/(2*g0**2))**(1/3)*n**(1/3)
    return dict(b_circular=float(b_cb), b_stationary=float(b_sb), mhat=int(mhat), n=n)
# daily linearised series of each ratio statistic (influence function of a ratio of sums)
D = pd.DataFrame({'E': E, 'Eu': Eu, 'L': L, 'd': d, 'pr': Eu*d/L, 'hp': np.minimum(Eu, d)}, index=V.index).groupby(day).sum()
lin = {'heat_share': D.d - point['heat_share']/100*D.L, 'SHHR_pr': D.pr - point['SHHR_pr']/100*D.d,
       'SHHR_hp': D.hp - point['SHHR_hp']/100*D.d, 'HEOR': D.E - point['HEOR']/100*D.d}
mon = pd.Series(D.index.month, index=D.index)
PW = {}
for k, z in lin.items():
    PW[k] = dict(raw=pw_block(z.values), deseasonalised=pw_block((z - z.groupby(mon).transform('mean')).values))
print(pd.DataFrame({k: {'raw': v['raw']['b_circular'], 'deseason': v['deseasonalised']['b_circular']} for k, v in PW.items()}).T.round(1))

# ---------- resampling schemes (all resample whole days in blocks) ----------
rng = np.random.default_rng(17); NBOOT = 2000
pos_by_day = pd.Series(np.arange(len(V))).groupby(dnum).apply(np.array).to_dict()
days = np.array(sorted(pos_by_day)); ndays = days.max() + 1
def rows_from_days(dl): return np.concatenate([pos_by_day[x] for x in dl if x in pos_by_day])
def scheme_blocks(blen):                    # non-overlapping calendar blocks, unstratified
    ids = days // blen; groups = {b: days[ids == b] for b in np.unique(ids)}; keys = list(groups)
    return lambda: rows_from_days(np.concatenate([groups[b] for b in rng.choice(keys, len(keys))]))
def scheme_circular(blen):                  # circular block bootstrap of calendar days, length blen
    blen = max(1, int(round(blen))); nb = int(np.ceil(ndays/blen))
    return lambda: rows_from_days(np.concatenate([(s + np.arange(blen)) % ndays for s in rng.integers(0, ndays, nb)]))
def scheme_strat_season(blen):              # blocks of blen days resampled within the season of the block
    ids = days // blen; groups = {b: days[ids == b] for b in np.unique(ids)}
    seas = {b: SEASON[(T0 + pd.Timedelta(days=int(np.median(g)))).month] for b, g in groups.items()}
    strata = {s: [b for b in groups if seas[b] == s] for s in sorted(set(seas.values()))}   # sorted: set order varies between runs (hash randomisation) [rev]
    return lambda: rows_from_days(np.concatenate([groups[b] for s, bs in strata.items() for b in rng.choice(bs, len(bs))]))
def scheme_strat_month():                   # whole calendar months resampled among the same month of other years
    mkey = {x: (T0 + pd.Timedelta(days=int(x))) for x in days}
    groups = pd.Series(days).groupby([mkey[x].strftime('%Y-%m') for x in days]).apply(np.array).to_dict()
    moy = {}
    for k in groups: moy.setdefault(int(k[5:]), []).append(k)
    return lambda: rows_from_days(np.concatenate([groups[k] for m, ks in moy.items() for k in rng.choice(ks, len(ks))]))
b_raw = PW['heat_share']['raw']['b_circular']; b_des = PW['heat_share']['deseasonalised']['b_circular']
SCHEMES = {'28-day blocks (current)': scheme_blocks(28),
           'Calendar-month stratified (whole months)': scheme_strat_month(),
           'Season-stratified 28-day blocks': scheme_strat_season(28),
           f'Politis-White circular, raw (b = {b_raw:.0f} d)': scheme_circular(b_raw),
           f'Politis-White on deseasonalised series, season-stratified (b = {b_des:.0f} d)': scheme_strat_season(max(1, int(round(b_des))))}
OUT = {'point': point, 'politis_white': PW, 'schemes': {}}
for name, draw in SCHEMES.items():
    B = pd.DataFrame([ind(draw()) for _ in range(NBOOT)])
    OUT['schemes'][name] = {c: dict(lo=float(B[c].quantile(.025)), hi=float(B[c].quantile(.975)), sd=float(B[c].std())) for c in B}
# monthly peak relief: stratified by calendar month vs moving blocks of months
pk = pd.read_csv('peaks_monthly.csv'); x = pk.dP.values; mm = pd.PeriodIndex(pk.month, freq='M').month
bym = {m: x[mm == m] for m in np.unique(mm)}
OUT['dP'] = {'iid months': np.percentile([rng.choice(x, len(x)).mean() for _ in range(NBOOT)], [2.5, 97.5]).tolist(),
             'calendar-month stratified': np.percentile([np.concatenate([rng.choice(v, len(v)) for v in bym.values()]).mean() for _ in range(NBOOT)], [2.5, 97.5]).tolist()}
json.dump(OUT, open('results_a17.json', 'w'), indent=1, default=float)
T = pd.DataFrame({n: {c: f"{v['lo']:.2f}–{v['hi']:.2f}" for c, v in s.items()} for n, s in OUT['schemes'].items()}).T
pd.set_option('display.width', 250); print(T.to_string()); print('dP', {k: np.round(v, 0).tolist() for k, v in OUT['dP'].items()})
