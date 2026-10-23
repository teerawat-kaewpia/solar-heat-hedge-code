"""Step 4 (revision): allocation sensitivity, model-ensemble + weekly block-bootstrap uncertainty,
hourly change-point check, matched heatwave design with paired tests, peak-shift tests,
grouped out-of-sample permutation importance, gamma sensitivity, uncertainty of k* multipliers."""
import pandas as pd, numpy as np, json, warnings
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
import statsmodels.api as sm
warnings.filterwarnings('ignore')
rng = np.random.default_rng(2026)
def blk28(idx):   # 28-day bootstrap blocks (weekly blocks understated seasonal autocorrelation) [rev A7]
    return pd.Index(((pd.DatetimeIndex(idx).normalize() - pd.Timestamp('2023-03-03')).days // 28).values)
R = json.load(open('results.json')); OUT = {}
H = pd.read_pickle('hosp2.pkl'); TN = R['TN']; KWP = R['KWP']; TAR = R['TAR']
FEAT = ['hh','dow','hol','Tamb','T3','T24','T72','G2','trend']; MONO = [0,0,0,1,1,1,1,1,0]
TFE = ['Tamb','T3','T24','T72']
M = H.dropna(subset=FEAT+['L']).copy(); M['hol'] = M.hol.astype(int)
def mk(seed=0, it=600):
    return HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=40,
                                         l2_regularization=1.0, monotonic_cst=MONO, categorical_features=[0,1,2], random_state=seed)
def neutral(X, tn):
    X = X.copy()
    for c in TFE: X[c] = np.minimum(X[c], tn)
    return X
pk = pd.read_csv('peaks_monthly.csv'); months = set(pk.month)

# ---------- 1. model ensemble (weekly block bootstrap refits) ----------
from bootdraws import sampler, month_weights, NDRAW   # shared season-stratified block-bootstrap draws [rev A7]
import joblib
smpM = sampler(M.index); NB = 100; ens_models = []   # 100 members, draw j for member j [rev A7]
ens_dLh, ens_scen = [], []
S = H.dropna(subset=FEAT+['L','E','Tamb','G']).copy(); S['hol'] = S.hol.astype(int)
S = S[S.month.astype(str).isin(months)]
Tc0 = S.Tamb + S.G*(25/800)
def scen_metrics(model, dT, gam=-0.004, ks=(1, 1.5, 2, 3, 4, 5)):
    f0 = model.predict(S[FEAT]); X = S[FEAT].copy(); X[TFE] += dT
    fx = model.predict(X); L = (S.L + fx - f0).clip(lower=0).values
    dL = np.minimum(np.clip(fx - model.predict(neutral(X, TN)), 0, None), L)   # cap at scenario L [rev A1]
    fac = ((1 + gam*(Tc0 + dT - 25))/(1 + gam*(Tc0 - 25))).values
    res = {}
    for k in ks:
        E = S.E.values*k*fac; imp = np.clip(L - E, 0, None)
        tmp = pd.DataFrame({'imp': imp, 'm': S.month.values, 'on': S.on.values})
        res[k] = dict(SHHR=np.minimum(np.minimum(E, L), dL).sum()/dL.sum()*100,
                      peak=(tmp[tmp.on].groupby('m').imp.max()*2).mean())
    return res
def kstar(base, res, key):
    ks = sorted(res); v = np.array([res[k][key] for k in ks])
    if key == 'peak': v = -v; base = -base
    return float(np.interp(base, v, ks, right=np.nan)) if v.max() >= base else np.nan
for b in range(NB):
    idx = smpM(b)
    m = mk(seed=b, it=600).fit(   # same iterations as the reference model [rev A7]
        M.iloc[idx][FEAT], M.iloc[idx].L)
    d = np.minimum(np.clip(m.predict(M[FEAT]) - m.predict(neutral(M[FEAT], TN)), 0, None), M.L.values)   # cap at L_t [rev A1]
    ens_dLh.append(pd.Series(d, index=M.index))
    r0 = scen_metrics(m, 0.0); r2 = scen_metrics(m, 2.0)
    ens_scen.append(dict(SHHR0=r0[1]['SHHR'], peak0=r0[1]['peak'], SHHR2=r2[1]['SHHR'], peak2=r2[1]['peak'],
                         kS=kstar(r0[1]['SHHR'], r2, 'SHHR'), kP=kstar(r0[1]['peak'], r2, 'peak')))
    print('ens', b, {k: round(v, 3) for k, v in ens_scen[-1].items()})
    ens_models.append(m)
joblib.dump(ens_models, 'ens_models.joblib', compress=3)   # reused by a8 so that all CIs come from one ensemble
ES = pd.DataFrame(ens_scen); OUT['ens_scen'] = ES.describe(percentiles=[.025, .5, .975]).to_dict()

# ---------- 2. indicators with allocation variants + weekly block bootstrap ----------
V = H.dropna(subset=['E','L','B','dLh']).copy()
def ind(V, d):
    E, L, X = V.E.values, V.L.values, V.X.values
    Eu = np.minimum(E, L)                                        # [rev A5: self-used PV, E_u = min(E, L)]
    hp = np.minimum(Eu, d).sum()/d.sum()*100                   # heat-priority (upper bound)
    pr = (Eu*d/L).sum()/d.sum()*100                              # proportional allocation
    bl = np.minimum(np.clip(Eu - (L - d), 0, None), d).sum()/d.sum()*100   # base-load priority (lower bound)
    return dict(heat_share=d.sum()/L.sum()*100, HEOR=E.sum()/d.sum()*100, SHHR_hp=hp, SHHR_pr=pr, SHHR_bl=bl,
                load_cover=E.sum()/L.sum()*100)
point = ind(V, V.dLh.values); OUT['ind_point'] = point
smpV = sampler(V.index)
ensV = [e.reindex(V.index).values for e in ens_dLh]
boot = []
conv = {}
for nb in (20, 50, NB):                       # convergence of the interval with ensemble size [rev A7]
    boot = []
    for b in range(NDRAW):
        idx = smpV(b); d = ensV[b % nb][idx]; boot.append(ind(V.iloc[idx], d))
    B = pd.DataFrame(boot); conv[nb] = {c: [float(B[c].quantile(.025)), float(B[c].quantile(.975))] for c in B}
OUT['ind_CI'] = conv[NB]; OUT['ind_CI_convergence'] = conv
OUT['ci_note'] = 'point estimates: reference model on the full record; intervals: all NDRAW = 2000 season-stratified block draws, member b % n of the 100-member ensemble'
print('point', {k: round(v, 2) for k, v in point.items()}); print('CI', {k: np.round(v, 2).tolist() for k, v in OUT['ind_CI'].items()})
# by year / by Tmax bin with both allocations
V['date'] = V.index.normalize()
D = pd.read_pickle('daily.pkl')
V = V.join(D[['Tmax','HW']], on='date')
def two(g): d = g.dLh.values; Eu = np.minimum(g.E, g.L).values; return pd.Series(dict(hp=np.minimum(Eu, d).sum()/d.sum()*100, pr=(Eu*d/g.L.values).sum()/d.sum()*100))   # [rev A5: self-used PV, E_u = min(E, L)]
OUT['year_alloc'] = V.groupby(V.index.year).apply(two).round(2).to_dict('index')
V['bin'] = pd.cut(V.Tmax, [0, 30, 32, 34, 36, 50], labels=['<30','30-32','32-34','34-36','>=36'], right=False)
binb = []
for b in range(NDRAW):
    idx = smpV(b); Vb = V.iloc[idx].copy(); Vb['dLh'] = ensV[b % NB][idx]
    binb.append(Vb.groupby('bin').apply(two))
BB = pd.concat(binb, keys=range(len(binb)))
pt = V.groupby('bin').apply(two)
lo = BB.groupby(level=1).quantile(.025); hi = BB.groupby(level=1).quantile(.975)
OUT['bin_alloc'] = {str(i): dict(hp=pt.loc[i,'hp'], hp_lo=lo.loc[i,'hp'], hp_hi=hi.loc[i,'hp'], pr=pt.loc[i,'pr'], pr_lo=lo.loc[i,'pr'], pr_hi=hi.loc[i,'pr']) for i in pt.index}
print(pd.DataFrame(OUT['bin_alloc']).T.round(1))

# ---------- 3. thermal hedge elasticity with HAC (Newey-West) SE ----------
Dd = D.dropna(subset=['dLh']).copy(); w = Dd[Dd['T'] > TN].sort_index()
el = {}
for y in ['E', 'dLh', 'L']:
    r = sm.OLS(w[y], sm.add_constant(w.Tmax)).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
    el[y] = (float(r.params['Tmax']), float(r.bse['Tmax']))
# bootstrap ratio
smpW = sampler(w.index); etas = []
for b in range(NDRAW):
    s = w.iloc[smpW(b)]
    etas.append(np.polyfit(s.Tmax, s.E, 1)[0]/np.polyfit(s.Tmax, s.dLh, 1)[0])
OUT['elastic'] = dict(bE=el['E'], bdLh=el['dLh'], bL=el['L'], eta=el['E'][0]/el['dLh'][0], eta_CI=np.percentile(etas, [2.5, 97.5]).tolist(), n=len(w))
print('elastic', OUT['elastic'])

# ---------- 4. hourly-resolved change-point check ----------
W = H[(~H.hol)].dropna(subset=['L','Tamb','T24']).copy(); W['h'] = W.index.hour   # T24 can be NaN after outages with time-based windows [rev A4]
def cp3(x, y):
    best = None
    for tb in np.arange(14, 30.01, 0.2):
        X = np.c_[np.ones(len(x)), np.clip(x - tb, 0, None)]; b, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((y - X@b)**2).sum()
        if best is None or sse < best[0]: best = (sse, tb, b[1])
    return best[1], best[2]
hcp = {int(h): cp3(g.T24.values if 'T24' in g else g.Tamb.values, g.L.values*2) for h, g in W.groupby('h')}
tbs = np.array([v[0] for v in hcp.values()])
OUT['hourly_cp'] = dict(Tb_median=float(np.median(tbs)), Tb_IQR=np.percentile(tbs, [25, 75]).tolist(), by_hour={h: [round(v[0],1), round(v[1],1)] for h, v in hcp.items()})
print('hourly cp (T24)', OUT['hourly_cp']['Tb_median'], OUT['hourly_cp']['Tb_IQR'])
# night share of heat-driven load + share when instantaneous T below Tn
hhv = V.index.hour + V.index.minute/60
OUT['night_share'] = float(V.loc[(hhv < 7) | (hhv >= 18.5), 'dLh'].sum()/V.dLh.sum()*100)
OUT['dLh_share_Tinst_below_Tn'] = float(V.loc[V.Tamb < TN, 'dLh'].sum()/V.dLh.sum()*100)
OUT['night_mean_T_hot_season'] = float(V[(V.index.month.isin([3,4,5])) & ((hhv < 7) | (hhv >= 18.5))].Tamb.mean())

# ---------- 5. matched heatwave design ----------
Dm = D.dropna(subset=['dLh']).copy(); Dm['cov_hp'] = V.assign(c=np.minimum(np.minimum(V.E, V.L), V.dLh)).groupby('date').c.sum()
Dm['cov_pr'] = V.assign(c=np.minimum(V.E, V.L)*V.dLh/V.L).groupby('date').c.sum()
pairs = []
for d0, r in Dm[Dm.HW].iterrows():
    c = Dm[(~Dm.HW) & (Dm.hol == r.hol) & (Dm.index.year == d0.year) & (abs((Dm.index - d0).days) <= 15)]
    if len(c) >= 2:
        pairs.append(dict(date=d0, **{f'{v}_hw': r[v] for v in ['Tmax','L','E','dLh','pkL','pkB','Hg','cov_hp','cov_pr']},
                          **{f'{v}_ref': c[v].mean() for v in ['Tmax','L','E','dLh','pkL','pkB','Hg','cov_hp','cov_pr']}, nref=len(c)))
P = pd.DataFrame(pairs); hw = {}
for v in ['Tmax','L','E','dLh','pkL','pkB','Hg']:
    dif = P[f'{v}_hw'] - P[f'{v}_ref']
    hw[v] = dict(hw=P[f'{v}_hw'].mean(), ref=P[f'{v}_ref'].mean(), diff=dif.mean(), pct=dif.mean()/P[f'{v}_ref'].mean()*100,
                 p=float(stats.wilcoxon(dif).pvalue))
for a in ['hp','pr']:
    s_hw = P[f'cov_{a}_hw'].sum()/P.dLh_hw.sum()*100; s_rf = P[f'cov_{a}_ref'].sum()/P.dLh_ref.sum()*100
    dd = P[f'cov_{a}_hw']/P.dLh_hw*100 - P[f'cov_{a}_ref']/P.dLh_ref*100
    hw[f'SHHR_{a}'] = dict(hw=s_hw, ref=s_rf, diff=s_hw - s_rf, p=float(stats.wilcoxon(dd).pvalue))
OUT['heatwave_matched'] = dict(n_pairs=len(P), mean_ref_days=float(P.nref.mean()), res=hw)
# event-level test: consecutive heatwave days share reference windows, so aggregate to events first [rev A8]
P = P.sort_values('date'); P['event'] = (P.date.diff().dt.days != 1).cumsum()
Pe = P.groupby('event').agg({c: 'sum' for c in P.columns if c.endswith('_hw') or c.endswith('_ref')})
Pe['ndays'] = P.groupby('event').size(); hwe = {}; rng_e = np.random.default_rng(7)
ev = Pe.index.values; BE = [rng_e.choice(ev, len(ev)) for _ in range(2000)]
for v in ['Tmax','L','E','dLh','pkL','pkB','Hg']:
    a, b = Pe[f'{v}_hw']/Pe.ndays, Pe[f'{v}_ref']/Pe.ndays; dif = a - b
    boot = [dif.loc[s].mean() for s in BE]
    hwe[v] = dict(hw=a.mean(), ref=b.mean(), diff=dif.mean(), pct=dif.mean()/b.mean()*100,
                  ci=np.percentile(boot, [2.5, 97.5]).tolist(), p=float(stats.wilcoxon(dif).pvalue))
for a_ in ['hp','pr']:
    a = Pe[f'cov_{a_}_hw']/Pe.dLh_hw*100
    b = Pe[f'cov_{a_}_ref']/Pe.dLh_ref*100
    dd = a - b
    boot = [dd.loc[s].mean() for s in BE]
    hwe[f'SHHR_{a_}'] = dict(hw=a.mean(), ref=b.mean(), diff=dd.mean(),
                              ci=np.percentile(boot, [2.5, 97.5]).tolist(), p=float(stats.wilcoxon(dd).pvalue))
OUT['heatwave_event_level'] = dict(n_events=len(Pe), n_days=int(Pe.ndays.sum()), res=hwe)
print('HW event level', len(Pe)); print(pd.DataFrame(hwe).T)
print('HW matched', len(P)); print(pd.DataFrame(hw).T.round(3))

# ---------- 6. peak-shift tests ----------
dh = pk.hB - pk.hL; nz = dh[dh != 0]
OUT['peak_tests'] = dict(n=len(pk), later=int((dh > 0).sum()), earlier=int((dh < 0).sum()), same=int((dh == 0).sum()),
                         sign_p=float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue),
                         wilcoxon_p=float(stats.wilcoxon(nz).pvalue), mean_shift_h=float(dh.mean()))
a15 = pk.hL >= 15; b15 = pk.hB >= 15; n01 = int(((~a15) & b15).sum()); n10 = int((a15 & ~b15).sum())
OUT['peak_tests'].update(mcnemar_b=n01, mcnemar_c=n10, mcnemar_p=float(stats.binomtest(n01, n01+n10, 0.5).pvalue))
dPs = []
uPk = np.arange(len(pk))
pkm = pk.month.astype(str).values
for b in range(NDRAW): wt = month_weights(b, pkm); dPs.append((wt*pk.dP.values).sum()/wt.sum())   # same draws, months weighted by days drawn [rev A7]
OUT['dP_CI'] = np.percentile(dPs, [2.5, 97.5]).tolist(); OUT['ECC_CI'] = (np.array(OUT['dP_CI'])/KWP*100).tolist()
print('peak tests', OUT['peak_tests'], OUT['dP_CI'])

# ---------- 7. bill saving CI (monthly bootstrap) + 15-min proxy note ----------
Vv = H.dropna(subset=['E','L','B']).copy(); Vv['ms'] = Vv.month.astype(str); Vv = Vv[Vv.ms.isin(months)]
def mbill(g, col):
    on = g[g.on][col].sum(); off = g[~g.on][col].sum()
    return on*(TAR['on']+TAR['ft']) + off*(TAR['off']+TAR['ft']), (g[g.on][col].max()*2)*TAR['dem']
mb = []
for m_, g in Vv.groupby('ms'):
    eL, dL = mbill(g, 'L'); eB, dB = mbill(g, 'B'); mb.append(dict(m=m_, se=eL-eB, sd=dL-dB))
MB = pd.DataFrame(mb); sav = []
for b in range(NDRAW):
    wt = month_weights(b, MB.m.values); sav.append(((wt*(MB.se + MB.sd)).sum()/wt.sum()*12/1e6, (wt*MB.sd).sum()/(wt*(MB.se + MB.sd)).sum()*100))
sav = np.array(sav)
OUT['bill_CI'] = dict(saving=[float(np.percentile(sav[:,0], 2.5)), float(np.percentile(sav[:,0], 97.5))],
                      dem_share=[float(np.percentile(sav[:,1], 2.5)), float(np.percentile(sav[:,1], 97.5))])
print('bill CI', OUT['bill_CI'])

# ---------- 8. out-of-sample grouped permutation importance (LOYO) ----------
groups = {'Time of day': ['hh'], 'Calendar (day type, weekday)': ['hol','dow'], 'Temperature (all lags)': TFE,
          'Irradiance': ['G2'], 'Trend': ['trend']}
imp = {g: [] for g in groups}; cvs = []
for y in sorted(M.index.year.unique()):
    tr, te = M[M.index.year != y], M[M.index.year == y]
    m = mk().fit(tr[FEAT], tr.L); te = te.sample(min(8000, len(te)), random_state=1)
    base = ((te.L - m.predict(te[FEAT]))**2).mean()
    for g, cols in groups.items():
        t2 = te.copy(); perm = rng.permutation(len(t2))
        for c in cols: t2[c] = t2[c].values[perm]
        imp[g].append(((t2.L - m.predict(t2[FEAT]))**2).mean()/base - 1)
OUT['grouped_importance'] = {g: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for g, v in imp.items()}
cv = pd.DataFrame(R['ml_cv']); full = cv[cv.year.isin([2024, 2025])]
OUT['cv_summary'] = dict(all=dict(R2=cv.R2.mean(), CV=cv.CVRMSE.mean(), NMBE=cv.NMBE.mean()),
                         complete_years=dict(R2=full.R2.mean(), CV=full.CVRMSE.mean(), NMBE=full.NMBE.mean()),
                         excl2023=dict(R2=cv[cv.year != 2023].R2.mean(), CV=cv[cv.year != 2023].CVRMSE.mean()))
print('grouped importance', OUT['grouped_importance'])

# ---------- 9. gamma sensitivity ----------
iv = pd.read_pickle('iv.pkl'); gs = {}
for g in [-0.0030, -0.0040, -0.0045]:
    tl = {}
    for z in [1, 2, 3]:
        Z = iv[(iv.z == z) & (iv.G > 0)].copy(); kwp = Z.kWp.iloc[0]
        PR0 = R['pv'][str(z)]['PR0']; Tc = Z.Tamb + Z.G*25/800
        tl[z] = float((kwp*Z.G*0.5/1000*PR0*(-g)*(Tc - 25).clip(lower=0)).sum()/Z.E.sum()*100)
    gs[str(g)] = tl
M0 = mk().fit(M[FEAT], M.L)
for g in [-0.0030, -0.0045]:
    r0 = scen_metrics(M0, 0.0, gam=g); r2 = scen_metrics(M0, 2.0, gam=g)
    gs[f'{g}_scen'] = dict(SHHR2=r2[1]['SHHR'], peak2=r2[1]['peak'], kS=kstar(r0[1]['SHHR'], r2, 'SHHR'), kP=kstar(r0[1]['peak'], r2, 'peak'))
OUT['gamma_sens'] = gs; print('gamma', gs)

# ---------- 10. daily screening check ----------
p90 = R['heatwave']['P90']
g = H.groupby('date').agg(n=('L','count'), Tmax=('Tamb','max'), L=('L','sum'))
g = g[g.n >= 46]; rmed = g.L.rolling('15D', center=True, min_periods=5).median()   # calendar window [rev A4]
rem = g[(g.L <= 0.5*rmed) | (g.L >= 1.8*rmed)]
OUT['screen'] = dict(days_removed=len(rem), hot_removed=int((rem.Tmax >= p90).sum()))
print('screen', OUT['screen'])
json.dump(OUT, open('results_rev.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
P.to_csv('hw_pairs.csv', index=False)
pd.DataFrame(ens_scen).to_csv('ensemble_scen.csv', index=False)
print('done')
