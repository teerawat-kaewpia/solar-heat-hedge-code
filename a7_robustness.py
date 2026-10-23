"""Step 7 (Q1 assessment): attribution credibility, extrapolation, cross-model stress test,
baseline sensitivity of the headline multipliers, 2023 diagnostics, monthly billing table,
uncertainty decomposition."""
import pandas as pd, numpy as np, json, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
warnings.filterwarnings('ignore')
R = json.load(open('results.json')); TN0 = R['TN']; TAR = R['TAR']; KWP = R['KWP']
from bootdraws import sampler, month_weights, NDRAW   # shared season-stratified block-bootstrap draws [rev A7]
def blk28(idx):   # 28-day bootstrap blocks (weekly blocks understated seasonal autocorrelation) [rev A7]
    return pd.Index(((pd.DatetimeIndex(idx).normalize() - pd.Timestamp('2023-03-03')).days // 28).values)
H = pd.read_pickle('hosp2.pkl')
FEAT = ['hh','dow','hol','Tamb','T3','T24','T72','G2','trend']; TFE = ['Tamb','T3','T24','T72']
M = H.dropna(subset=FEAT + ['L']).copy(); M['hol'] = M.hol.astype(int)
H['doy'] = H.index.dayofyear
M['s1'] = np.sin(2*np.pi*M.index.dayofyear/365.25); M['c1'] = np.cos(2*np.pi*M.index.dayofyear/365.25)
OUT = {}

# ---------------- model families (as in a5) ----------------
class GBM:
    def __init__(s, feats, mono, cats=(0,1,2)): s.f, s.m, s.c = feats, mono, list(cats)
    def fit(s, D):
        s.mod = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=40,
                    l2_regularization=1.0, monotonic_cst=s.m, categorical_features=s.c, random_state=0).fit(D[s.f], D.L); return s
    def predict(s, D): return s.mod.predict(D[s.f])
class Linear:
    def __init__(s, tn): s.tn = tn
    def design(s, D):
        g = (D.hh.astype(int)*2 + D.hol.astype(int)).values; Xi = np.zeros((len(D), 96)); Xi[np.arange(len(D)), g] = 1
        hh = D.hh.astype(int).values; c24 = np.clip(D.T24.values - s.tn, 0, None); c0 = np.clip(D.Tamb.values - s.tn, 0, None)
        c72 = np.clip(D.T72.values - s.tn, 0, None)
        S24 = np.zeros((len(D), 48)); S24[np.arange(len(D)), hh] = c24; S0 = np.zeros((len(D), 48)); S0[np.arange(len(D)), hh] = c0
        return np.c_[Xi, S24, S0, c72, D.G2.values/1000]
    def fit(s, D): s.mod = LinearRegression(fit_intercept=False).fit(s.design(D), D.L); return s
    def predict(s, D): return s.mod.predict(s.design(D))
class Spline:
    def fit(s, D):
        s.oh = OneHotEncoder(handle_unknown='ignore').fit(np.c_[D.hh.astype(int)*2 + D.hol])
        s.sp = {c: SplineTransformer(n_knots=8, degree=3, extrapolation='linear').fit(D[[c]]) for c in TFE + ['G2']}
        s.mod = Ridge(alpha=1.0).fit(s.design(D), D.L); return s
    def design(s, D): return np.hstack([s.oh.transform(np.c_[D.hh.astype(int)*2 + D.hol]).toarray()] + [s.sp[c].transform(D[[c]]) for c in TFE + ['G2']])
    def predict(s, D): return s.mod.predict(s.design(D))
class HourlyCP:
    def fit(s, D):
        s.p = {}
        for h, g in D.groupby(D.hh.astype(int)):
            best = None
            for tb in np.arange(16, 26.01, 0.2):
                X = np.c_[np.ones(len(g)), g.hol, np.clip(g.T24 - tb, 0, None)]
                b, *_ = np.linalg.lstsq(X, g.L.values, rcond=None); sse = ((g.L.values - X@b)**2).sum()
                if best is None or sse < best[0]: best = (sse, tb, b)
            s.p[h] = best[1:]
        return s
    def parts(s, D):
        full = np.empty(len(D)); base = np.empty(len(D)); hh = D.hh.astype(int).values
        for h, (tb, b) in s.p.items():
            i = hh == h; base[i] = b[0] + b[1]*D.hol.values[i]; full[i] = base[i] + b[2]*np.clip(D.T24.values[i] - tb, 0, None)
        return full, base
    def predict(s, D): return s.parts(D)[0]
def neutral(X, tn):
    X = X.copy()
    for c in TFE: X[c] = np.minimum(X[c], tn)
    return X
def dLh_of(m, D, tn, L=None):
    return np.minimum(_dLh_raw(m, D, tn), D.L.values if L is None else L)   # cap at L_t [rev A1]
def _dLh_raw(m, D, tn):
    if isinstance(m, HourlyCP):
        f, b = m.parts(D); return np.clip(f - b, 0, None)
    return np.clip(m.predict(D) - m.predict(neutral(D, tn)), 0, None)
FAM = {
 'GBM, monotone (reference)': lambda tn: GBM(FEAT, [0,0,0,1,1,1,1,1,0]),
 'GBM, short lags only': lambda tn: GBM(['hh','dow','hol','Tamb','T24','G2','trend'], [0,0,0,1,1,1,0]),
 'GBM, unconstrained': lambda tn: GBM(FEAT, [0]*9),
 'Spline additive (GAM-like)': lambda tn: Spline(),
 'Linear hourly cooling-degree': lambda tn: Linear(tn),
 'Hourly change-point': lambda tn: HourlyCP(),
}

# ---------------- 1. seasonal-control attribution & day-type slope test ----------------
seas = GBM(FEAT + ['s1','c1'], [0,0,0,1,1,1,1,1,0,0,0]).fit(M)
V = M.dropna(subset=['E']).copy()
def ind(V, d):
    E, L = V.E.values, V.L.values; Eu = np.minimum(E, L); hp = np.minimum(Eu, d).sum()/d.sum()*100; pr = (Eu*d/L).sum()/d.sum()*100   # [rev A5: self-used PV, E_u = min(E, L)]
    bl = np.minimum(np.clip(Eu - (L - d), 0, None), d).sum()/d.sum()*100
    hhv = V.index.hour + V.index.minute/60
    return dict(heat_share=d.sum()/L.sum()*100, SHHR_pr=pr, SHHR_hp=hp, SHHR_bl=bl, SHHR_sh=(hp+bl)/2,
                HEOR=E.sum()/d.sum()*100, night=d[(hhv < 7) | (hhv >= 18.5)].sum()/d.sum()*100)
ref = FAM['GBM, monotone (reference)'](TN0).fit(M)
d_ref = dLh_of(ref, V, TN0); d_seas = dLh_of(seas, V, TN0)
OUT['seasonal_control'] = dict(reference=ind(V, d_ref), with_doy_harmonics=ind(V, d_seas))
OUT['dLh_diag'] = dict(share_dLh_gt_L=float((d_ref > V.L.values).mean()*100), n_gt_L=int((d_ref > V.L.values).sum()),
                       share_zero=float((d_ref == 0).mean()*100), note='ΔL>=0 by monotone construction before clipping')
D = pd.read_pickle('daily.pkl').dropna(subset=['L','T'])
def cp(Dx):
    best = None
    for tb in np.arange(16, 28.01, 0.1):
        X = np.c_[np.ones(len(Dx)), np.clip(Dx['T'] - tb, 0, None)]; b, *_ = np.linalg.lstsq(X, Dx.L.values, rcond=None)
        sse = ((Dx.L.values - X@b)**2).sum()
        if best is None or sse < best[0]: best = (sse, tb, b)
    return best[1], best[2][1], best[2][0]
rng = np.random.default_rng(7); dt = {}
for lab, g in [('working', D[~D.hol.astype(bool)]), ('non-working', D[D.hol.astype(bool)])]:
    tb, sl, b0 = cp(g); smpG = sampler(g.index); bs = []
    for j in range(NDRAW):
        bs.append(cp(g.iloc[smpG(j)])[1])
    dt[lab] = dict(Tb=tb, slope=sl, slope_CI=np.percentile(bs, [2.5, 97.5]).tolist(), base=b0, n=len(g))
OUT['daytype_cp'] = dt; print('daytype', dt); print('seasonal', OUT['seasonal_control'])

# ---------------- 2. fit families, response curves, extrapolation ----------------
fams = {k: f(TN0).fit(M) for k, f in FAM.items()}
tmax = {c: float(M[c].max()) for c in TFE}
S = H.dropna(subset=FEAT + ['L','E','Tamb','G']).copy(); S['hol'] = S.hol.astype(int)
pk = pd.read_csv('peaks_monthly.csv'); S = S[S.month.astype(str).isin(set(pk.month))]
ext = {}
for dT in [0.5, 1, 1.5, 2, 2.5, 3]:
    ext[dT] = {c: float(((S[c] + dT) > tmax[c]).mean()*100) for c in TFE}
    ext[dT]['any_Tamb_hot_hours'] = float(((S.Tamb + dT) > tmax['Tamb'])[(S.index.hour >= 12) & (S.index.hour < 18)].mean()*100)
OUT['extrapolation'] = dict(train_max=tmax, pct_beyond=ext); print('extrap', ext)
smp = M.sample(15000, random_state=3); curves = {}
for name in ['GBM, monotone (reference)', 'Linear hourly cooling-degree', 'Hourly change-point', 'Spline additive (GAM-like)']:
    m = fams[name]; cc = {}
    for grp, hrs in [('night 00-06', range(0, 6)), ('morning 06-12', range(6, 12)), ('afternoon 12-18', range(12, 18)), ('evening 18-24', range(18, 24))]:
        sub = smp[smp.index.hour.isin(list(hrs))]; vals = []
        for T in np.arange(14, 45.1, 1.0):
            X = sub.copy(); X[TFE] = T; vals.append(float(m.predict(X).mean()*2))
        cc[grp] = vals
    curves[name] = cc
OUT['curves'] = dict(T=list(np.arange(14, 45.1, 1.0)), data=curves)

# ---------------- 3. cross-model stress test, fine k grid ----------------
KS = np.round(np.arange(1.0, 6.001, 0.05), 3)
Tc0 = (S.Tamb + S.G*(25/800)).values; GAM = -0.004
def stress(m, tn, dT):
    X = S.copy(); X[TFE] = X[TFE] + dT
    f0 = m.predict(S); fx = m.predict(X); L = np.clip(S.L.values + fx - f0, 0, None)
    d = dLh_of(m, X, tn, L); fac = (1 + GAM*(Tc0 + dT - 25))/(1 + GAM*(Tc0 - 25))
    on = S.on.values; mon = S.month.astype(str).values; res = []
    for k in KS:
        E = S.E.values*k*fac; imp = np.clip(L - E, 0, None)
        pkm = pd.Series(imp[on]*2).groupby(mon[on]).max().mean()
        res.append((k, np.minimum(np.minimum(E, L), d).sum()/d.sum()*100, (np.minimum(E, L)*d/L).sum()/d.sum()*100, pkm))
    return pd.DataFrame(res, columns=['k','SHHR_hp','SHHR_pr','peak'])
def kstar(base, df, col, decreasing=False):
    v = df[col].values; ks = df.k.values
    ok = np.where(v <= base)[0] if decreasing else np.where(v >= base)[0]
    if len(ok) == 0: return np.nan
    i = ok[0]
    if i == 0: return float(ks[0])
    return float(np.interp(base, [v[i-1], v[i]][::-1] if decreasing else [v[i-1], v[i]], [ks[i-1], ks[i]][::-1] if decreasing else [ks[i-1], ks[i]]))
cm = {}
for name, m in fams.items():
    b = stress(m, TN0, 0.0).iloc[0]; row = dict(SHHR_hp0=b.SHHR_hp, SHHR_pr0=b.SHHR_pr, peak0=b.peak)
    for dT in [1.0, 2.0, 3.0]:
        df = stress(m, TN0, dT); r1 = df[df.k == 1.0].iloc[0]
        row[f'SHHR_hp_{dT}'] = r1.SHHR_hp; row[f'peak_{dT}'] = r1.peak
        row[f'kS_hp_{dT}'] = kstar(b.SHHR_hp, df, 'SHHR_hp'); row[f'kS_pr_{dT}'] = kstar(b.SHHR_pr, df, 'SHHR_pr')
        row[f'kP_{dT}'] = kstar(b.peak, df, 'peak', decreasing=True)
    cm[name] = row; print(name, {k: round(v, 3) for k, v in row.items()})
OUT['cross_model_stress'] = cm

# ---------------- 4. Tn sensitivity of headline quantities (reference model) ----------------
tns = {}
for tn in [round(TN0 + d, 1) for d in (-2, -1, 0, 1, 2)]:
    d = dLh_of(ref, V, tn); r = ind(V, d)
    b = stress(ref, tn, 0.0).iloc[0]; df = stress(ref, tn, 2.0)
    r.update(kS_hp_2=kstar(b.SHHR_hp, df, 'SHHR_hp'), kS_pr_2=kstar(b.SHHR_pr, df, 'SHHR_pr'), kP_2=kstar(b.peak, df, 'peak', decreasing=True))
    tns[tn] = r; print('Tn', tn, {k: round(v, 2) for k, v in r.items()})
OUT['tn_sensitivity'] = tns

# ---------------- 5. LOYO residual diagnostics (reference) ----------------
rows = []
for y in sorted(M.index.year.unique()):
    tr, te = M[M.index.year != y], M[M.index.year == y]
    m = FAM['GBM, monotone (reference)'](TN0).fit(tr); e = te.L.values - m.predict(te)
    df = pd.DataFrame({'e': e, 'L': te.L.values}, index=te.index)
    df['season'] = pd.cut(df.index.month, [0, 2, 5, 10, 12], labels=['cool dry (Nov–Feb)', 'hot (Mar–May)', 'rainy (Jun–Oct)', 'cool dry (Nov–Feb)'], ordered=False)
    df['hgrp'] = pd.cut(df.index.hour, [-1, 5, 11, 17, 23], labels=['00–06', '06–12', '12–18', '18–24'])
    for (s, h), g in df.groupby(['season', 'hgrp'], observed=True):
        rows.append(dict(year=int(y), season=s, hours=h, n=len(g), bias_pct=g.e.mean()/g.L.mean()*100, cv_pct=np.sqrt((g.e**2).mean())/g.L.mean()*100))
    if y == 2023:
        mm = df.groupby(df.index.month).apply(lambda g: pd.Series(dict(bias=g.e.mean()/g.L.mean()*100, cv=np.sqrt((g.e**2).mean())/g.L.mean()*100)))
        OUT['loyo_2023_by_month'] = mm.round(2).to_dict('index')
RD = pd.DataFrame(rows); RD.to_csv('tab_residuals.csv', index=False)
OUT['residual_by_year_season'] = RD.groupby(['year', 'season'], observed=True).apply(lambda g: pd.Series(dict(bias=np.average(g.bias_pct, weights=g.n), cv=np.average(g.cv_pct, weights=g.n)))).round(2).reset_index().to_dict('records')
print('2023 by month', OUT['loyo_2023_by_month'])

# ---------------- 6. monthly demand / billing table ----------------
Vv = H.dropna(subset=['E','L','B']).copy(); Vv['ms'] = Vv.month.astype(str); Vv = Vv[Vv.ms.isin(set(pk.month))]
mt = []
for m_, g in Vv.groupby('ms'):
    on = g[g.on]
    e_L = on.L.sum()*(TAR['on']+TAR['ft']) + g[~g.on].L.sum()*(TAR['off']+TAR['ft'])
    e_B = on.B.sum()*(TAR['on']+TAR['ft']) + g[~g.on].B.sum()*(TAR['off']+TAR['ft'])
    pL, pB = on.L.max()*2, on.B.max()*2
    mt.append(dict(month=m_, gross_peak_kW=pL, net_peak_kW=pB, peak_relief_kW=pL - pB, energy_saving_kTHB=(e_L - e_B)/1e3,
                   demand_saving_kTHB=(pL - pB)*TAR['dem']/1e3, PV_MWh=g.E.sum()/1e3, n_valid=len(g)))
MT = pd.DataFrame(mt); MT.round(1).to_csv('tab_monthly_billing.csv', index=False)
OUT['monthly_billing_summary'] = dict(n=len(MT), annualisation='energy: sum / valid-interval years; demand: sum x 12/n_months [rev A6]',
                                     energy_MTHB_yr=MT.energy_saving_kTHB.sum()/1e3/(MT.n_valid.sum()/(48*365.25)), demand_MTHB_yr=MT.demand_saving_kTHB.sum()/1e3*12/len(MT),
                                     relief_mean=MT.peak_relief_kW.mean(), relief_min=MT.peak_relief_kW.min(), relief_max=MT.peak_relief_kW.max())
print(OUT['monthly_billing_summary'])
json.dump(OUT, open('results_a7.json', 'w'), indent=1, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
print('done')
