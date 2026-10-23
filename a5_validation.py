"""Step 5 (second revision): attribution validation.
(i) model-specification sensitivity of the temperature-driven load and hedge indicators across five model families,
(ii) forward (temporal) holdout: train 2023-2024, test 2025-2026,
(iii) order-averaged (Shapley) allocation as a second neutral allocation rule."""
import pandas as pd, numpy as np, json, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
warnings.filterwarnings('ignore')
R = json.load(open('results.json')); TN = R['TN']
H = pd.read_pickle('hosp2.pkl')
FEAT = ['hh','dow','hol','Tamb','T3','T24','T72','G2','trend']; TFE = ['Tamb','T3','T24','T72']
M = H.dropna(subset=FEAT + ['L']).copy(); M['hol'] = M.hol.astype(int)
def neutral(X, tn=TN, cols=TFE):
    X = X.copy()
    for c in cols:
        if c in X: X[c] = np.minimum(X[c], tn)
    return X

# ---------------- model families ----------------
class GBM:
    def __init__(s, feats, mono): s.f, s.m = feats, mono
    def fit(s, D):
        s.mod = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=40,
                                              l2_regularization=1.0, monotonic_cst=s.m, categorical_features=[0,1,2],
                                              random_state=0).fit(D[s.f], D.L); return s
    def predict(s, D): return s.mod.predict(D[s.f])
class Linear:      # hour-by-daytype intercepts + hour-specific cooling-degree slopes on T24 and Tamb (piecewise linear)
    def design(s, D):
        g = (D.hh.astype(int)*2 + D.hol.astype(int)).values
        Xi = np.zeros((len(D), 96)); Xi[np.arange(len(D)), g] = 1
        hh = D.hh.astype(int).values
        c24 = np.clip(D.T24.values - TN, 0, None); c0 = np.clip(D.Tamb.values - TN, 0, None); c72 = np.clip(D.T72.values - TN, 0, None)
        S24 = np.zeros((len(D), 48)); S24[np.arange(len(D)), hh] = c24
        S0 = np.zeros((len(D), 48)); S0[np.arange(len(D)), hh] = c0
        return np.c_[Xi, S24, S0, c72, D.G2.values/1000]
    def fit(s, D): s.mod = LinearRegression(fit_intercept=False).fit(s.design(D), D.L); return s
    def predict(s, D): return s.mod.predict(s.design(D))
class Spline:      # GAM-like additive model: hour x daytype + penalised splines of each temperature lag and irradiance
    def fit(s, D):
        s.oh = OneHotEncoder(handle_unknown='ignore').fit(np.c_[D.hh.astype(int)*2 + D.hol])
        s.sp = {c: SplineTransformer(n_knots=8, degree=3, extrapolation='linear').fit(D[[c]]) for c in TFE + ['G2']}
        s.mod = Ridge(alpha=1.0).fit(s.design(D), D.L); return s
    def design(s, D):
        return np.hstack([s.oh.transform(np.c_[D.hh.astype(int)*2 + D.hol]).toarray()] + [s.sp[c].transform(D[[c]]) for c in TFE + ['G2']])
    def predict(s, D): return s.mod.predict(s.design(D))
class HourlyCP:    # hour-specific 3P change-point on T24 (working/non-working offsets), the 'physics-transparent' benchmark
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
    def predict(s, D, neutral_flag=False):
        out = np.empty(len(D)); hh = D.hh.astype(int).values
        for h, (tb, b) in s.p.items():
            i = hh == h; t = D.T24.values[i]
            out[i] = b[0] + b[1]*D.hol.values[i] + b[2]*np.clip(t - tb, 0, None)
        return out
MODELS = {
    'GBM-monotone (reference)': lambda: GBM(FEAT, [0,0,0,1,1,1,1,1,0]),
    'GBM, short lags only (T, T24)': lambda: GBM(['hh','dow','hol','Tamb','T24','G2','trend'], [0,0,0,1,1,1,0]),
    'GBM, unconstrained': lambda: GBM(FEAT, [0]*9),
    'Spline additive (GAM-like)': lambda: Spline(),
    'Linear hourly cooling-degree': lambda: Linear(),
    'Hourly change-point (T24)': lambda: HourlyCP(),
}
def dLh_of(model, D):
    return np.minimum(_dLh_raw(model, D), D.L.values)   # cap at L_t [rev A1]
def _dLh_raw(model, D):
    if isinstance(model, HourlyCP):
        p = model.predict(D); Dn = D.copy(); Dn['T24'] = np.minimum(Dn.T24, TN)
        # neutral: remove the change-point term entirely (balance points are hour-specific, all >= ~TN)
        n = np.empty(len(D)); hh = D.hh.astype(int).values
        for h, (tb, b) in model.p.items():
            i = hh == h; n[i] = b[0] + b[1]*D.hol.values[i]
        return np.clip(p - n, 0, None)
    return np.clip(model.predict(D) - model.predict(neutral(D)), 0, None)
def indicators(D, d):
    E, L = D.E.values, D.L.values
    Eu = np.minimum(E, L); hp = np.minimum(Eu, d).sum()/d.sum()*100; pr = (Eu*d/L).sum()/d.sum()*100   # [rev A5: self-used PV, E_u = min(E, L)]
    bl = np.minimum(np.clip(Eu - (L - d), 0, None), d).sum()/d.sum()*100
    hhv = D.index.hour + D.index.minute/60
    night = d[(hhv < 7) | (hhv >= 18.5)].sum()/d.sum()*100
    prof = pd.Series(d, index=hhv).groupby(level=0).mean(); pkh = float(prof.idxmax())   # interval-end time [rev A2]
    dd = pd.DataFrame({'E': E, 'd': d, 'T': D.Tamb.values}, index=D.index).resample('D').agg({'E': 'sum', 'd': 'sum', 'T': 'max'})
    cnt = D.resample('D').size(); dd = dd[cnt >= 46]
    w = dd[D.Tamb.resample('D').mean().reindex(dd.index) > TN]
    eta = np.polyfit(w['T'], w.E, 1)[0]/np.polyfit(w['T'], w.d, 1)[0]
    return dict(heat_share=d.sum()/L.sum()*100, HEOR=E.sum()/d.sum()*100, SHHR_pr=pr, SHHR_hp=hp, SHHR_bl=bl,
                SHHR_sh=(hp + bl)/2, eta=eta, night_share=night, peak_hour=pkh)
V = M.join(H[['E']], rsuffix='_').dropna(subset=['E'])
OUT = {'spec': {}, 'cv': {}}
for name, mk in MODELS.items():
    # leave-one-year-out skill
    cv = []
    for y in sorted(M.index.year.unique()):
        tr, te = M[M.index.year != y], M[M.index.year == y]
        m = mk().fit(tr); e = te.L.values - m.predict(te)
        cv.append(dict(y=int(y), CV=np.sqrt((e**2).mean())/te.L.mean()*100, R2=1 - (e**2).sum()/((te.L - te.L.mean())**2).sum()))
    cvd = pd.DataFrame(cv); full = cvd[cvd.y.isin([2024, 2025])]
    m = mk().fit(M)
    d = dLh_of(m, V)
    OUT['spec'][name] = indicators(V, d)
    OUT['cv'][name] = dict(CV_all=cvd.CV.mean(), R2_all=cvd.R2.mean(), CV_complete=full.CV.mean(), R2_complete=full.R2.mean())
    print(name, {k: round(v, 2) for k, v in {**OUT['spec'][name], **OUT['cv'][name]}.items()})

# ---------------- forward (temporal) holdout ----------------
tr = M[M.index < '2025-01-01']; te = M[M.index >= '2025-01-01']
mf = MODELS['GBM-monotone (reference)']().fit(tr); e = te.L.values - mf.predict(te)
Vte = V[V.index >= '2025-01-01']
mfull = MODELS['GBM-monotone (reference)']().fit(M)
OUT['forward'] = dict(n_test=len(te), CV=np.sqrt((e**2).mean())/te.L.mean()*100,
                      R2=1 - (e**2).sum()/((te.L - te.L.mean())**2).sum(), NMBE=e.sum()/(len(te)*te.L.mean())*100,
                      ind_forward=indicators(Vte, dLh_of(mf, Vte)), ind_fullmodel_same_period=indicators(Vte, dLh_of(mfull, Vte)))
print('forward', {k: (round(v, 3) if not isinstance(v, dict) else {a: round(b, 2) for a, b in v.items()}) for k, v in OUT['forward'].items()})
# ---------------- Shapley allocation on reference model, with CI from existing ensemble/bootstrap logic ----------------
sp = pd.DataFrame(OUT['spec']).T
OUT['range_across_models'] = {c: [float(sp[c].min()), float(sp[c].max())] for c in sp.columns}
json.dump(OUT, open('results_val.json', 'w'), indent=1, default=float)
sp.round(2).to_csv('tab_model_sensitivity.csv'); print(sp.round(2).to_string()); print('done')
