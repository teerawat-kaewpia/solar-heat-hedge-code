"""Step 8: fine-grid (dk = 0.05) preservation multipliers for every warming level (reference model)
and their 20-member weekly-block-bootstrap ensemble spread at +2 degC; check of the d<=L cap."""
from pathlib import Path
import pandas as pd, numpy as np, json, warnings, importlib.util, sys
warnings.filterwarnings('ignore')
src = (Path(__file__).resolve().parent / 'a7_robustness.py').read_text(encoding='utf-8').split('# ---------------- 1. seasonal-control')[0]
exec(src)                                    # model classes, helpers, data (M, H, S not yet built)
from sklearn.ensemble import HistGradientBoostingRegressor
S = H.dropna(subset=FEAT + ['L','E','Tamb','G']).copy(); S['hol'] = S.hol.astype(int)
pk = pd.read_csv('peaks_monthly.csv'); S = S[S.month.astype(str).isin(set(pk.month))]
KS = np.round(np.arange(1.0, 6.001, 0.05), 3); Tc0 = (S.Tamb + S.G*(25/800)).values; GAM = -0.004
def stress(m, tn, dT):
    X = S.copy(); X[TFE] = X[TFE] + dT
    f0 = m.predict(S); fx = m.predict(X); L = np.clip(S.L.values + fx - f0, 0, None)
    d = dLh_of(m, X, tn, L); fac = (1 + GAM*(Tc0 + dT - 25))/(1 + GAM*(Tc0 - 25))
    on = S.on.values; mon = S.month.astype(str).values; res = []
    for k in KS:
        E = S.E.values*k*fac; imp = np.clip(L - E, 0, None); Eu = np.minimum(E, L)
        res.append((k, np.minimum(Eu, d).sum()/d.sum()*100, (Eu*d/np.maximum(L, 1e-9)).sum()/d.sum()*100,
                    pd.Series(imp[on]*2).groupby(mon[on]).max().mean(), L.sum(), d.sum(), (E - Eu).sum()/E.sum()*100))
    return pd.DataFrame(res, columns=['k','SHHR_hp','SHHR_pr','peak','L','d','curt'])
def kstar(base, df, col, decreasing=False):
    v = df[col].values; ks = df.k.values
    ok = np.where(v <= base)[0] if decreasing else np.where(v >= base)[0]
    if len(ok) == 0: return np.nan
    i = ok[0]
    if i == 0: return float(ks[0])
    xp, fp = ([v[i], v[i-1]], [ks[i], ks[i-1]]) if decreasing else ([v[i-1], v[i]], [ks[i-1], ks[i]])
    return float(np.interp(base, xp, fp))
OUT = {}
ref = FAM['GBM, monotone (reference)'](TN0).fit(M)
b0 = stress(ref, TN0, 0.0); base = b0.iloc[0]; rows = []
yrs = len(S)/(48*365.25)   # valid-interval years [rev A6]
for dT in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    df = stress(ref, TN0, dT); r1 = df.iloc[0]
    rows.append(dict(dT=dT, L_GWh=r1.L/1e6/yrs, d_GWh=r1.d/1e6/yrs, SHHR_hp=r1.SHHR_hp, SHHR_pr=r1.SHHR_pr, peak=r1.peak,
                     kS_hp=kstar(base.SHHR_hp, df, 'SHHR_hp'), kS_pr=kstar(base.SHHR_pr, df, 'SHHR_pr'),
                     kP=kstar(base.peak, df, 'peak', decreasing=True),
                     curt_at_kP=float(np.interp(kstar(base.peak, df, 'peak', decreasing=True), df.k, df.curt)) if not np.isnan(kstar(base.peak, df, 'peak', decreasing=True)) else np.nan))
T = pd.DataFrame(rows); T.to_csv('tab_kstar_fine.csv', index=False); print(T.round(3).to_string())
OUT['fine'] = T.to_dict('records')
# capped vs uncapped SHHR at baseline
V = M.dropna(subset=['E']).copy(); d = _dLh_raw(ref, V, TN0); dc = np.minimum(d, V.L.values); E = V.E.values; L = V.L.values   # diagnostic keeps the uncapped value [rev A1]
OUT['cap_check'] = dict(SHHR_pr_uncapped=(E*d/L).sum()/d.sum()*100, SHHR_pr_capped=(E*dc/L).sum()/dc.sum()*100,
                        SHHR_hp_uncapped=np.minimum(E, d).sum()/d.sum()*100, SHHR_hp_capped=np.minimum(E, dc).sum()/dc.sum()*100,
                        heat_share_capped=dc.sum()/L.sum()*100)
print(OUT['cap_check'])
# ensemble at +2 (weekly block bootstrap refits)
import joblib
ens = []; members = joblib.load('ens_models.joblib')   # the same 100 members as a4 [rev A7]
for bi, mod in enumerate(members):
    m = GBM(FEAT, [0,0,0,1,1,1,1,1,0]); m.mod = mod
    b = stress(m, TN0, 0.0).iloc[0]; df = stress(m, TN0, 2.0)
    ens.append(dict(kS_hp=kstar(b.SHHR_hp, df, 'SHHR_hp'), kS_pr=kstar(b.SHHR_pr, df, 'SHHR_pr'), kP=kstar(b.peak, df, 'peak', decreasing=True),
                    SHHR_hp2=df.iloc[0].SHHR_hp, peak2=df.iloc[0].peak))
    print(bi, ens[-1])
E2 = pd.DataFrame(ens); OUT['ensemble_2C'] = {c: dict(mean=float(E2[c].mean()), lo=float(E2[c].quantile(.025)), hi=float(E2[c].quantile(.975))) for c in E2}
OUT['ensemble_2C_convergence'] = {n: {c: [float(E2[c][:n].quantile(.025)), float(E2[c][:n].quantile(.975))] for c in E2} for n in (20, 50, len(E2))}   # [rev A7]
print(OUT['ensemble_2C'])
json.dump(OUT, open('results_a8.json', 'w'), indent=1, default=float); print('done')
