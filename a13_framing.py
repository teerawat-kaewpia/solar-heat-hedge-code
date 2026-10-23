"""Step 13 (revision, items B1-B4): like-for-like preservation multipliers, PV-irreducible on-peak floor,
lag counterfactual for the peak shift, support of the neutral counterfactual and extrapolation of the
temperature features."""
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
src = (Path(__file__).resolve().parent / 'a8_finegrid.py').read_text(encoding='utf-8').split('OUT = {}')[0]
exec(src)                                    # a7 helpers, model families, S, KS, Tc0, GAM, kstar
OUT = {}
ref = FAM['GBM, monotone (reference)'](TN0).fit(M)
on = S.on.values.astype(bool); mon = S.month.astype(str).values
f0 = ref.predict(S)

# ---------- B1/B2. stress test with like-for-like metrics ----------
def stress_full(dT, ks=KS):
    X = S.copy(); X[TFE] = X[TFE] + dT
    L = np.clip(S.L.values + ref.predict(X) - f0, 0, None)
    d = dLh_of(ref, X, TN0, L); fac = (1 + GAM*(Tc0 + dT - 25))/(1 + GAM*(Tc0 - 25))
    gross_pk = pd.Series(L[on]*2).groupby(mon[on]).max()
    nopv = on & (S.E.values <= 0)                         # on-peak intervals without any PV output
    floor = pd.Series(L[nopv]*2).groupby(mon[nopv]).max().reindex(gross_pk.index)
    res = []
    for k in ks:
        E = S.E.values*k*fac; Eu = np.minimum(E, L); imp = L - Eu
        net_pk = pd.Series(imp[on]*2).groupby(mon[on]).max()
        res.append(dict(k=k, SHHR_hp=np.minimum(Eu, d).sum()/d.sum()*100, SHHR_pr=(Eu*d/np.maximum(L, 1e-9)).sum()/d.sum()*100,
                        gap_abs=(d - np.minimum(Eu, d)).sum(),             # uncovered temperature-driven energy (heat priority)
                        gap_abs_pr=(d - Eu*d/np.maximum(L, 1e-9)).sum(),   # uncovered temperature-driven energy (proportional)
                        imp=imp.sum(),                                     # grid import energy
                        peak=net_pk.mean(),                                # absolute net peak (as in the paper)
                        peak_ratio=(net_pk/gross_pk).mean(),               # net/gross peak ratio (relative peak)
                        floor=floor.mean()))
    return pd.DataFrame(res)
b0 = stress_full(0.0); base = b0.iloc[0]
rows = []; curves = []
for dT in [0.0, 1.0, 2.0, 3.0]:
    df = stress_full(dT); r1 = df.iloc[0]; curves.append(df.assign(dT=dT))
    rows.append(dict(dT=dT,
        k_SHHR_hp=kstar(base.SHHR_hp, df, 'SHHR_hp'), k_SHHR_pr=kstar(base.SHHR_pr, df, 'SHHR_pr'),
        k_gap_abs_hp=kstar(base.gap_abs, df, 'gap_abs', decreasing=True), k_gap_abs_pr=kstar(base.gap_abs_pr, df, 'gap_abs_pr', decreasing=True),
        k_import=kstar(base.imp, df, 'imp', decreasing=True),
        k_peak=kstar(base.peak, df, 'peak', decreasing=True), k_peak_ratio=kstar(base.peak_ratio, df, 'peak_ratio', decreasing=True),
        peak_k1=r1.peak, peak_k6=df.iloc[-1].peak, floor=r1.floor, present_peak=base.peak,
        floor_minus_present=r1.floor - base.peak))
T = pd.DataFrame(rows); T.round(3).to_csv('tab_like_for_like.csv', index=False)
pd.concat(curves).round(4).to_csv('tab_like_for_like_curves.csv', index=False)   # k-curves for the revised Fig. 5
OUT['like_for_like'] = T.to_dict('records'); print(T.round(2).to_string())

# ---------- B1 across the six model families (absolute and relative targets at +2 degC) ----------
def k_targets(m, dT=2.0):
    g0 = m.predict(S); X = S.copy(); X[TFE] = X[TFE] + dT
    out = {}
    for lab, L in [('base', S.L.values), ('warm', np.clip(S.L.values + m.predict(X) - g0, 0, None))]:
        fac = 1.0 if lab == 'base' else (1 + GAM*(Tc0 + dT - 25))/(1 + GAM*(Tc0 - 25))
        gross_pk = pd.Series(L[on]*2).groupby(mon[on]).max(); rows = []
        for k in (KS[:1] if lab == 'base' else KS):
            E = S.E.values*k*fac; imp = L - np.minimum(E, L); net_pk = pd.Series(imp[on]*2).groupby(mon[on]).max()
            rows.append(dict(k=k, imp=imp.sum(), peak=net_pk.mean(), peak_ratio=(net_pk/gross_pk).mean()))
        out[lab] = pd.DataFrame(rows)
    b = out['base'].iloc[0]; w = out['warm']
    return dict(k_import=kstar(b.imp, w, 'imp', decreasing=True), k_peak=kstar(b.peak, w, 'peak', decreasing=True),
                k_peak_ratio=kstar(b.peak_ratio, w, 'peak_ratio', decreasing=True))
fam = {}
for name, mkf in FAM.items():
    m = ref if name.startswith('GBM, monotone') else mkf(TN0).fit(M)
    fam[name] = k_targets(m)
OUT['families_2C'] = fam; print(pd.DataFrame(fam).T.round(2).to_string())

# ---------- B2. lag counterfactual: temperature-driven load moved 2.5 h earlier ----------
G = H.copy()                                             # regular 30-min grid (a1 reindex)
Vv = G.dropna(subset=['E','L','B','dLh']).copy()
months = set(pk.month)
def peak_stats(Lc, label):
    Z = G.assign(Lc=Lc).dropna(subset=['E','Lc']); Z = Z[Z.month.astype(str).isin(months)]
    Z['net'] = (Z.Lc - np.minimum(Z.E, Z.Lc)); o = Z[Z.on]
    g = o.groupby(o.month.astype(str))
    tL, tB = g.Lc.idxmax(), g.net.idxmax()
    hL = tL.map(lambda t: t.hour + t.minute/60); hB = tB.map(lambda t: t.hour + t.minute/60)
    return dict(case=label, gross_peak=float((g.Lc.max()*2).mean()), net_peak=float((g.net.max()*2).mean()),
                relief=float(((g.Lc.max() - g.net.max())*2).mean()), share_net_after15=float((hB >= 15).mean()*100),
                share_gross_after15=float((hL >= 15).mean()*100), n_later=int((hB > hL).sum()), n_earlier=int((hB < hL).sum()))
d = G.dLh
lag = []
lag.append(peak_stats(G.L, 'observed'))
for sh in [1, 3, 5]:                                    # 0.5, 1.5 and 2.5 h earlier
    moved = d.shift(-sh)
    Lc = (G.L - d + moved).where(d.notna() & moved.notna())
    lag.append(peak_stats(Lc, f'dLh {sh*0.5:.1f} h earlier'))
lag.append(peak_stats((G.L - d).where(d.notna()), 'no temperature-driven load'))
OUT['lag_counterfactual'] = lag; print(pd.DataFrame(lag).round(1).to_string())

# ---------- B3. support of the neutral counterfactual ----------
V = M.dropna(subset=['E']).copy(); dref = dLh_of(ref, V, TN0); V['d'] = dref
V['moy'] = V.index.month
cell = V.groupby(['moy','hh'])
tol = 1.0
mins = cell[TFE].min()
joint = V.assign(ok=(V[TFE] <= TN0 + tol).all(axis=1)).groupby(['moy','hh']).ok.any()
V = V.join(mins.add_suffix('_cellmin'), on=['moy','hh']).join(joint.rename('joint_ok'), on=['moy','hh'])
sup = {}
for c in TFE:
    out_c = V[f'{c}_cellmin'] > TN0 + tol
    sup[c] = dict(rows_pct=float(out_c.mean()*100), dLh_pct=float(V.d[out_c].sum()/V.d.sum()*100))
sup['all_four_jointly'] = dict(rows_pct=float((~V.joint_ok).mean()*100), dLh_pct=float(V.d[~V.joint_ok].sum()/V.d.sum()*100))
season = V.assign(out=~V.joint_ok).groupby(np.where(V.index.month.isin([3,4,5]), 'hot-dry', 'other'))
sup['by_season_dLh_out_pct'] = {k: float(g.d[g.out].sum()/g.d.sum()*100) for k, g in season}
sup['tolerance_C'] = tol; sup['Tn'] = TN0
OUT['support'] = sup; print(json.dumps(sup, indent=1))

# ---------- B4. extrapolation and saturation ----------
obs_max = {c: float(H[c].max()) for c in TFE}
pdT = pd.DataFrame(R['pdT'])
pdT.columns = ['T', 'kW'] if pdT.shape[1] == 2 else pdT.columns
top = pdT.kW.max(); t95 = float(pdT.loc[pdT.kW >= pdT.kW.min() + 0.95*(top - pdT.kW.min()), 'T'].min())
OUT['saturation'] = dict(observed_max=obs_max, pd_T_at_95pct_of_range=t95,
                         share_intervals_T72_above_pd95=float((H.T72 > t95).mean()*100),
                         share_intervals_Tamb_above_pd95=float((H.Tamb > t95).mean()*100))
print(OUT['saturation'])
json.dump(OUT, open('results_a13.json', 'w'), indent=1, default=float); print('done')
