"""Step 10: (1) reconciliation of portal data with 39+ months of PEA invoices, bill-implied Ft,
(2) raw-portal sensitivity of the headline indicators, (3) SHHR vs conventional load-match
metrics by season and year, (4) six-family warming x PV-scale surfaces, (5) consistency table."""
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
src = (Path(__file__).resolve().parent / 'a7_robustness.py').read_text(encoding='utf-8').split('# ---------------- 1. seasonal-control')[0]
exec(src)
def ind(V, d):
    E, L = V.E.values, V.L.values; Eu = np.minimum(E, L); hp = np.minimum(Eu, d).sum()/d.sum()*100; pr = (Eu*d/L).sum()/d.sum()*100   # [rev A5: self-used PV, E_u = min(E, L)]
    bl = np.minimum(np.clip(Eu - (L - d), 0, None), d).sum()/d.sum()*100
    hhv = V.index.hour + V.index.minute/60
    return dict(heat_share=d.sum()/L.sum()*100, SHHR_pr=pr, SHHR_hp=hp, SHHR_bl=bl, SHHR_sh=(hp+bl)/2,
                HEOR=E.sum()/d.sum()*100, night=d[(hhv < 7) | (hhv >= 18.5)].sum()/d.sum()*100)
OUT = {}
# ---------------- 1. utility invoices (PEA main account, TOU 4.2.2, 22-33 kV) ----------------
from pea_invoices import INVOICES
INV = pd.DataFrame(INVOICES, columns=['month', 'kWh', 'kW', 'THB']).set_index('month')
c = float(open('c_ct.txt').read())
# itemised invoices (08/2026, 09/2026): on/off-peak kWh, billed kW, base amount -> tariff check
item = {'2026-08': (480890.95, 589072.25, 2759.07, 3912842.49), '2026-09': (525841.70, 554148.70, 2732.41, 4006437.56)}
OUT['tariff_check'] = {m: dict(invoice_base=b, recomputed=on*TAR['on'] + off*TAR['off'] + kw*TAR['dem'] + 312.24,
                               on_share_invoice=on/(on + off)*100) for m, (on, off, kw, b) in item.items()}
Hm = H.copy(); Hm['on'] = Hm.on.astype(bool); Hm['month'] = Hm.month.astype(str)
rows = []
for m, sub in Hm.groupby('month'):
    if m not in INV.index: continue
    cov = sub.B.notna().sum()/(pd.Period(m).days_in_month*48)
    if cov < 0.9: continue
    Bc = sub.B.sum()/cov; s_on = sub.B[sub.on].sum()/sub.B.sum()
    rows.append(dict(month=m, coverage_pct=cov*100, B_raw_MWh=Bc*c/1e3, B_cal_MWh=Bc/1e3, inv_MWh=INV.kWh[m]/1e3,
                     ratio_raw=Bc*c/INV.kWh[m], ratio_cal=Bc/INV.kWh[m], on_share=s_on*100,
                     net30_cal=(sub.B[sub.on]*2).max(), net30_raw=(sub.B[sub.on]*2).max()*c, billed15=INV.kW[m]))
REC = pd.DataFrame(rows); REC['peak_ratio_cal'] = REC.net30_cal/REC.billed15; REC['peak_ratio_raw'] = REC.net30_raw/REC.billed15
# bill-implied fuel-adjustment charge (THB/kWh): invoice total / 1.07 - base (portal on-peak share applied to billed kWh)
REC['Ft_implied'] = [(INV.THB[m]/1.07 - (INV.kWh[m]*(r.on_share/100*TAR['on'] + (1 - r.on_share/100)*TAR['off']) + INV.kW[m]*TAR['dem'] + 312.24))/INV.kWh[m]
                     for m, r in REC.set_index('month').iterrows()]
REC.to_csv('tab_bill_reconciliation.csv', index=False); print(REC.round(3).to_string())
q = lambda x: dict(median=float(x.median()), q25=float(x.quantile(.25)), q75=float(x.quantile(.75)), min=float(x.min()), max=float(x.max()), sd=float(x.std()))
OUT['bill'] = dict(c=c, n=len(REC), ratio_raw=q(REC.ratio_raw), ratio_cal=q(REC.ratio_cal), peak_ratio_cal=q(REC.peak_ratio_cal),
                   peak_ratio_raw=q(REC.peak_ratio_raw), share_raw_peak_above_billed=float((REC.peak_ratio_raw > 1).mean()*100),
                   share_cal_peak_above_billed=float((REC.peak_ratio_cal > 1).mean()*100), period=[REC.month.min(), REC.month.max()])
print(OUT['bill'], OUT['tariff_check'])

# ---------------- 2. headline indicators: invoice-calibrated (primary) vs raw portal (sensitivity) ----------------
FT = REC.set_index('month').Ft_implied
def headline(Mx, label):
    ref = FAM['GBM, monotone (reference)'](TN0).fit(Mx)
    V = Mx.dropna(subset=['E']).copy(); d = np.minimum(dLh_of(ref, V, TN0), V.L.values)
    o = ind(V, d)
    pkm = pd.read_csv('peaks_monthly.csv'); Vm = V[V.month.astype(str).isin(set(pkm.month))].copy(); Vm['month'] = Vm.month.astype(str)
    on = Vm.on.astype(bool)
    gp = (Vm.L[on]*2).groupby(Vm.month[on]).max(); np_ = (Vm.B2[on]*2).groupby(Vm.month[on]).max()
    yrs = len(Vm)/(48*365.25); yrs_m = Vm.month.nunique()/12   # energy: valid-interval years; demand: months [rev A6]
    Eself = np.minimum(Vm.E, Vm.L)
    en = ((Eself[on]).sum()*(TAR['on'] + TAR['ft']) + (Eself[~on]).sum()*(TAR['off'] + TAR['ft']))/yrs/1e6
    ft = Vm.month.map(FT).fillna(TAR['ft']).values
    en_ft = ((Eself*np.where(on, TAR['on'], TAR['off'])).sum() + (Eself*ft).sum())/yrs/1e6
    dem = (gp - np_).clip(lower=0).sum()*TAR['dem']/yrs_m/1e6
    o.update(peak_relief=float((gp - np_).mean()), gross_peak=float(gp.mean()), net_peak=float(np_.mean()),
             energy_MTHB=en, demand_MTHB=dem, total_MTHB=en + dem, demand_share=dem/(en + dem)*100,
             energy_MTHB_billFt=en_ft, total_MTHB_billFt=en_ft + dem, Ft_weighted=float((Eself*ft).sum()/Eself.sum()),
             SC=float(Eself.sum()/Vm.E.sum()*100), SS=float(Eself.sum()/Vm.L.sum()*100), L_GWh_yr=float(Vm.L.sum()/yrs/1e6))
    print(label, {k: round(v, 3) for k, v in o.items()}); return o, ref
M0 = M.copy(); M0['B2'] = M0.B
base, ref0 = headline(M0, 'calibrated (primary)')
Mr = M.copy(); Mr['B2'] = Mr.B*c; Mr['L'] = (Mr.E.fillna(0) + Mr.B*c - Mr.X.fillna(0)*c).clip(lower=0); Mr = Mr.dropna(subset=['L'])
raw, _ = headline(Mr, 'raw portal (sensitivity)')
OUT['calibrated'] = dict(primary=base, raw_portal=raw, c=c)

# ---------------- 3. SHHR vs conventional metrics by season and year ----------------
V = M.dropna(subset=['E']).copy(); d = np.minimum(dLh_of(ref0, V, TN0), V.L.values); V['d'] = d
V['Eu'] = np.minimum(V.E, V.L)
mo = V.index.month
V['season'] = np.select([np.isin(mo, [3, 4, 5]), np.isin(mo, [6, 7, 8, 9, 10])], ['Hot-dry (Mar–May)', 'Wet (Jun–Oct)'], 'Cool-dry (Nov–Feb)')
V['year'] = V.index.year.astype(str)
def metrics(s):
    E, L, X, Eu, dd = s.E.values, s.L.values, s.X.values, s.Eu.values, s.d.values
    fl = np.mean(np.minimum(1, E/np.maximum(L, 1e-9)))            # load-cover (load-matching) index, 30-min
    hrs = s.index.hour + s.index.minute/60; day = (s.G > 5)
    return dict(SC=(1 - X.sum()/E.sum())*100, SS=Eu.sum()/L.sum()*100, LMI=fl*100,
                heat_share=dd.sum()/L.sum()*100, SHHR_pr=(Eu*dd/L).sum()/dd.sum()*100, SHHR_hp=np.minimum(Eu, dd).sum()/dd.sum()*100,
                night=dd[(hrs < 7) | (hrs >= 18.5)].sum()/dd.sum()*100)
seas = {k: metrics(s) for k, s in V.groupby('season')}; yr = {k: metrics(s) for k, s in V.groupby('year')}
allm = metrics(V)
pk = pd.read_csv('peaks_monthly.csv'); pk['m'] = pd.PeriodIndex(pk.month, freq='M').month; pk['y'] = pk.month.str[:4]
pk['season'] = np.select([pk.m.isin([3, 4, 5]), pk.m.isin([6, 7, 8, 9, 10])], ['Hot-dry (Mar–May)', 'Wet (Jun–Oct)'], 'Cool-dry (Nov–Feb)')
pk['rel'] = pk.dP/pk.Lpk*100; pk['after15'] = pk.hB >= 15
def pkm(s): return dict(n_months=len(s), relief_kW=s.dP.mean(), relief_pct=s.rel.mean(), net_after15=s.after15.mean()*100)
for k, s in pk.groupby('season'): seas[k].update(pkm(s))
for k, s in pk.groupby('y'):
    if k in yr: yr[k].update(pkm(s))
allm.update(pkm(pk))
OUT['seasonal'] = seas; OUT['yearly'] = yr; OUT['all'] = allm
T = pd.DataFrame({**{'All months': allm}, **seas, **yr}).T; T.to_csv('tab_metric_season_year.csv'); print(T.round(1).to_string())

# ---------------- 4. six-family warming x PV-scale surfaces ----------------
S = H.dropna(subset=FEAT + ['L', 'E', 'Tamb', 'G']).copy(); S['hol'] = S.hol.astype(int)
S = S[S.month.astype(str).isin(set(pk.month))]
KS = np.round(np.arange(1.0, 6.001, 0.1), 3); DTS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
Tc0 = (S.Tamb + S.G*(25/800)).values; GAM = -0.004
def surf(m):
    f0 = m.predict(S); res = []
    on = S.on.values.astype(bool); mon = S.month.astype(str).values
    for dT in DTS:
        X = S.copy(); X[TFE] = X[TFE] + dT; fx = m.predict(X); L = np.clip(S.L.values + fx - f0, 0, None)
        dd = dLh_of(m, X, TN0, L); fac = (1 + GAM*(Tc0 + dT - 25))/(1 + GAM*(Tc0 - 25))
        for k in KS:
            E = S.E.values*k*fac; Eu = np.minimum(E, L); imp = L - Eu
            res.append(dict(dT=dT, k=k, SHHR_hp=np.minimum(Eu, dd).sum()/dd.sum()*100,
                            peak=pd.Series(imp[on]*2).groupby(mon[on]).max().mean()))
    return pd.DataFrame(res)
SURF = {}
for name, f in FAM.items():
    m = f(TN0).fit(M); SURF[name] = surf(m); print('surface', name)
allS = pd.concat([v.assign(model=k) for k, v in SURF.items()]); allS.to_csv('tab_surfaces_6families.csv', index=False)
json.dump(OUT, open('results_a10.json', 'w'), indent=1, default=float); print('done')
