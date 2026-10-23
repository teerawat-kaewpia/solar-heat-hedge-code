"""Step 2: core analysis — thermal load models, PV thermal model, solar-heat hedge metrics,
heat stratification, heatwave events, warming x PV-scale stress test, battery, economics."""
import pandas as pd, numpy as np, json, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
warnings.filterwarnings('ignore')
rng = np.random.default_rng(42)
R = {}
def blk28(idx):   # 28-day bootstrap blocks (weekly blocks understated seasonal autocorrelation) [rev A7]
    return pd.Index(((pd.DatetimeIndex(idx).normalize() - pd.Timestamp('2023-03-03')).days // 28).values)
H = pd.read_pickle('hosp.pkl'); iv = pd.read_pickle('iv.pkl')
KWP = H.kWp.iloc[0]
# ---------------- tariff / carbon assumptions (PEA TOU 4.2, 22-33 kV) ----------------
TAR = dict(on=4.1839, off=2.6037, dem=132.93, ft=0.3972, ef=0.4857)
def onpeak_mask(idx, hol):
    h = idx.hour + idx.minute/60
    return (~hol) & (h > 9) & (h <= 22)          # interval ending in (09:00, 22:00]
H['on'] = onpeak_mask(H.index, H.hol.values)
H['month'] = H.index.to_period('M')

# ======================= A. daily 3P change-point cooling model ======================
def daily(df):
    g = df.groupby('date')
    D = g.agg(E=('E','sum'), L=('L','sum'), B=('B','sum'), n=('L','count'), T=('Tamb','mean'),
              Tmax=('Tamb','max'), Hg=('G', lambda s: s.sum()*0.5/1000), hol=('hol','first'))
    D = D[(D.n >= 46) & D['T'].notna()].copy()
    rmed = D.L.rolling('15D', center=True, min_periods=5).median()   # calendar window, days may be missing [rev A4]
    return D[(D.L > 0.5*rmed) & (D.L < 1.8*rmed)].copy()      # remove residual meter-outage days
D = daily(H)
def fit3pc(D, y='L'):
    best = None
    for tb in np.arange(16, 28.01, 0.1):
        X = np.c_[np.ones(len(D)), D.hol.astype(float), np.clip(D['T']-tb, 0, None)]
        b, *_ = np.linalg.lstsq(X, D[y].values, rcond=None)
        sse = ((D[y].values - X@b)**2).sum()
        if best is None or sse < best[0]: best = (sse, tb, b)
    sse, tb, b = best
    yh = np.c_[np.ones(len(D)), D.hol.astype(float), np.clip(D['T']-tb,0,None)] @ b
    r2 = 1 - sse/((D[y]-D[y].mean())**2).sum()
    cv = np.sqrt(sse/(len(D)-4))/D[y].mean()*100
    nmbe = (D[y]-yh).sum()/((len(D)-4)*D[y].mean())*100
    return dict(Tb=tb, b0=b[0], bhol=b[1], slope=b[2], R2=r2, CVRMSE=cv, NMBE=nmbe, n=len(D))
cp = fit3pc(D)
# weekly block bootstrap
from bootdraws import sampler, month_weights, NDRAW   # shared season-stratified block-bootstrap draws [rev A7]
smpD = sampler(D.index); boots = []
for j in range(NDRAW):
    Db = D.iloc[smpD(j)]
    f = fit3pc(Db); boots.append((f['Tb'], f['slope']))
boots = np.array(boots)
cp['Tb_CI'] = list(np.percentile(boots[:,0], [2.5, 97.5])); cp['slope_CI'] = list(np.percentile(boots[:,1], [2.5, 97.5]))
R['cp_hospital'] = cp
# per zone
th = H[['hol']]
zcp = {}
for z in [1,2,3]:
    Z = iv[iv.z==z].copy(); Z['hol'] = th.reindex(Z.index).hol.fillna(False).astype(bool).values; Z['date'] = Z.index.normalize()
    Dz = daily(Z); f = fit3pc(Dz); f['Lmean'] = Dz.L.mean(); f['Emean'] = Dz.E.mean(); zcp[z] = f
R['cp_zone'] = zcp
print('3PC hospital', {k: (np.round(v,3) if not isinstance(v, list) else np.round(v,2).tolist()) for k,v in cp.items()})

# ======================= B. hourly monotone-constrained gradient boosting ==================
H['hh'] = H.index.hour*2 + H.index.minute//30
H['dow'] = H.index.dayofweek
assert (H.index.to_series().diff().dropna() == pd.Timedelta('30min')).all()   # regular grid required [rev A4]
H['T3'] = H.Tamb.rolling('3h', min_periods=1).mean()     # time-based windows [rev A4]
H['T24'] = H.Tamb.rolling('24h', min_periods=12).mean()
H['T72'] = H.Tamb.rolling('72h', min_periods=36).mean()
H['G2'] = H.G.rolling('2h', min_periods=1).mean()
H['trend'] = (H.index - H.index[0]).days
FEAT = ['hh','dow','hol','Tamb','T3','T24','T72','G2','trend']
MONO = [0, 0, 0, 1, 1, 1, 1, 1, 0]
M = H.dropna(subset=FEAT+['L']).copy(); M['hol'] = M.hol.astype(int)
def mk(): return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=48,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               monotonic_cst=MONO, categorical_features=[0,1,2], random_state=0)
# leave-one-year-out validation
yrs = M.index.year; cvres = []
for y in sorted(yrs.unique()):
    tr, te = M[yrs != y], M[yrs == y]
    m = mk().fit(tr[FEAT], tr.L); p = m.predict(te[FEAT]); e = te.L - p
    cvres.append(dict(year=int(y), n=len(te), R2=1-(e**2).sum()/((te.L-te.L.mean())**2).sum(),
                      CVRMSE=np.sqrt((e**2).mean())/te.L.mean()*100, NMBE=e.sum()/(len(te)*te.L.mean())*100))
R['ml_cv'] = cvres; print(pd.DataFrame(cvres).round(3))
model = mk().fit(M[FEAT], M.L)
# permutation importance (on 20k sample)
smp = M.sample(20000, random_state=1); base = ((smp.L - model.predict(smp[FEAT]))**2).mean(); imp = {}
for f in FEAT:
    s2 = smp.copy(); s2[f] = rng.permutation(s2[f].values)
    imp[f] = ((s2.L - model.predict(s2[FEAT]))**2).mean()/base - 1
R['perm_importance'] = imp
TN = cp['Tb']   # thermal-neutral temperature from daily change-point model
def neutral(X, tn):
    X = X.copy()
    for c in ['Tamb','T3','T24','T72']: X[c] = np.minimum(X[c], tn)
    return X
M['Lhat'] = model.predict(M[FEAT])
M['Lneu'] = model.predict(neutral(M[FEAT], TN))
M['dLh'] = np.minimum((M.Lhat - M.Lneu).clip(lower=0), M.L)   # heat-driven increment (kWh per 30 min), capped at L_t [rev A1]
H = H.join(M[['Lhat','Lneu','dLh']])
# temperature partial dependence (hospital, 30-min kW) for figure
pdT = []
for T in np.arange(12, 42.1, 1.0):
    X = smp[FEAT].copy(); X[['Tamb','T3','T24','T72']] = T
    pdT.append((T, model.predict(X).mean()*2))
R['pdT'] = pdT

# ======================= C. PV thermal performance model ==========================
pv = {}
for z in [1,2,3]:
    Z = iv[iv.z==z].copy(); kwp = Z.kWp.iloc[0]
    Z['Tc'] = Z.Tamb + Z.G*(45-20)/800
    Z['y'] = Z.E/(kwp*Z.G*0.5/1000)
    ok = (Z.G > 300) & Z.y.between(0.2, 1.1) & (Z.st != 'fault')
    hub = HuberRegressor().fit((Z.loc[ok,'Tc']-25).values.reshape(-1,1), Z.loc[ok,'y'])
    pr0 = hub.intercept_; gam = hub.coef_[0]/pr0
    # thermal loss energy share
    Zd = Z[Z.G > 0]
    ref = kwp*Zd.G*0.5/1000*pr0
    tl = (ref*0.0040*(Zd.Tc-25).clip(lower=0)).sum()      # nominal gamma = -0.40 %/degC
    pv[z] = dict(kWp=kwp, PR0=pr0, gamma_pct=gam*100, n=int(ok.sum()), thermal_loss_pct=tl/Zd.E.sum()*100,
                 Tc_mean_day=Z.loc[Z.G>300,'Tc'].mean(), PR_obs=Z.E.sum()/(kwp*Z.G.sum()*0.5/1000))
R['pv'] = pv; print(pd.DataFrame(pv).T.round(3))
# field-fitted gamma is confounded by soiling/haze/clipping -> nominal datasheet-class value adopted
GAM = -0.0040
R['gamma_weighted'] = GAM

# ======================= D. hedge metrics (full record + per year) ======================
V = H.dropna(subset=['E','L','B','dLh']).copy()
def hedge(V):
    L, E, B, X, dL = V.L, V.E, V.B, V.X, V.dLh
    cov = np.minimum(np.minimum(E, L), dL)   # [rev A5: self-used PV, E_u = min(E, L)]
    tail = V.index.hour >= 16
    return dict(n=len(V), L_GWh=L.sum()/1e6, E_GWh=E.sum()/1e6, dLh_GWh=dL.sum()/1e6,
                heat_share=dL.sum()/L.sum()*100, load_cover=E.sum()/L.sum()*100,
                supply_cover=(1-X.sum()/E.sum())*100, SHHR=cov.sum()/dL.sum()*100,
                HEOR=E.sum()/dL.sum()*100, gap_tail_share=(dL-cov)[tail].sum()/(dL-cov).sum()*100,
                gap_GWh=(dL-cov).sum()/1e6)
R['hedge_all'] = hedge(V)
R['hedge_year'] = {int(y): hedge(g) for y, g in V.groupby(V.index.year)}
# sensitivity of SHHR to neutral temperature
sens = {}
for dt in [-2, -1, 0, 1, 2]:
    ln = model.predict(neutral(M[FEAT], TN+dt)); d = (M.Lhat - ln).clip(lower=0)
    j = M.join(H[['E']], rsuffix='_').dropna(subset=['E'])
    dd = d.reindex(j.index)
    sens[dt] = dict(Tn=TN+dt, SHHR=np.minimum(j.E, dd).sum()/dd.sum()*100, heat_share=dd.sum()/j.L.sum()*100)
R['shhr_sens'] = sens
print('hedge', {k: round(v,3) for k,v in R['hedge_all'].items()})
# diurnal profiles by thermal regime
V['hhf'] = V.index.hour + V.index.minute/60   # time stamps are interval ends; no extra +0.5 h [rev A2]
Dd = daily(H.assign(dLh=H.dLh)); Dd['dLh'] = H.groupby('date').dLh.sum()
V = V.join(Dd[['Tmax']], on='date')
V['reg'] = pd.cut(V.Tmax, [0, 30, 34, 50], labels=['cool (Tmax<30)', 'warm (30-34)', 'hot (>=34)'])
prof = V[~V.hol].groupby(['reg','hhf'])[['L','E','B','dLh']].mean()*2
prof.to_csv('prof_regime.csv')

# ======================= E. peaks, demand charge, economics =========================
def peaks(V, lcol='L', ncol='B'):
    on = V[V.on]
    g = on.groupby('month')
    pk = pd.DataFrame({'Lpk': g[lcol].max()*2, 'Bpk': g[ncol].max()*2,
                       'tL': g[lcol].idxmax(), 'tB': g[ncol].idxmax()})
    pk['E_at_Lpk'] = [V.loc[t, 'E']*2 for t in pk.tL]
    pk['dP'] = pk.Lpk - pk.Bpk
    return pk
Vv = H.dropna(subset=['E','L','B'])
pk = peaks(Vv)
cnt = Vv.groupby('month').size(); pk = pk[cnt.reindex(pk.index) > 48*25]   # full months only
pk['hL'] = pk.tL.dt.hour + pk.tL.dt.minute/60; pk['hB'] = pk.tB.dt.hour + pk.tB.dt.minute/60
pk.to_csv('peaks_monthly.csv')
R['peaks'] = dict(n_months=len(pk), Lpk_mean=pk.Lpk.mean(), Bpk_mean=pk.Bpk.mean(), dP_mean=pk.dP.mean(),
                  dP_min=pk.dP.min(), dP_max=pk.dP.max(), ECC=pk.dP.mean()/KWP*100,
                  E_at_Lpk_cf=(pk.E_at_Lpk/KWP).mean()*100, hL_median=pk.hL.median(), hB_median=pk.hB.median(),
                  share_B_after15=(pk.hB >= 15).mean()*100, share_L_after15=(pk.hL >= 15).mean()*100)
print('peaks', {k: round(v,2) for k,v in R['peaks'].items()})
def bill(V, col, pkser=None):
    on = V[V.on][col].sum(); off = V[~V.on][col].sum()
    en = on*(TAR['on']+TAR['ft']) + off*(TAR['off']+TAR['ft'])
    dem = (V[V.on].groupby('month')[col].max()*2).sum()*TAR['dem']
    return en, dem
full = Vv[Vv.month.isin(pk.index)]
nmon = len(pk)
eL, dL_ = bill(full, 'L'); eB, dB_ = bill(full, 'B')
yrs_e = len(full)/(48*365.25)   # energy sums cover valid intervals only; demand is per month [rev A6]
R['econ'] = dict(months=nmon, valid_years=yrs_e, energy_saving_MTHB_yr=(eL-eB)/yrs_e/1e6, demand_saving_MTHB_yr=(dL_-dB_)/nmon*12/1e6,
                 bill_L_MTHB_yr=(eL/yrs_e+dL_/nmon*12)/1e6, bill_B_MTHB_yr=(eB/yrs_e+dB_/nmon*12)/1e6,
                 E_MWh_yr=full.E.sum()/yrs_e/1e3, CO2_t_yr=full.E.sum()/yrs_e*TAR['ef']/1e3,
                 onpeak_share_E=full[full.on].E.sum()/full.E.sum()*100)
R['econ']['demand_share_saving'] = R['econ']['demand_saving_MTHB_yr']/(R['econ']['energy_saving_MTHB_yr']+R['econ']['demand_saving_MTHB_yr'])*100
print('econ', {k: round(v,3) for k,v in R['econ'].items()})

# ======================= F. heat stratification & heatwaves ===========================
Dd = Dd.dropna(subset=['dLh'])
Dd['pkL'] = H.groupby('date').L.max()*2; Dd['pkB'] = H.groupby('date').B.max()*2
Dd['cov'] = V.assign(c=np.minimum(np.minimum(V.E, V.L), V.dLh)).groupby('date').c.sum()
bins = [0, 30, 32, 34, 36, 50]; lab = ['<30','30-32','32-34','34-36','>=36']
Dd['bin'] = pd.cut(Dd.Tmax, bins, labels=lab, right=False)
st = Dd.groupby('bin').agg(days=('L','size'), L=('L','mean'), E=('E','mean'), dLh=('dLh','mean'), cov=('cov','sum'),
                           dsum=('dLh','sum'), Hg=('Hg','mean'), pkL=('pkL','mean'), pkB=('pkB','mean'))
st['SHHR'] = st['cov']/st.dsum*100; st['pk_red'] = st.pkL - st.pkB
st['pk_red_pct'] = st.pk_red/st.pkL*100
st = st.drop(columns=['cov','dsum']); st.to_csv('tab_strat.csv'); R['strat'] = st.round(3).reset_index().to_dict('records')
print(st.round(1))
# daily elasticities (warm days above Tn)
w = Dd[Dd['T'] > TN]
bE = np.polyfit(w.Tmax, w.E, 1)[0]; bL = np.polyfit(w.Tmax, w.dLh, 1)[0]; bLt = np.polyfit(w.Tmax, w.L, 1)[0]
R['elastic'] = dict(bE=bE, bdLh=bL, bL=bLt, eta=bE/bL, n=len(w))
# heatwave: >=3 consecutive days Tmax >= P90 of record
p90 = Dd.Tmax.quantile(0.9); hw = (Dd.Tmax >= p90).astype(int)
grp = (hw.diff() != 0).cumsum(); runlen = hw.groupby(grp).transform('sum')
Dd['HW'] = (hw == 1) & (runlen >= 3)
months = Dd[Dd.HW].index.month.unique()
ref = Dd[(~Dd.HW) & Dd.index.month.isin(months)]
hwd = Dd[Dd.HW]
def summ(x): return dict(days=len(x), Tmax=x.Tmax.mean(), L=x.L.mean(), E=x.E.mean(), dLh=x.dLh.mean(),
                        SHHR=x['cov'].sum()/x.dLh.sum()*100, pkL=x.pkL.mean(), pkB=x.pkB.mean(), Hg=x.Hg.mean())
R['heatwave'] = dict(P90=p90, events=int(Dd[Dd.HW].pipe(lambda x: (grp[x.index].nunique()))), HW=summ(hwd), REF=summ(ref), months=list(map(int, months)))
print('heatwave', R['heatwave'])
Dd.to_pickle('daily.pkl')

# ======================= G. climate stress test: warming x PV scale ======================
S = H.dropna(subset=FEAT+['L','E','Tamb','G']).copy(); S['hol'] = S.hol.astype(int)
S = S[S.month.isin(pk.index)]
f0 = model.predict(S[FEAT])
Tc0 = S.Tamb + S.G*(25/800)
def scen(dT, k, cap=0.0, eta=0.95, pmax=None):
    X = S[FEAT].copy(); X[['Tamb','T3','T24','T72']] += dT
    L = (S.L + model.predict(X) - f0).clip(lower=0)                 # delta method keeps observed variability
    Xn = neutral(X, TN); dLh = np.minimum(np.clip(model.predict(X) - model.predict(Xn), 0, None), L.values)   # cap at scenario L [rev A1]
    fac = (1 + GAM*(Tc0 + dT - 25))/(1 + GAM*(Tc0 - 25))
    E = S.E*k*fac
    net = (L - E).values; Ev = E.values
    if cap > 0:
        pm = (pmax or cap/2)*0.5  # kWh per interval
        soc = 0.0; net2 = np.empty_like(net); hr = S.index.hour.values; on = S.on.values
        for i in range(len(net)):
            n = net[i]
            if n < 0:
                ch = min(-n, pm, (cap - soc)/eta); soc += ch*eta; n += ch
            elif on[i] and hr[i] >= 15 and soc > 0:
                dc = min(n, pm, soc*eta); soc -= dc/eta; n -= dc
            net2[i] = n
        net = net2
    imp = np.clip(net, 0, None); curt = np.clip(-net, 0, None)
    self_used = Ev.sum() - curt.sum()
    cov = np.minimum(np.minimum(Ev, L.values), dLh).sum()
    tmp = pd.DataFrame({'imp': imp, 'month': S.month.values, 'on': S.on.values}, index=S.index)
    en = (tmp[tmp.on].imp.sum()*(TAR['on']+TAR['ft']) + tmp[~tmp.on].imp.sum()*(TAR['off']+TAR['ft']))
    pkm = tmp[tmp.on].groupby('month').imp.max()*2
    yrs = len(S)/(48*365.25)   # valid-interval years for energy; demand annualised by months [rev A6]
    return dict(dT=dT, k=k, cap=cap, L_GWh=L.sum()/1e6/yrs, E_GWh=Ev.sum()/1e6/yrs, imp_GWh=imp.sum()/1e6/yrs,
                curt_pct=curt.sum()/max(Ev.sum(),1e-9)*100, SSR=self_used/L.sum()*100, SHHR=cov/dLh.sum()*100,
                dLh_GWh=dLh.sum()/1e6/yrs, peak_net=pkm.mean(), bill=(en/yrs + pkm.sum()*TAR['dem']/(nmon/12))/1e6,
                CO2=imp.sum()*TAR['ef']/1e3/yrs)
grid = [scen(dT, k) for dT in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] for k in [0, 1, 1.5, 2, 3, 4, 5]]
G = pd.DataFrame(grid); G.to_csv('stress_grid.csv', index=False); print(G[G.dT.isin([0, 2])].round(2).to_string())
bat = [scen(dT, k, cap) for dT in [0, 2.0] for k in [2, 3, 4] for cap in [0, 2500, 5000, 10000]]
Bt = pd.DataFrame(bat); Bt.to_csv('battery.csv', index=False); print(Bt.round(2).to_string())
R['TN'] = TN; R['KWP'] = KWP; R['TAR'] = TAR
R['period'] = [str(H.index.min()), str(H.index.max())]; R['n_days_valid'] = len(D)
json.dump(R, open('results.json','w'), indent=1, default=lambda o: o.item() if hasattr(o,'item') else str(o))
H.to_pickle('hosp2.pkl')
print('done')
