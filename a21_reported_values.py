"""Step 21 (revision): recompute descriptive values that the earlier manuscript quoted without a script, and store
each with its definition and denominator (results_a21.json)."""
import pandas as pd, numpy as np, json
H = pd.read_pickle('hosp2.pkl'); D = pd.read_pickle('daily.pkl')
OUT = {}
# 1. seasonal level of daily demand and PV: mean over valid days (>= 46 valid intervals, outage screen) by calendar month,
#    pooled across years; the reported range is the range of the monthly means within each group of months
mm = D.groupby(D.index.month).agg(L=('L', 'mean'), E=('E', 'mean'), n=('L', 'size'))
grp = {'Dec-Jan': [12, 1], 'Apr-Jun': [4, 5, 6]}
OUT['daily_load_by_month'] = {
    'definition': 'mean daily gross load (MWh/d) of valid days by calendar month, pooled over 2023-2026; range of the monthly means within the group',
    'monthly_mean_MWh_d': {int(m): round(float(r.L)/1e3, 3) for m, r in mm.iterrows()}, 'days_per_month': {int(m): int(r.n) for m, r in mm.iterrows()},
    'groups': {g: dict(lo=float(mm.loc[ms, 'L'].min()/1e3), hi=float(mm.loc[ms, 'L'].max()/1e3), days=int(mm.loc[ms, 'n'].sum())) for g, ms in grp.items()},
    'pv_monthly_mean_range_MWh_d': [float(mm.E.min()/1e3), float(mm.E.max()/1e3)], 'n_days': int(len(D))}
# 2. intervals in which the uncapped model-attributed increment exceeded gross load (and was therefore capped)
V = H.dropna(subset=['dLh', 'L', 'Lhat', 'Lneu']).copy()
raw = (V.Lhat - V.Lneu).clip(lower=0)
capped = raw > V.L
OUT['cap_share'] = {'definition': 'share of intervals with a valid attribution (the denominator used for all indicators) in which the uncapped increment exceeded L_t',
                    'n_capped': int(capped.sum()), 'n_valid': int(len(V)), 'pct': float(capped.mean()*100),
                    'energy_removed_pct_of_dLh': float((raw - V.dLh).clip(lower=0).sum()/raw.sum()*100)}
json.dump(OUT, open('results_a21.json', 'w'), indent=1)
print(json.dumps({k: (v['groups'] if k == 'daily_load_by_month' else v) for k, v in OUT.items()}, indent=1))
print('PV monthly-mean range', OUT['daily_load_by_month']['pv_monthly_mean_range_MWh_d'])
