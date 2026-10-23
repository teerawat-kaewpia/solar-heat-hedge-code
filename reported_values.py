"""Definition of every number quoted in the manuscript and the Supplementary Material, computed from the result files of one run.
usage: python reported_values.py <results_dir> <out.json>   (reproduce.py runs it and compares with results/reported_values.json)"""
import json, sys, os, pandas as pd
D = sys.argv[1]; J = lambda f: json.load(open(os.path.join(D, f), encoding='utf-8'))
R, RV, X, A7, A8, A10, A13, A14, A15, VAL = (J('results.json'), J('results_rev.json'), J('results_extra.json'), J('results_a7.json'),
    J('results_a8.json'), J('results_a10.json'), J('results_a13.json'), J('results_a14.json'), J('results_a15.json'), J('results_val.json'))
V = {}
cp = R['cp_hospital']; V.update(Tb=cp['Tb'], Tb_lo=cp['Tb_CI'][0], Tb_hi=cp['Tb_CI'][1], slope=cp['slope'], slope_lo=cp['slope_CI'][0],
                                slope_hi=cp['slope_CI'][1], cp_R2=cp['R2'], cp_CV=cp['CVRMSE'], n_days=R['n_days_valid'])
h = R['hedge_all']; V.update(n_int=h['n'], L_GWh=h['L_GWh'], E_GWh=h['E_GWh'], dLh_GWh=h['dLh_GWh'], gap_GWh=h['gap_GWh'],
                             gap_tail=h['gap_tail_share'], SC=h['supply_cover'], SS=h['load_cover'])
ip, ic = RV['ind_point'], RV['ind_CI']
for k in ['heat_share', 'SHHR_pr', 'SHHR_hp', 'SHHR_bl', 'HEOR', 'load_cover']:
    V[k] = ip[k]; V[k + '_lo'], V[k + '_hi'] = ic[k]
V['SHHR_sh'] = A10['calibrated']['primary']['SHHR_sh']
V['enrich'] = V['SHHR_pr']/V['SS']; V['timing_pen'] = V['HEOR'] - V['SHHR_hp']
V['night'] = RV['night_share']; V['night_T'] = RV['night_mean_T_hot_season']
V.update(pkE=X['peak_hour_E'], pkD=X['peak_hour_dLh'], lag_xcorr=X['lag_best_h'])
el = RV['elastic']; V.update(bdLh=el['bdLh'][0], bdLh_se=el['bdLh'][1], bE=el['bE'][0], bE_se=el['bE'][1], eta=el['eta'],
                             eta_lo=el['eta_CI'][0], eta_hi=el['eta_CI'][1])
cvy = {r['year']: r for r in R['ml_cv']}; V.update(cv2023=cvy[2023]['CVRMSE'], cv_lo=min(cvy[y]['CVRMSE'] for y in (2024, 2025, 2026)),
                                                   cv_hi=max(cvy[y]['CVRMSE'] for y in (2024, 2025, 2026)))
fw = VAL['forward']; V.update(fw_CV=fw['CV'], fw_R2=fw['R2'], fw_NMBE=fw['NMBE'])
p = R['peaks']; V.update(gross_pk=p['Lpk_mean'], net_pk=p['Bpk_mean'], dP=p['dP_mean'], ECC=p['ECC'], after15_B=p['share_B_after15'],
                         after15_L=p['share_L_after15'], dP_lo=RV['dP_CI'][0], dP_hi=RV['dP_CI'][1])
pt = RV['peak_tests']; V.update(n_later=pt['later'], n_earlier=pt['earlier'], sign_p=pt['sign_p'], mcnemar_p=pt['mcnemar_p'])
e = R['econ']; V.update(save_E=e['energy_saving_MTHB_yr'], save_D=e['demand_saving_MTHB_yr'], bill_L=e['bill_L_MTHB_yr'], bill_B=e['bill_B_MTHB_yr'],
                        dem_share=e['demand_share_saving'], CO2=e['CO2_t_yr'], E_MWh_yr=e['E_MWh_yr'],
                        save_lo=RV['bill_CI']['saving'][0], save_hi=RV['bill_CI']['saving'][1])
V['save_tot'] = V['save_E'] + V['save_D']; V['save_Ft'] = A10['calibrated']['primary']['total_MTHB_billFt']
b = A10['bill']; V.update(cinv_raw=b['ratio_raw']['median'], cinv_raw_q25=b['ratio_raw']['q25'], cinv_raw_q75=b['ratio_raw']['q75'],
                          cinv_cal=b['ratio_cal']['median'], cinv_cal_q25=b['ratio_cal']['q25'], cinv_cal_q75=b['ratio_cal']['q75'],
                          pkratio_cal=b['peak_ratio_cal']['median'], n_inv=b['n'])
unc = A10.get('uncalibrated') or {}
fine = {r['dT']: r for r in A8['fine']}
for dT in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    r = fine[dT]; t = f'{dT:.1f}'
    V.update({f'T4_L_{t}': r['L_GWh'], f'T4_d_{t}': r['d_GWh'], f'T4_hp_{t}': r['SHHR_hp'], f'T4_pk_{t}': r['peak'],
              f'T4_kShp_{t}': r['kS_hp'], f'T4_kSpr_{t}': r['kS_pr'], f'T4_kP_{t}': r['kP']})
V['curt_kP2'] = fine[2.0]['curt_at_kP']
en = A8['ensemble_2C']; V.update(ens_kS_lo=en['kS_hp']['lo'], ens_kS_hi=en['kS_hp']['hi'], ens_kP_lo=en['kP']['lo'], ens_kP_hi=en['kP']['hi'])
V.update(load_per_deg=X['load_per_degC_GWh'], dLh_per_deg=X['dLh_per_degC_pct'], SHHR_per_deg=X['SHHR_per_degC'])
# model families
fam = VAL['spec']; V['fam_share_lo'] = min(s['heat_share'] for s in fam.values()); V['fam_share_hi'] = max(s['heat_share'] for s in fam.values())
V['fam_pr_lo'] = min(s['SHHR_pr'] for s in fam.values()); V['fam_pr_hi'] = max(s['SHHR_pr'] for s in fam.values())
V['fam_hp_lo'] = min(s['SHHR_hp'] for s in fam.values()); V['fam_hp_hi'] = max(s['SHHR_hp'] for s in fam.values())
xm = A7['cross_model_stress']
V['fam_kS2_lo'] = min(v['kS_hp_2.0'] for v in xm.values()); V['fam_kS2_hi'] = max(v['kS_hp_2.0'] for v in xm.values())
V['fam_kP2_lo'] = min(v['kP_2.0'] for v in xm.values()); V['fam_kP2_hi'] = max(v['kP_2.0'] for v in xm.values())
V['fam_kP3_lo'] = min(v['kP_3.0'] for v in xm.values() if v.get('kP_3.0') == v.get('kP_3.0')); V['fam_kP3_hi'] = max(v['kP_3.0'] for v in xm.values() if v.get('kP_3.0') == v.get('kP_3.0'))
sc = A7['seasonal_control']['with_doy_harmonics']; V.update(seas_share=sc['heat_share'], seas_pr=sc['SHHR_pr'], seas_hp=sc['SHHR_hp'])
tn = A7['tn_sensitivity']; V.update(tn_share_lo=min(v['heat_share'] for v in tn.values()), tn_share_hi=max(v['heat_share'] for v in tn.values()),
                                   tn_pr_lo=min(v['SHHR_pr'] for v in tn.values()), tn_pr_hi=max(v['SHHR_pr'] for v in tn.values()),
                                   tn_hp_lo=min(v['SHHR_hp'] for v in tn.values()), tn_hp_hi=max(v['SHHR_hp'] for v in tn.values()),
                                   tn_kS_lo=min(v['kS_hp_2'] for v in tn.values()), tn_kS_hi=max(v['kS_hp_2'] for v in tn.values()))
dt = A7['daytype_cp']; V.update(Tb_work=dt['working']['Tb'], Tb_nonwork=dt['non-working']['Tb'], slope_ratio=dt['non-working']['slope']/dt['working']['slope'])
ex = A7['extrapolation']; V.update(T24_max=ex['train_max']['T24'], T72_max=ex['train_max']['T72'], Tamb_max=ex['train_max']['Tamb'],
                                  beyond2_T24=ex['pct_beyond']['2']['T24'] if '2' in ex['pct_beyond'] else ex['pct_beyond']['2.0']['T24'],
                                  beyond2_T72=ex['pct_beyond']['2']['T72'] if '2' in ex['pct_beyond'] else ex['pct_beyond']['2.0']['T72'])
# like-for-like (a13)
for r in A13['like_for_like']:
    t = f"{r['dT']:.1f}"
    for k in ['k_SHHR_hp', 'k_SHHR_pr', 'k_gap_abs_hp', 'k_gap_abs_pr', 'k_import', 'k_peak', 'k_peak_ratio', 'floor', 'peak_k1', 'peak_k6']:
        V[f'{k}_{t}'] = r[k]
for c in A13['lag_counterfactual']:
    key = {'observed': 'obs', 'dLh 2.5 h earlier': 'lag25', 'no temperature-driven load': 'nodlh'}.get(c['case'])
    if key: V.update({f'lag_{key}_after15': c['share_net_after15'], f'lag_{key}_net': c['net_peak'], f'lag_{key}_later': c['n_later'], f'lag_{key}_earlier': c['n_earlier']})
s = A13['support']; V.update(sup_joint=s['all_four_jointly']['dLh_pct'], sup_Tamb=s['Tamb']['dLh_pct'], sup_T72=s['T72']['dLh_pct'],
                             sup_hot=s['by_season_dLh_out_pct']['hot-dry'])
sat = A13['saturation']; V['pd95'] = sat['pd_T_at_95pct_of_range']
# block length (a14)
for blk in ['7d', '28d', '56d']:
    V[f'blk_share_{blk}'] = A14[blk]['heat_share']; V[f'blk_pr_{blk}'] = A14[blk]['SHHR_pr']
# metering (a15)
al = {str(a['read_day']): a for a in A15['alignment']}
V.update(align_IQR_month=al['month end']['energy_ratio_IQR'], align_IQR_min_other=min(a['energy_ratio_IQR'] for k, a in al.items() if k != 'month end'),
         align_IQR_max_other=max(a['energy_ratio_IQR'] for k, a in al.items() if k != 'month end'),
         n_exceed=A15['n_exceed'], n_months_pk=A15['n_months'])
V['timing_pairs'] = int(open(os.path.join(D, 'qc_timing_pairs.txt')).read()) if os.path.exists(os.path.join(D, 'qc_timing_pairs.txt')) else None
if A15['exceed_months']: V['exceed_max'] = max(r['ratio'] for r in A15['exceed_months']); V['exceed_min'] = min(r['ratio'] for r in A15['exceed_months'])
# heatwave
hd = RV['heatwave_matched']['res']; he = RV['heatwave_event_level']
V.update(hw_days=RV['heatwave_matched']['n_pairs'], hw_events=he['n_events'], hw_dLh_pct=he['res']['dLh']['pct'], hw_E_pct=he['res']['E']['pct'],
         hw_hp_diff=he['res']['SHHR_hp']['diff'], hw_hp_lo=he['res']['SHHR_hp']['ci'][0], hw_hp_hi=he['res']['SHHR_hp']['ci'][1], hw_hp_p=he['res']['SHHR_hp']['p'],
         hw_day_hp_p=hd['SHHR_hp']['p'])
ip_ = pd.read_csv(os.path.join(D, 'tab_ipcc_envelope.csv')).dropna()
V.update(ipcc_kS_lo=ip_.kS.min(), ipcc_kS_hi=ip_.kS.max(), ipcc_kP_lo=ip_.kP.min(), ipcc_kP_hi=ip_.kP.max(), ipcc_dT_lo=ip_.dT_vs_2011_2020.min(), ipcc_dT_hi=ip_.dT_vs_2011_2020.max())
# uncorrected metering sensitivity
a12 = J('results_a12.json'); V.update(raw_kS2=a12['2.0']['kS_hp'], raw_kP2=a12['2.0']['kP'])
rp = A10['calibrated']['raw_portal']; V.update(raw_pr=rp['SHHR_pr'], raw_share=rp['heat_share'], raw_hp=rp['SHHR_hp'], raw_dP=rp['peak_relief'])
conv = lambda v: [float(x) for x in v] if isinstance(v, list) else (float(v) if v is not None and not isinstance(v, str) else v)
json.dump({k: conv(v) for k, v in V.items()}, open(sys.argv[2], 'w'), indent=1)
print(len(V), 'values written')
# ---- tables and extra keys for the revised manuscript ----
W = json.load(open(sys.argv[2]))
hy, ya = R['hedge_year'], RV['year_alloc']
W['T2'] = {y: dict(L=hy[y]['L_GWh'], E=hy[y]['E_GWh'], d=hy[y]['dLh_GWh'], share=hy[y]['heat_share'], SS=hy[y]['load_cover'],
                   SC=hy[y]['supply_cover'], HEOR=hy[y]['HEOR'], pr=ya[y]['pr'], hp=ya[y]['hp'], gap=hy[y]['gap_GWh'], tail=hy[y]['gap_tail_share'])
           for y in hy}
cvs = VAL['cv']; W['T3'] = {n: dict(CV=cvs[n]['CV_complete'], **{k: s[k] for k in ['heat_share','SHHR_pr','SHHR_sh','SHHR_hp','eta','night_share','peak_hour']})
                           for n, s in VAL['spec'].items()}
W['T3']['forward'] = dict(CV=VAL['forward']['CV'], **{k: VAL['forward']['ind_forward'][k] for k in ['heat_share','SHHR_pr','SHHR_sh','SHHR_hp','eta','night_share','peak_hour']})
ph = [s['peak_hour'] for n, s in VAL['spec'].items() if not n.startswith('Spline')]; W['fam_pkh_lo'], W['fam_pkh_hi'] = min(ph), max(ph)
W['fam_pkh_spline'] = [s['peak_hour'] for n, s in VAL['spec'].items() if n.startswith('Spline')][0]
W['fam_night_spline'] = [s['night_share'] for n, s in VAL['spec'].items() if n.startswith('Spline')][0]
ns = [s['night_share'] for n, s in VAL['spec'].items() if not n.startswith('Spline')]; W['fam_night_lo'], W['fam_night_hi'] = min(ns), max(ns)
se = A10['seasonal']; key = lambda s: [k for k in se if k.startswith(s)][0]
W.update(seas_cool_hp=se[key('Cool')]['SHHR_hp'], seas_hot_hp=se[key('Hot')]['SHHR_hp'], seas_wet_hp=se[key('Wet')]['SHHR_hp'],
         enrich_lo=min(se[k]['SHHR_pr']/se[k]['SS'] for k in se), enrich_hi=max(se[k]['SHHR_pr']/se[k]['SS'] for k in se),
         SC_lo=min(se[k]['SC'] for k in se), SC_hi=max(se[k]['SC'] for k in se))
gs = RV['gamma_sens']; W.update(gam_kS=[gs['-0.003_scen']['kS'], gs['-0.0045_scen']['kS']], gam_kP=[gs['-0.003_scen']['kP'], gs['-0.0045_scen']['kP']])
W['fam_enrich'] = [min(s['SHHR_pr'] for s in VAL['spec'].values())/W['SS'], max(s['SHHR_pr'] for s in VAL['spec'].values())/W['SS']]
W['lag25_s'] = [c for c in A13['lag_counterfactual'] if c['case'] == 'dLh 2.5 h earlier'][0]
fm = A13['families_2C']
W.update(fam_kimp_lo=min(v['k_import'] for v in fm.values()), fam_kimp_hi=max(v['k_import'] for v in fm.values()),
         fam_kratio_lo=min(v['k_peak_ratio'] for v in fm.values()), fam_kratio_hi=max(v['k_peak_ratio'] for v in fm.values()),
         fam_pk_over_imp_lo=min(v['k_peak']/v['k_import'] for v in fm.values()), fam_pk_over_imp_hi=max(v['k_peak']/v['k_import'] for v in fm.values()))
ipm = ip_[ip_.period.str.startswith('2041')]
W.update(ipcc_mid_dT=[ipm.dT_vs_2011_2020.min(), ipm.dT_vs_2011_2020.max()], ipcc_mid_kS=[ipm.kS.min(), ipm.kS.max()], ipcc_mid_kP=[ipm.kP.min(), ipm.kP.max()])
Hh = pd.read_pickle(os.path.join(D, 'hosp2.pkl')).dropna(subset=['E','dLh']); hh_ = Hh.index.hour + Hh.index.minute/60
prof = Hh.groupby(hh_).dLh.mean(); top = prof[prof >= 0.95*prof.max()].index   # broad maximum: within 5% of the peak
W.update(plat_lo=float(top.min()), plat_hi=float(top.max()), lag_lo=float(top.min()) - W['pkE'], lag_hi=float(top.max()) - W['pkE'])
bd = json.load(open(os.path.join(D, 'boot_design.json'))); W.update(b_days=bd['b_days'], n_blocks=bd['n_blocks'])
if os.path.exists(os.path.join(D, 'results_a17.json')):
    A17 = J('results_a17.json'); sch = A17['schemes']; pick = lambda p: [v for k, v in sch.items() if k.startswith(p)][0]['heat_share']
    W.update(cmp_28=[pick('28-day')['lo'], pick('28-day')['hi']], cmp_pwraw=[pick('Politis-White circular')['lo'], pick('Politis-White circular')['hi']],
             cmp_s28=[pick('Season-stratified')['lo'], pick('Season-stratified')['hi']], cmp_pwdes=[pick('Politis-White on')['lo'], pick('Politis-White on')['hi']],
             pw_raw_b=A17['politis_white']['heat_share']['raw']['b_circular'])
cv_ = RV.get('ind_CI_convergence', {}); W['conv_share'] = {k: v['heat_share'] for k, v in cv_.items()}
W['conv_kP'] = {k: v['kP'] for k, v in A8.get('ensemble_2C_convergence', {}).items()}
if os.path.exists(os.path.join(D, 'results_a18.json')):
    for r in J('results_a18.json')['rows']:
        W[f"bs_{r['b']}_share"] = r['ci']['heat_share']; W[f"bs_{r['b']}_pr"] = r['ci']['SHHR_pr']; W[f"bs_{r['b']}_dP"] = r['ci']['dP']
A21 = J('results_a21.json'); g = A21['daily_load_by_month']['groups']
W.update(dl_win=[g['Dec-Jan']['lo'], g['Dec-Jan']['hi']], dl_hot=[g['Apr-Jun']['lo'], g['Apr-Jun']['hi']], dl_win_days=g['Dec-Jan']['days'], dl_hot_days=g['Apr-Jun']['days'],
         pv_mon=A21['daily_load_by_month']['pv_monthly_mean_range_MWh_d'], cap_pct=A21['cap_share']['pct'], cap_n=A21['cap_share']['n_capped'])
if os.path.exists(os.path.join(D, 'results_a22.json')):
    A22 = J('results_a22.json'); ref22 = [v for k, v in A22.items() if 'reference' in k][0]
    W['a22_ref'] = {d: dict(k_import=r['k_import'], k_peak=r['k_peak']) for d, r in ref22.items()}
    W['a22_kP_vs_a8_maxdiff'] = max(abs(r['k_peak'] - W[f'T4_kP_{d}']) for d, r in ref22.items())
json.dump(W, open(sys.argv[2], 'w'), indent=1, default=float); print('tables added')
