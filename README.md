# Annual-energy metrics overstate rooftop PV’s ability to limit hospital peak demand under warming: code package

Analysis code for the article *"Annual-energy metrics overstate rooftop PV’s ability to limit hospital peak demand under warming"* (Saengsuwan and Kaewpia). It contains the full pipeline behind the tables, figures and reported numbers of the article and its Supplementary Material. Full reproduction requires authorized access to the restricted data described below.

## Data
The data are not distributed with this code. The following can be requested from the corresponding author (teerawat.kae@crru.ac.th), subject to the hospital's approval (see `DATA_ACCESS.md`):

- the 30-min zone-level portal export used as input by `run_all.py`,
- the corrected hospital-level 30-min series (`data/hospital_30min_series.csv`, described in `DATA_DICTIONARY.md`),
- the de-identified monthly invoice summary (`data/invoice_summary_deidentified.csv`, read by `pea_invoices.py`),
- the result files (`results/`) with which `reproduce.py` redraws every figure and recomputes every reported number in about a minute.

## Run
```text
pip install -r requirements.txt
python run_all.py path/to/CRH_merged_30min_zone_level.csv --output-dir path/to/results   # full pipeline, about 45 min, needs the portal export and data/
python reproduce.py                                                                        # check of figures and reported numbers, needs results/
```
`run_all.py` writes all generated files outside the code folder, and `--dry-run` prints the commands without running them. `run_all.sh` is an optional wrapper for macOS, Linux and Git Bash.

## Steps
| Script | Purpose |
|---|---|
| `a1_prep.py` | Interval energies from lifetime counters (time stamps are interval ends), QC, closure check, hospital aggregation on the full 30-min grid, CT-ratio correction (1.20), counter-timing check of grid import, invoice ratio |
| `bootdraws.py` | Shared bootstrap design: Politis–White block length of the deseasonalised daily load (24 days), season-stratified non-overlapping blocks, 2,000 draws, seed 2026 (`boot_draws.npz`, `boot_design.json`) |
| `a2_analysis.py` | 3P change-point model, monotone GBM (LOYO-CV), model-attributed temperature-driven load (capped at L), PV thermal model, hedge indicators, peaks, economics (energy annualised over valid intervals), stress test, battery screening |
| `a4_revision.py` | 100-member model ensemble (member j refitted on draw j, saved as `ens_models.joblib`), allocation rules with self-used PV, intervals from the 2,000 draws, convergence at 20/50/100 members, heatwave design with event-level test, peak-shift tests |
| `a3_figs.py` | Derived metrics and main figures |
| `a5_validation.py` | Six model families and forward holdout |
| `a6_scenarios.py` | IPCC AR6 warming increments |
| `a7_robustness.py` | Seasonal control, day-type test, extrapolation, cross-model stress test, Tn sensitivity, residuals, monthly billing table |
| `a8_finegrid.py` | Fine-grid preservation multipliers and their ensemble intervals (same 100 members as a4) |
| `a9_robustfig.py`, `a11_surfaces_fig.py` | Robustness figure and six-family surfaces |
| `a10_v8.py` | Invoice reconciliation, uncorrected-data sensitivity, seasonal and annual indicators |
| `a12_calib_kstar.py` | Multipliers with uncorrected portal data |
| `a13_framing.py` | Like-for-like multipliers (relative and absolute targets, six families), PV-free on-peak floor, lag counterfactual, support of the neutral counterfactual |
| `a14_blocklength.py` | Unstratified block-length table (supplementary illustration) |
| `a15_metering.py` | Billed vs portal peak, billing-period alignment, counter-timing artefacts |
| `a16_fig5.py` | Fig. 5 |
| `a17_bootstrap_compare.py` | Five bootstrap designs, reference attribution held fixed (Table S2a) |
| `a18_block_sensitivity.py` | Block-length sensitivity under the primary design (Table S2b) |
| `a19_synthesis_fig.py` | Synthesis figure (Fig. S5), numbers read from the result files |
| `a20_export_dataset.py` | Public dataset |
| `a21_reported_values.py` | Descriptive values quoted in the text, each with its definition and denominator |
| `a22_family_targets.py` | $k^*_{imp}$ and $k^*_{peak}$ for the six model families at every warming level |
| `a23_fig6.py` | Fig. 6 (energy versus peak multipliers across families) and Fig. S6 |
| `pdfwrap.py` | Vector PDF versions of the `a3_figs.py` figures (PDF only, PNGs left untouched, checked by `run_all.py`) |
| `reported_values.py` | Definition of every number quoted in the article and the Supplementary Material, computed from the result files |
| `results_io.py`, `reproduce.py` | Exact pickle-free storage of the intermediate tables, and the check of every figure and reported number from `results/` (available on request) |
| `a24_graphical_abstract.py` | Graphical abstract (`graphical_abstract.tif/.pdf/.png`, 13 cm wide) |

## Intervals
Point estimates come from the reference model fitted to the full record. Sampling uncertainty uses the 2,000 shared draws. Model-refitting uncertainty uses the 100-member ensemble, and the same draws and members serve every quantity. Preservation-multiplier intervals reflect model refitting only. The heatwave event-level test resamples the heatwave events.

## Key assumptions (top of `a2_analysis.py`)
PEA TOU tariff 4.2 (22–33 kV): 4.1839 / 2.6037 THB kWh⁻¹, 132.93 THB kW⁻¹, Ft 0.3972 THB kWh⁻¹ (monthly Ft from invoices as sensitivity). TGO emission factor 0.4857 tCO2 MWh⁻¹ (location-based). PV temperature coefficient −0.40% °C⁻¹ (−0.30 / −0.45 in `a4_revision.py`). Fixed random seeds.
