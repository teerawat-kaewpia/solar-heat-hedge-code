"""Reviewer check: redraw every figure and recompute every reported number from the result files shipped in results/.
usage: python reproduce.py [--output-dir reproduced]        (about 1-2 min, needs only requirements.txt)

1. copies results/ to the output folder and rebuilds the two intermediate tables from their exact gzip CSV copies,
2. runs the figure scripts (Fig. 1-6, Fig. S1-S6, graphical abstract) and compares each image with results/reference_figures,
3. runs reported_values.py and compares every number with results/reported_values.json, the values quoted in the paper,
4. writes reproduce_report.md in the output folder and exits with status 1 if anything differs.
The full pipeline from the raw zone-level export (run_all.py) needs data that are available on request only."""
import argparse, json, math, os, shutil, subprocess, sys, hashlib
from pathlib import Path

CODE = Path(__file__).resolve().parent; RES = CODE / 'results'
FIG_SCRIPTS = ['a3_figs.py', 'a11_surfaces_fig.py', 'a16_fig5.py', 'a19_synthesis_fig.py', 'a23_fig6.py', 'a24_graphical_abstract.py']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def compare_values(a, b, path='', out=None):
    """Exact comparison of nested JSON values, floats to a relative tolerance of 1e-12."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: out.append((path + '/' + k, 'missing in ' + ('recomputed' if k not in a else 'reported')))
            else: compare_values(a[k], b[k], path + '/' + k, out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): compare_values(x, y, f'{path}[{i}]', out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not ((math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))): out.append((path, f'{a!r} != {b!r}'))
    elif a != b: out.append((path, f'{a!r} != {b!r}'))
    return out

def compare_image(new, ref):
    if sha(new) == sha(ref): return True, 'byte-identical'
    from PIL import Image, ImageChops
    a, b = Image.open(new).convert('RGB'), Image.open(ref).convert('RGB')
    if a.size != b.size: return False, f'size {a.size} vs reference {b.size}'
    d = ImageChops.difference(a, b).convert('L'); n = sum(1 for v in d.getdata() if v > 0)
    return n == 0, 'pixel-identical (file metadata differ)' if n == 0 else f'{n} of {a.size[0]*a.size[1]} pixels differ (max {max(d.getdata())}/255)'

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--output-dir', type=Path, default=Path('reproduced'))
    out = ap.parse_args().output_dir.resolve()
    if out.exists() and any(out.iterdir()): sys.exit(f'Output folder is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    for f in RES.iterdir():
        if f.is_file() and not f.name.endswith(('.csv.gz', '.schema.json')) and f.name not in ('reported_values.json', 'figure_map.json'): shutil.copy2(f, out)
    sys.path.insert(0, str(CODE)); import results_io; results_io.restore(RES, out)
    rows, fails = [], 0
    def row(ok, what, detail):
        nonlocal fails; fails += not ok; rows.append(f"| {'PASS' if ok else 'FAIL'} | {what} | {detail} |"); print(f"{'PASS' if ok else 'FAIL'}  {what}: {detail}", flush=True)
    for s in FIG_SCRIPTS:
        print('+ running', s, flush=True)
        subprocess.run([sys.executable, str(CODE / s)], cwd=out, check=True, stdout=subprocess.DEVNULL)
    # a3_figs.py also rewrites results_extra.json from the shipped inputs, which must come back unchanged
    a = json.load(open(out / 'results_extra.json', encoding='utf-8')); b = json.load(open(RES / 'results_extra.json', encoding='utf-8'))
    d = compare_values(a, b); row(not d, 'results_extra.json rebuilt by a3_figs.py', 'identical' if not d else f'{len(d)} differences, first {d[0]}')
    for label, base in json.load(open(RES / 'figure_map.json', encoding='utf-8')).items():
        ok, detail = compare_image(out / (base + '.png'), RES / 'reference_figures' / (base + '.png')); row(ok, f'{label} ({base}.png)', detail)
    subprocess.run([sys.executable, str(CODE / 'reported_values.py'), str(out), str(out / 'reported_values_recomputed.json')], cwd=out, check=True, stdout=subprocess.DEVNULL)
    rec = json.load(open(out / 'reported_values_recomputed.json', encoding='utf-8')); rep = json.load(open(RES / 'reported_values.json', encoding='utf-8'))
    d = compare_values(rec, rep)
    row(not d, 'numbers quoted in the paper (reported_values.json)', f'all {len(rep)} entries identical' if not d else f'{len(d)} differences, first {d[:3]}')
    report = ['# Reproduction check', '', f'Code: `{CODE}`  ', f'Output: `{out}`  ', f'Python {sys.version.split()[0]}', '',
              '| Result | Check | Detail |', '|---|---|---|', *rows, '', f'{len(rows) - fails} of {len(rows)} checks passed.']
    (out / 'reproduce_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'\n{len(rows) - fails} of {len(rows)} checks passed. Figures and report in {out}')
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    main()
