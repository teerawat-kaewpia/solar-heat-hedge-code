"""Vector PDF versions of the figures of a plotting script, without touching its PNG files.
usage: python pdfwrap.py a3_figs.py
Every savefig('*.png') of the wrapped script is redirected to '*.pdf' (vector, no dpi), and the PNG itself is not written,
so the PNGs stay exactly as produced by the plain run in run_all.sh. Other outputs of the wrapped script are rewritten by it
as in the plain run, run_all.sh checks that they come out byte-identical. [rev]"""
import sys, matplotlib
matplotlib.use('Agg')
import matplotlib.figure as mf
_orig = mf.Figure.savefig
def sv(self, fname, *a, **k):
    if isinstance(fname, str) and fname.endswith('.png'):
        k2 = dict(k); k2.pop('dpi', None); k2.pop('pil_kwargs', None)
        return _orig(self, fname[:-4] + '.pdf', *a, **k2)   # PDF only, the PNG is left untouched
    return _orig(self, fname, *a, **k)
mf.Figure.savefig = sv
script = sys.argv[1]; sys.argv = sys.argv[1:]
exec(compile(open(script, encoding='utf-8').read(), script, 'exec'), {'__name__': '__main__'})
