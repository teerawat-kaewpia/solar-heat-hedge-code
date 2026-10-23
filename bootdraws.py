"""Shared bootstrap design [rev A7]. One set of resampling draws, fixed seed, used by every script so that all
intervals come from the same resamples.
Scheme: season-stratified block bootstrap of calendar days. Non-overlapping blocks of B_LEN days from 3 March 2023;
each block takes the season of its middle day (cool-dry Nov-Feb, hot-dry Mar-May, wet Jun-Oct) and is resampled with
replacement among the blocks of the same season, so every draw keeps the observed seasonal composition. B_LEN is the
Politis-White (2004, corrected 2009) circular-block length of the deseasonalised daily gross load (attribution-free).
Run as a script once (after a1_prep.py) to create boot_draws.npz; import elsewhere."""
import numpy as np, pandas as pd, json, os
T0 = pd.Timestamp('2023-03-03'); NDRAW = 2000; SEED = 2026
SEASON = {11: 'cool', 12: 'cool', 1: 'cool', 2: 'cool', 3: 'hot', 4: 'hot', 5: 'hot', 6: 'wet', 7: 'wet', 8: 'wet', 9: 'wet', 10: 'wet'}

def pw_block(x):
    """Politis-White automatic block length (flat-top lag window, Patton-Politis-White 2009 constants)."""
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    def acov(k): return (x[:n-k]*x[k:]).sum()/n
    r0 = acov(0); KN = max(5, int(np.ceil(np.sqrt(np.log10(n))))); c = 2*np.sqrt(np.log10(n)/n)
    mmax = int(np.ceil(np.sqrt(n))) + KN
    rho = np.array([acov(k)/r0 for k in range(mmax + KN + 1)])
    mhat = next((m for m in range(1, mmax + 1) if np.all(np.abs(rho[m+1:m+KN+1]) < c)), mmax)
    M = min(2*mhat, mmax)
    lam = lambda t: 1.0 if abs(t) <= 0.5 else (2*(1 - abs(t)) if abs(t) <= 1 else 0.0)
    ks = np.arange(-M, M + 1); R = np.array([acov(abs(k)) for k in ks]); w = np.array([lam(k/M) for k in ks])
    G = (w*np.abs(ks)*R).sum(); g0 = (w*R).sum()
    return float((2*G**2/((4/3)*g0**2))**(1/3)*n**(1/3))

def build(hosp='hosp.pkl', out='boot_draws.npz'):
    H = pd.read_pickle(hosp); Dl = H.L.groupby(H.index.normalize()).agg(['sum', 'count']); Dl = Dl[Dl['count'] >= 46]['sum']
    b = int(round(pw_block((Dl - Dl.groupby(Dl.index.month).transform('mean')).values)))
    ndays = int((H.index.max().normalize() - T0).days) + 1
    block_of_day = np.arange(ndays) // b; nblk = block_of_day.max() + 1
    mid = {k: SEASON[(T0 + pd.Timedelta(days=int(np.median(np.where(block_of_day == k)[0])))).month] for k in range(nblk)}
    strata = {s: np.array([k for k in range(nblk) if mid[k] == s]) for s in sorted(set(mid.values()))}
    rng = np.random.default_rng(SEED)
    draws = np.stack([np.concatenate([rng.choice(ks, len(ks)) for s, ks in strata.items()]) for _ in range(NDRAW)])
    np.savez(out, block_of_day=block_of_day, draws=draws, b=b, seed=SEED)
    json.dump(dict(b_days=b, n_blocks=int(nblk), strata={s: int(len(k)) for s, k in strata.items()}, n_draws=NDRAW, seed=SEED),
              open('boot_design.json', 'w'), indent=1)
    print('bootstrap design: block', b, 'days,', nblk, 'blocks', {s: len(k) for s, k in strata.items()})

_Z = np.load('boot_draws.npz') if os.path.exists('boot_draws.npz') else None
def sampler(index):
    """Return f(j) -> row positions of `index` for draw j (rows repeat when a block is drawn more than once)."""
    blk = _Z['block_of_day'][((pd.DatetimeIndex(index).normalize() - T0).days).values]
    pos = pd.Series(np.arange(len(blk))).groupby(blk).apply(np.array).to_dict()
    empty = np.array([], dtype=int)
    return lambda j: np.concatenate([pos.get(k, empty) for k in _Z['draws'][j]])

if __name__ == '__main__':
    build()

def month_weights(j, month_labels):
    """Weight of each calendar month ('YYYY-MM') in draw j: number of its days drawn, relative to its length."""
    days = np.concatenate([np.where(_Z['block_of_day'] == k)[0] for k in _Z['draws'][j]])
    lab = pd.Series((T0 + pd.to_timedelta(days, unit='D')).strftime('%Y-%m')).value_counts()
    full = pd.Series((T0 + pd.to_timedelta(np.arange(len(_Z['block_of_day'])), unit='D')).strftime('%Y-%m')).value_counts()
    return np.array([lab.get(m, 0)/full.get(m, 1) for m in month_labels])
