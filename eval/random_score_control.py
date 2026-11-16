"""
CMPE 258 - control: a random-score detector on every slice
===========================================================

The README lists this control with the expectation "EER near 50% within
sampling variation". A detector that outputs random numbers knows nothing, so
on any slice it must sit at chance. If some slice comes out far from 50%, the
slice construction, the EER code or the bootstrap is manufacturing a result,
and every real number computed on that slice is suspect.

It needs only the eval protocol. No score files, no GPU.

  PART 1  load the protocol and draw random scores (three independent seeds)
  PART 2  EER with speaker-bootstrap intervals on every slice of the main table
  PART 3  the same random scores at a fixed threshold (covers fixed_threshold.py)
  PART 4  verdict and save

The slice construction, EER and bootstrap below are copied from
both_detectors.py on purpose: the control has to exercise the same code that
produced the main table, not a cleaner rewrite of it.

What a pass looks like: every EER close to 50%, and about 95% of the intervals
containing 50. With about 60 intervals, a few misses are expected by chance;
a slice that misses on all three seeds, or sits several points from 50, is not.

Not covered: the calibration and risk-coverage code, which lives in Zahira's
branch and is not in main.
"""

import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------- config
SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'
PROTO_EVAL = f'{SHRIYA_DIR}/keys/ASVspoof5.eval.track_1.tsv'
OUT_TABLE = '/content/drive/MyDrive/258_random_score_control.csv'

PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
EXPECTED_ROWS = 680774

UNCODED = '-'
ATTACK = 'ATTACK_LABEL'
SEEDS = [258, 259, 260]      # three independent random detectors
BOOT_SEED = 258
B_BOOT = 200                 # same as the main table

FAMILIES = {
    'legacy unit selection': ['A19'],
    'zero-shot TTS':         ['A17', 'A21', 'A22', 'A28', 'A29'],
    'voice conversion':      ['A24', 'A25', 'A26'],
}
MALAFIDE = {'A18': 'A17', 'A30': 'A17', 'A20': None, 'A23': None}
MALACOPULA = ['A27', 'A31', 'A32']


# ------------------------------------------- copied from both_detectors.py
def _eer(tar, non):
    if tar.size == 0 or non.size == 0:
        return np.nan
    n = tar.size + non.size
    s = np.concatenate((tar, non))
    lab = np.concatenate((np.ones(tar.size), np.zeros(non.size)))
    o = np.argsort(s, kind='mergesort')
    lab = lab[o]
    ts = np.cumsum(lab)
    ns = non.size - (np.arange(1, n + 1) - ts)
    frr = np.concatenate((np.atleast_1d(0.0), ts / tar.size))
    far = np.concatenate((np.atleast_1d(1.0), ns / non.size))
    i = np.argmin(np.abs(frr - far))
    return 100 * float((frr[i] + far[i]) / 2)


def eer_pct(df):
    return _eer(df.loc[df.KEY == 'bonafide', 'score'].to_numpy(),
                df.loc[df.KEY == 'spoof', 'score'].to_numpy())


def bootstrap(df, b=B_BOOT, seed=BOOT_SEED):
    """95% CI by resampling speakers."""
    spk = df.SPEAKER_ID.to_numpy()
    sc = df['score'].to_numpy()
    isb = (df.KEY == 'bonafide').to_numpy()
    uniq, inv = np.unique(spk, return_inverse=True)
    tar_by = [sc[(inv == i) & isb] for i in range(len(uniq))]
    non_by = [sc[(inv == i) & ~isb] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        p = rng.integers(0, len(uniq), len(uniq))
        v = _eer(np.concatenate([tar_by[i] for i in p]),
                 np.concatenate([non_by[i] for i in p]))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def crossed(df):
    coded = df[df.CODEC != UNCODED]
    n = coded.CODEC.nunique()
    sizes = coded.CODEC_SEED.value_counts()
    return coded[coded.CODEC_SEED.isin(sizes[sizes == n].index)]


def slices(df):
    """Every slice of the main table, built exactly as both_detectors.decompose does."""
    unc = df[df.CODEC == UNCODED]
    bona = unc[unc.KEY == 'bonafide']
    cb = crossed(df)
    out = [('ALL attacks, uncoded', 'reference', unc)]
    for fam, atks in FAMILIES.items():
        sel = unc[(unc[ATTACK].isin(atks)) & (unc.KEY == 'spoof')]
        if len(sel):
            out.append((f'{fam} ({"+".join(atks)})', 'generator', pd.concat([bona, sel])))
    for atk, src in MALAFIDE.items():
        sel = unc[(unc[ATTACK] == atk) & (unc.KEY == 'spoof')]
        if len(sel):
            lab = f'Malafide {atk}' + (f' (from {src})' if src else ' (src in dev)')
            out.append((lab, 'adversarial', pd.concat([bona, sel])))
    sel = unc[(unc[ATTACK].isin(MALACOPULA)) & (unc.KEY == 'spoof')]
    if len(sel):
        out.append((f'Malacopula ({"+".join(MALACOPULA)})', 'adversarial', pd.concat([bona, sel])))
    src_sel = unc[(unc[ATTACK] == 'A17') & (unc.KEY == 'spoof')]
    if len(src_sel):
        out.append(('  source A17 alone', 'adv-paired', pd.concat([bona, src_sel])))
    for c in sorted(cb.CODEC.unique()):
        out.append((f'codec {c}', 'codec', cb[cb.CODEC == c]))
    return out


# --------------------------------------------------------------------- parts
def load():
    print('PART 1. PROTOCOL AND RANDOM SCORES')
    print('==================================')
    if not os.path.exists(PROTO_EVAL):
        print(f'  [FAIL] {PROTO_EVAL} not found. Remount Drive and check SHRIYA_DIR.')
        return None
    proto = pd.read_csv(PROTO_EVAL, sep=r'\s+', header=None, names=PROTO_COLS, dtype=str)
    ok = len(proto) == EXPECTED_ROWS
    print(f'    [{"PASS" if ok else "FAIL"}] row count  {len(proto):,}')
    if not ok:
        return None
    print(f'    random detectors: seeds {SEEDS}, scores ~ N(0, 1), independent of everything')
    return proto[['SPEAKER_ID', 'CODEC', 'CODEC_SEED', ATTACK, 'KEY']]


def run_eer(proto):
    print('\n\nPART 2. EER ON EVERY SLICE')
    print('==========================')
    print(f'  speaker bootstrap, B = {B_BOOT}. Takes a few minutes.\n')
    rows = []
    for seed in SEEDS:
        df = proto.assign(score=np.random.default_rng(seed).standard_normal(len(proto)))
        for name, kind, sub in slices(df):
            lo, hi = bootstrap(sub)
            rows.append({'seed': seed, 'slice': name, 'kind': kind,
                         'n_bonafide': int((sub.KEY == 'bonafide').sum()),
                         'n_spoof': int((sub.KEY == 'spoof').sum()),
                         'EER_%': round(eer_pct(sub), 2), 'lo': round(lo, 2), 'hi': round(hi, 2)})
        print(f'  seed {seed} done')
    tab = pd.DataFrame(rows)
    tab['contains_50'] = (tab.lo <= 50) & (tab.hi >= 50)

    print(f'\n  {"slice":<40}' + ''.join(f'{"seed " + str(s):>26}' for s in SEEDS))
    for sl in tab['slice'].unique():
        line = f'  {sl:<40}'
        for s in SEEDS:
            r = tab[(tab['slice'] == sl) & (tab.seed == s)].iloc[0]
            line += f'{r["EER_%"]:>8.2f} [{r["lo"]:5.2f},{r["hi"]:5.2f}]{" " if r["contains_50"] else "x"}'
        print(line)
    print('\n  x = interval does not contain 50')
    return tab


def run_fixed(proto):
    print('\n\nPART 3. THE SAME SCORES AT A FIXED THRESHOLD')
    print('============================================')
    print('  Threshold 0 (the median of the random scores). A random detector must')
    print('  reject about half of real speech and accept about half of the fakes,')
    print('  on every slice.\n')
    df = proto.assign(score=np.random.default_rng(SEEDS[0]).standard_normal(len(proto)))
    isb = df.KEY == 'bonafide'
    rows = []
    for c in sorted(df.CODEC.unique()):
        m = df.CODEC == c
        rows.append({'slice': 'uncoded' if c == UNCODED else f'codec {c}',
                     'FRR_%': round(100 * float((df.score[m & isb] < 0).mean()), 2),
                     'FAR_%': round(100 * float((df.score[m & ~isb] >= 0).mean()), 2)})
    unc = df.CODEC == UNCODED
    for a in sorted(df.loc[~isb, ATTACK].unique()):
        rows.append({'slice': f'attack {a}', 'FRR_%': np.nan,
                     'FAR_%': round(100 * float((df.score[unc & ~isb & (df[ATTACK] == a)] >= 0).mean()), 2)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    worst = float(np.nanmax(np.abs(t[['FRR_%', 'FAR_%']].to_numpy() - 50)))
    print(f'\n  largest distance from 50%: {worst:.2f} points')
    return t, worst


def main():
    proto = load()
    if proto is None:
        return None
    tab = run_eer(proto)
    _, worst_rate = run_fixed(proto)

    print('\n\nPART 4. VERDICT')
    print('===============')
    n, hit = len(tab), int(tab.contains_50.sum())
    worst = tab.loc[(tab['EER_%'] - 50).abs().idxmax()]
    by_slice = tab.groupby('slice').contains_50.sum()
    always_miss = list(by_slice[by_slice == 0].index)
    print(f'  intervals containing 50:   {hit} of {n}  ({100 * hit / n:.0f}%; about 95% expected)')
    print(f'  EER range:                 {tab["EER_%"].min():.2f} to {tab["EER_%"].max():.2f}')
    print(f'  furthest from 50:          {worst["EER_%"]:.2f} on "{worst["slice"].strip()}" (seed {worst["seed"]})')
    print(f'  slices missing on all {len(SEEDS)} seeds: {always_miss if always_miss else "none"}')
    print(f'  fixed threshold, worst rate: {worst_rate:.2f} points from 50')
    ok = (hit / n >= 0.85) and not always_miss and abs(worst['EER_%'] - 50) < 5 and worst_rate < 5
    print(f'\n  [{"PASS" if ok else "CHECK"}] '
          + ('random scores sit at chance on every slice.' if ok
             else 'something is off chance. Paste this output before using the main table.'))

    tab.to_csv(OUT_TABLE, index=False)
    print(f'\n  saved: {OUT_TABLE}  ({len(tab)} rows)')
    print('\n\nDONE. Paste the output back.')
    return tab


if __name__ == '__main__':
    random_control_table = main()
