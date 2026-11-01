"""
CMPE 258 - dev-stage check on the XLS-R scores
==============================================

Run this before the eval scores land. Three jobs:

  1. VALIDATE   Zahira's score files, the same way we validated Shriya's.
                A format surprise is cheap to find now on a 6 MB dev file and
                expensive to find tonight on the eval set.
  2. SEEDS      Do the three seeds agree? Three seeds exist so this can be
                checked; if they disagree materially, every RQ2 claim built on
                the SSL column inherits that instability.
  3. HEAD-TO-HEAD  First architecture comparison. Both detectors scored the
                same 140,950 dev files (attacks A09-A16).

Self-contained: downloads the 19 MB protocol archive if the dev protocol is not
already on disk. No audio, no GPU.
"""

import glob
import os
import subprocess

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- paths
SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'
ZAHIRA_DIR = '/content/drive/MyDrive/XLS-R'        # adjust if the shortcut lands elsewhere
PROTO_DIR = '/content/asvspoof5_protocols'

AASIST_DEV = f'{SHRIYA_DIR}/scores/aasist_dev.tsv'
SEED_FILES = {258: f'{ZAHIRA_DIR}/scores/xlsr_dev_seed_258.tsv',
              259: f'{ZAHIRA_DIR}/scores/xlsr_dev_seed_259.tsv',
              260: f'{ZAHIRA_DIR}/scores/xlsr_dev_seed_260.tsv'}
COMBINED = f'{ZAHIRA_DIR}/scores/xlsr_dev.tsv'

PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
EXPECTED_DEV_ROWS = 140950
ZENODO = 'https://zenodo.org/records/14498691/files/ASVspoof5_protocols.tar.gz'


# ----------------------------------------------------------------------- EER
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


def eer_pct(df, col='score'):
    return _eer(df.loc[df.KEY == 'bonafide', col].to_numpy(),
                df.loc[df.KEY == 'spoof', col].to_numpy())


def auc_pct(df, col='score'):
    """P(bona fide scores above spoof). Below 0.5 = inverted."""
    from scipy.stats import rankdata
    pos = df.loc[df.KEY == 'bonafide', col].to_numpy()
    neg = df.loc[df.KEY == 'spoof', col].to_numpy()
    if not len(pos) or not len(neg):
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# -------------------------------------------------------------------- loading
def dev_protocol():
    hits = glob.glob(os.path.join(PROTO_DIR, '**', 'ASVspoof5.dev.track_1.tsv'),
                     recursive=True)
    if not hits:
        print('  dev protocol not found locally; downloading (19 MB)...')
        os.makedirs(PROTO_DIR, exist_ok=True)
        tgz = '/content/ASVspoof5_protocols.tar.gz'
        subprocess.run(['wget', '-q', '-O', tgz, ZENODO], check=True)
        subprocess.run(['tar', '-xf', tgz, '-C', PROTO_DIR], check=True)
        os.remove(tgz)
        hits = glob.glob(os.path.join(PROTO_DIR, '**', 'ASVspoof5.dev.track_1.tsv'),
                         recursive=True)
    df = pd.read_csv(hits[0], sep=r'\s+', header=None, names=PROTO_COLS, dtype=str)
    df['filename'] = df['FLAC_FILE_NAME']
    return df


def read_scores(path, label):
    """Read a score file and normalise its score column to `score`.
    Prints the real columns, because a mismatch here is what breaks tonight."""
    df = pd.read_csv(path, sep='\t', dtype={'filename': str})
    cols = list(df.columns)
    candidates = ['cm-score', 'cm_score', 'score', 'logit_bonafide', 'bonafide']
    col = next((c for c in candidates if c in cols), None)
    if col is None:
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        col = numeric[0] if numeric else None
    print(f'  {label:<22} {len(df):>8,} rows   columns: {cols}')
    if col is None:
        print(f'    [FAIL] no usable score column in {label}')
        return None
    if col != 'cm-score':
        print(f'    [NOTE] using "{col}" as the score column')
    out = df[['filename', col]].rename(columns={col: 'score'})
    return out


# ------------------------------------------------------------------- sections
def validate(name, df, proto):
    ok = True

    def check(label, cond, detail=''):
        nonlocal ok
        if not cond:
            ok = False
        print(f'    [{"PASS" if cond else "FAIL"}] {label}{("  " + detail) if detail else ""}')

    print(f'\n  {name}')
    check('row count', len(df) == EXPECTED_DEV_ROWS,
          f'{len(df):,} (expected {EXPECTED_DEV_ROWS:,})')
    check('no duplicate filenames', not df.filename.duplicated().any(),
          f'{int(df.filename.duplicated().sum())} duplicates')
    m = df.merge(proto[['filename', 'KEY', 'SPEAKER_ID', 'ATTACK_LABEL']], on='filename')
    check('join coverage 100%', len(m) == len(df),
          f'{len(m):,} of {len(df):,}')
    check('scores finite', np.isfinite(m.score).all())
    e = eer_pct(m)
    a = auc_pct(m)
    print(f'    EER {e:6.2f}%   AUC {a:.4f}'
          + ('   [WARN] AUC < 0.5: scores may be sign-flipped' if a < 0.5 else ''))
    return m, ok


def identify_combined(seeds, combined):
    """Is xlsr_dev.tsv a 4th seed, a copy of one, or the mean of the three?
    Treating an average as a seed would understate the seed variance."""
    print('\n\n2. WHAT IS xlsr_dev.tsv?')
    print('=========================')
    base = combined[['filename', 'score']].rename(columns={'score': 'comb'})
    for s, df in seeds.items():
        j = base.merge(df[['filename', 'score']], on='filename')
        if np.allclose(j.comb, j.score, atol=1e-6):
            print(f'  identical to seed {s}. Treat it as that seed, not a 4th run.')
            return f'copy of seed {s}'
    stack = None
    for s, df in seeds.items():
        col = df[['filename', 'score']].rename(columns={'score': f's{s}'})
        stack = col if stack is None else stack.merge(col, on='filename')
    j = base.merge(stack, on='filename')
    mean = j[[c for c in j.columns if c.startswith('s')]].mean(axis=1)
    if np.allclose(j.comb, mean, atol=1e-4):
        print('  equals the MEAN of the three seeds (a seed-averaged ensemble).')
        print('  Use it as the headline SSL score, but compute seed variance')
        print('  from the three individual files, never from this one.')
        return 'mean of seeds'
    print('  matches neither a single seed nor their mean - ask Zahira what it is')
    print(f'  corr with seed mean: {np.corrcoef(j.comb, mean)[0, 1]:.6f}')
    return 'unknown'


def seed_agreement(seeds):
    print('\n\n3. DO THE THREE SEEDS AGREE?')
    print('=============================')
    rows = [{'seed': s, 'EER_%': round(eer_pct(df), 3), 'AUC': round(auc_pct(df), 4)}
            for s, df in seeds.items()]
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    spread = tab['EER_%'].max() - tab['EER_%'].min()
    print(f'\n  EER spread across seeds: {spread:.3f} points '
          f'(mean {tab["EER_%"].mean():.2f}, sd {tab["EER_%"].std():.3f})')

    merged = None
    for s, df in seeds.items():
        col = df[['filename', 'score']].rename(columns={'score': f'seed_{s}'})
        merged = col if merged is None else merged.merge(col, on='filename')
    sc = merged[[c for c in merged.columns if c.startswith('seed_')]]
    print('\n  pairwise score correlation between seeds')
    print(sc.corr().round(4).to_string())

    if spread < 0.5:
        print('\n  -> seeds agree closely. The SSL column is stable; report the')
        print('     mean with the seed spread as a footnote.')
    elif spread < 2.0:
        print('\n  -> moderate spread. Report mean +/- spread everywhere, and do')
        print('     not claim a difference smaller than the spread itself.')
    else:
        print('\n  -> LARGE spread. The head is unstable across initialisations.')
        print('     Any RQ2 claim inherits this. Raise it with Zahira today.')
    return tab


def head_to_head(aasist, xlsr, proto):
    print('\n\n4. AASIST vs XLS-R ON DEV  (A09-A16, same 140,950 files)')
    print('=========================================================')
    print(f'  {"detector":<12} {"EER %":>8} {"AUC":>8}')
    print(f'  {"AASIST":<12} {eer_pct(aasist):>8.2f} {auc_pct(aasist):>8.4f}')
    print(f'  {"XLS-R":<12} {eer_pct(xlsr):>8.2f} {auc_pct(xlsr):>8.4f}')

    print('\n  per attack (each attack vs the full bona fide pool)')
    rows = []
    for name, df in [('AASIST', aasist), ('XLS-R', xlsr)]:
        bona = df[df.KEY == 'bonafide']
        for a in sorted(df.loc[df.KEY == 'spoof', 'ATTACK_LABEL'].unique()):
            sub = pd.concat([bona, df[(df.ATTACK_LABEL == a) & (df.KEY == 'spoof')]])
            rows.append({'detector': name, 'attack': a,
                         'n_spoof': int((sub.KEY == 'spoof').sum()),
                         'EER_%': round(eer_pct(sub), 2)})
    tab = pd.DataFrame(rows).pivot(index='attack', columns='detector', values='EER_%')
    tab['diff (XLS-R - AASIST)'] = (tab['XLS-R'] - tab['AASIST']).round(2)
    print(tab.to_string())
    print('\n  Negative diff = XLS-R is better on that attack. What matters for RQ2')
    print('  is whether the two detectors RANK the attacks the same way, not')
    print('  whether one is lower overall.')
    r = tab[['AASIST', 'XLS-R']].corr(method='spearman').iloc[0, 1]
    print(f'\n  Spearman rank correlation of per-attack difficulty: {r:.3f}')
    if r > 0.8:
        print('  -> the architectures find the same attacks hard. Early evidence')
        print('     that the failure modes are shared, not architecture-specific.')
    elif r > 0.4:
        print('  -> partially shared ordering. Worth a close look on the eval slices.')
    else:
        print('  -> the architectures fail on DIFFERENT attacks. That is the')
        print('     interesting answer to RQ2 if it holds on eval.')
    return tab


def main():
    print('1. LOAD AND VALIDATE')
    print('====================')
    proto = dev_protocol()
    print(f'  dev protocol           {len(proto):>8,} rows')

    aasist_raw = read_scores(AASIST_DEV, 'AASIST dev')
    seeds_raw = {s: read_scores(p, f'XLS-R seed {s}') for s, p in SEED_FILES.items()}
    comb_raw = read_scores(COMBINED, 'XLS-R combined')

    aasist, _ = validate('AASIST dev', aasist_raw, proto)
    seeds = {}
    for s, df in seeds_raw.items():
        seeds[s], _ = validate(f'XLS-R seed {s}', df, proto)
    comb, _ = validate('XLS-R combined', comb_raw, proto)

    identify_combined(seeds, comb)
    seed_agreement(seeds)
    head_to_head(aasist, comb, proto)

    print('\n\nDONE. Paste the output back.')
    return aasist, seeds, comb


if __name__ == '__main__':
    aasist_dev, xlsr_seeds, xlsr_comb = main()
