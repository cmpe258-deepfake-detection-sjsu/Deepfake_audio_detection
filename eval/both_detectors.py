"""
CMPE 258 - the main table: shift decomposition across BOTH detectors
====================================================================

Run in the same notebook. Remount Drive first - ssl_detector/ was created after
your last mount, so Colab will not see it otherwise:

    from google.colab import drive
    drive.flush_and_unmount()
    drive.mount('/content/drive', force_remount=True)

Order of output is deliberate:

  PART 0  find Zahira's eval score file (prints the folder so you can correct it)
  PART 1  load and validate both detectors on eval
  PART 2  INVERSION SCAN - fast, and answers the A19 question first
  PART 3  full shift decomposition, both detectors, with bootstrap intervals
  PART 4  side-by-side summary

Part 3 is the slow one. Parts 0-2 finish in under a minute, so the headline
result is on screen long before the bootstrap completes.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------- config
SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'
ZAHIRA_DIR = '/content/drive/MyDrive/XLS-R'
SSL_SCORES = f'{ZAHIRA_DIR}/ssl_detector/scores'

AASIST_EVAL = f'{SHRIYA_DIR}/scores/aasist_eval.tsv'
PROTO_EVAL = f'{SHRIYA_DIR}/keys/ASVspoof5.eval.track_1.tsv'

PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
EXPECTED_ROWS = 680774

UNCODED = '-'
ATTACK = 'ATTACK_LABEL'
SEED = 258
B_BOOT = 200          # raise to 1000 for the final table

FAMILIES = {
    'legacy unit selection': ['A19'],
    'zero-shot TTS':         ['A17', 'A21', 'A22', 'A28', 'A29'],
    'voice conversion':      ['A24', 'A25', 'A26'],
}
MALAFIDE = {'A18': 'A17', 'A30': 'A17', 'A20': None, 'A23': None}
MALACOPULA = ['A27', 'A31', 'A32']


# ---------------------------------------------------------------------- math
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


def auc_val(pos, neg):
    from scipy.stats import rankdata
    if not len(pos) or not len(neg):
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap(df, b=B_BOOT, seed=SEED):
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


# ------------------------------------------------------------------- PART 0
def find_ssl_eval():
    print('PART 0. LOCATE THE XLS-R EVAL SCORES')
    print('=====================================')
    if not os.path.isdir(SSL_SCORES):
        print(f'  [FAIL] {SSL_SCORES} not found.')
        print('  Remount Drive (ssl_detector/ was created after your last mount),')
        print('  then check the path. Parent listing:')
        parent = os.path.dirname(SSL_SCORES)
        if os.path.isdir(parent):
            print('   ', os.listdir(parent))
        return None
    files = sorted(os.listdir(SSL_SCORES))
    print(f'  {SSL_SCORES} contains:')
    for f in files:
        size = os.path.getsize(os.path.join(SSL_SCORES, f)) / 1e6
        print(f'    {f:<45} {size:8.1f} MB')

    evals = [f for f in files if 'eval' in f.lower() and f.endswith(('.tsv', '.csv'))]
    if not evals:
        print('  [FAIL] no eval score file here. Set the path by hand.')
        return None
    # prefer an ensemble file (no "seed" in the name) over a per-seed one
    ensemble = [f for f in evals if 'seed' not in f.lower()]
    pick = (ensemble or evals)[0]
    print(f'\n  using: {pick}'
          + ('' if ensemble else '   [NOTE] this is a per-seed file, not the ensemble'))
    return os.path.join(SSL_SCORES, pick)


# ------------------------------------------------------------------- PART 1
def load(ssl_path):
    print('\n\nPART 1. LOAD AND VALIDATE')
    print('==========================')
    proto = pd.read_csv(PROTO_EVAL, sep=r'\s+', header=None, names=PROTO_COLS, dtype=str)
    proto['filename'] = proto['FLAC_FILE_NAME']
    print(f'  protocol              {len(proto):>8,} rows')

    out = {}
    for name, path in [('AASIST', AASIST_EVAL), ('XLS-R', ssl_path)]:
        raw = pd.read_csv(path, sep='\t', dtype={'filename': str})
        cand = ['cm-score', 'cm_score', 'score', 'logit_bonafide']
        col = next((c for c in cand if c in raw.columns), None)
        if col is None:
            num = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
            col = num[0] if num else None
        print(f'\n  {name}')
        print(f'    columns: {list(raw.columns)}')
        if col != 'cm-score':
            print(f'    [NOTE] using "{col}" as the score column')
        df = raw[['filename', col]].rename(columns={col: 'score'}).merge(
            proto[['filename', 'KEY', 'SPEAKER_ID', 'CODEC', 'CODEC_SEED', ATTACK]],
            on='filename')
        for lab, cond, det in [
                ('row count', len(raw) == EXPECTED_ROWS, f'{len(raw):,}'),
                ('no duplicates', not raw.filename.duplicated().any(), ''),
                ('join coverage 100%', len(df) == len(raw), f'{len(df):,} of {len(raw):,}'),
                ('scores finite', np.isfinite(df.score).all(), '')]:
            print(f'    [{"PASS" if cond else "FAIL"}] {lab}{("  " + det) if det else ""}')
        e = eer_pct(df)
        a = auc_val(df.loc[df.KEY == 'bonafide', 'score'].to_numpy(),
                    df.loc[df.KEY == 'spoof', 'score'].to_numpy())
        print(f'    pooled eval EER {e:6.2f}%   AUC {a:.4f}'
              + ('   [WARN] AUC < 0.5, scores may be sign-flipped' if a < 0.5 else ''))
        out[name] = df
    return out


# ------------------------------------------------------------------- PART 2
def inversion_scan(data):
    print('\n\nPART 2. INVERSION SCAN ON EVAL  (the A19 question)')
    print('==================================================')
    print('  AUC = P(bona fide outscores the attack). Below 0.5 means the')
    print('  detector is BACKWARDS on that attack, not merely weak.\n')
    rows = []
    for name, df in data.items():
        unc = df[df.CODEC == UNCODED]
        bona = unc.loc[unc.KEY == 'bonafide', 'score'].to_numpy()
        bona_med = float(np.median(bona))
        for a in sorted(unc.loc[unc.KEY == 'spoof', ATTACK].unique()):
            x = unc.loc[(unc[ATTACK] == a) & (unc.KEY == 'spoof'), 'score'].to_numpy()
            rows.append({'detector': name, 'attack': a, 'n': len(x),
                         'AUC': round(auc_val(bona, x), 4),
                         'med_shift': round(float(np.median(x)) - bona_med, 2)})
    tab = pd.DataFrame(rows)
    print(tab.pivot(index='attack', columns='detector', values='AUC').to_string())

    print('\n  INVERTED (AUC < 0.5):')
    inv = tab[tab.AUC < 0.5].sort_values('AUC')
    if len(inv):
        for _, r in inv.iterrows():
            print(f'    {r["detector"]:<8} {r["attack"]}   AUC {r["AUC"]:.4f}   '
                  f'median {r["med_shift"]:+.2f} vs bona fide   n={r["n"]:,}')
    else:
        print('    none')
    print('\n  per detector:', {n: int(((tab.detector == n) & (tab.AUC < 0.5)).sum())
                                for n in data})

    # the specific question
    print('\n  A19 (waveform concatenation, the sibling of dev A12):')
    for name in data:
        r = tab[(tab.detector == name) & (tab.attack == 'A19')]
        if len(r):
            v = float(r['AUC'].iloc[0])
            verdict = 'INVERTED' if v < 0.5 else ('near chance' if v < 0.6 else 'handled')
            print(f'    {name:<8} AUC {v:.4f}   {verdict}')
    print('\n  If XLS-R handles A19 while AASIST inverts, the blind spot is')
    print('  confirmed on both concatenation attacks across both partitions.')
    print('  If BOTH invert, vocoder-free synthesis is hard for artifact-based')
    print('  detectors generally - a different and also publishable claim.')
    return tab


# ------------------------------------------------------------------- PART 3
def crossed(df):
    coded = df[df.CODEC != UNCODED]
    n = coded.CODEC.nunique()
    sizes = coded.CODEC_SEED.value_counts()
    return coded[coded.CODEC_SEED.isin(sizes[sizes == n].index)]


def decompose(data):
    print('\n\nPART 3. SHIFT DECOMPOSITION, BOTH DETECTORS')
    print('============================================')
    print(f'  bootstrap B={B_BOOT}; this is the slow part.\n')
    rows = []
    for name, df in data.items():
        print(f'  {name}...')
        unc = df[df.CODEC == UNCODED]
        bona = unc[unc.KEY == 'bonafide']
        cb = crossed(df)

        def add(slice_name, kind, sub):
            lo, hi = bootstrap(sub)
            rows.append({'detector': name, 'slice': slice_name, 'kind': kind,
                         'n_spoof': int((sub.KEY == 'spoof').sum()),
                         'EER_%': round(eer_pct(sub), 2),
                         'lo': round(lo, 2), 'hi': round(hi, 2)})

        add('ALL attacks, uncoded', 'reference', unc)
        for fam, atks in FAMILIES.items():
            sel = unc[(unc[ATTACK].isin(atks)) & (unc.KEY == 'spoof')]
            if len(sel):
                add(f'{fam} ({"+".join(atks)})', 'generator', pd.concat([bona, sel]))
        for atk, src in MALAFIDE.items():
            sel = unc[(unc[ATTACK] == atk) & (unc.KEY == 'spoof')]
            if len(sel):
                lab = f'Malafide {atk}' + (f' (from {src})' if src else ' (src in dev)')
                add(lab, 'adversarial', pd.concat([bona, sel]))
        sel = unc[(unc[ATTACK].isin(MALACOPULA)) & (unc.KEY == 'spoof')]
        if len(sel):
            add(f'Malacopula ({"+".join(MALACOPULA)})', 'adversarial', pd.concat([bona, sel]))
        for atk, src in MALAFIDE.items():
            if src is None:
                continue
            s_sel = unc[(unc[ATTACK] == src) & (unc.KEY == 'spoof')]
            a_sel = unc[(unc[ATTACK] == atk) & (unc.KEY == 'spoof')]
            if len(s_sel) and len(a_sel):
                add(f'  source {src} alone', 'adv-paired', pd.concat([bona, s_sel]))
                add(f'  -> {atk} adversarial', 'adv-paired', pd.concat([bona, a_sel]))
        for c in sorted(cb.CODEC.unique()):
            add(f'codec {c}', 'codec', cb[cb.CODEC == c])
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- PART 4
def report(tab):
    print('\n\nPART 4. SIDE BY SIDE  (EER %, 95% speaker-level CI)')
    print('===================================================')
    for kind in ['reference', 'generator', 'adversarial', 'adv-paired', 'codec']:
        part = tab[tab['kind'] == kind]
        if not len(part):
            continue
        print(f'\n{kind.upper()}')
        for sl in part['slice'].unique():
            line = f'  {sl:<38}'
            for det in tab.detector.unique():
                r = part[(part['slice'] == sl) & (part.detector == det)]
                if len(r):
                    r = r.iloc[0]
                    line += f'  {det} {r["EER_%"]:>6.2f} [{r["lo"]:>5.2f},{r["hi"]:>5.2f}]'
            print(line)

    print('\n\n  SPREAD WITHIN EACH SHIFT TYPE  (what answers "which shift breaks it")')
    for det in tab.detector.unique():
        print(f'\n  {det}')
        for kind in ['generator', 'adversarial', 'codec']:
            p = tab[(tab.detector == det) & (tab['kind'] == kind)]
            if len(p):
                print(f'    {kind:<13} {p["EER_%"].min():6.2f} to {p["EER_%"].max():6.2f}'
                      f'   (spread {p["EER_%"].max() - p["EER_%"].min():6.2f} points)')


def main():
    ssl_path = find_ssl_eval()
    if ssl_path is None:
        return None
    data = load(ssl_path)
    inv = inversion_scan(data)
    tab = decompose(data)
    report(tab)
    out = '/content/drive/MyDrive/258_results_both_detectors.csv'
    tab.to_csv(out, index=False)
    print(f'\n  saved: {out}')
    print('\n\nDONE. Paste the output back.')
    return tab, inv


if __name__ == '__main__':
    both_table, inversion_table = main()
