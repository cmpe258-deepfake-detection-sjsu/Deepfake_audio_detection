"""
CMPE 258 - fold DFADD into the main results table
==================================================

DFADD is the open-world test: five diffusion / flow-matching TTS systems that
ASVspoof 5 never saw, with VCTK bona fide speech. Shriya scored it with AASIST
weeks ago, but the numbers only live in her metrics file. This script puts them
in the main table next to the ASVspoof 5 slices, with intervals.

Run it in the same Colab notebook as the harness (Drive mounted).

  PART 0  locate the score files (AASIST is required, XLS-R is optional)
  PART 1  load, validate, and reproduce Shriya's published DFADD EERs
  PART 2  per-generator EER and AUC with bootstrap intervals
  PART 3  bona fide check: is real VCTK speech scored like real ASVspoof 5 speech?
  PART 4  append the DFADD rows to the main table and save

Two things are different from the ASVspoof 5 slices, on purpose:

  * The interval is an UTTERANCE bootstrap, not a speaker bootstrap. The DFADD
    test split has two speakers (p227, p228). Resampling two speakers gives
    three possible outcomes, which is not an interval. Utterance resampling
    understates uncertainty across speakers, so these intervals are narrower
    than they should be and the table says which kind each row uses.
  * All five generators share the same 755 bona fide files, so the five rows
    are not independent of each other.

If no XLS-R DFADD score file exists, the XLS-R rows are written as "not run".
Nothing is estimated.
"""

import glob
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------- config
SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'
ZAHIRA_DIR = '/content/drive/MyDrive/XLS-R'
SSL_SCORES = f'{ZAHIRA_DIR}/ssl_detector/scores'

AASIST_DFADD = f'{SHRIYA_DIR}/scores/aasist_dfadd.tsv'
AASIST_DEV = f'{SHRIYA_DIR}/scores/aasist_dev.tsv'
DEV_KEYS = f'{SHRIYA_DIR}/keys/dev_cm_keys.tsv'
DFADD_PROTO = f'{SHRIYA_DIR}/dfadd/dfadd_test_protocol.tsv'

MAIN_TABLE = '/content/drive/MyDrive/258_results_both_detectors.csv'
DFADD_TABLE = '/content/drive/MyDrive/258_dfadd_table.csv'

SEED = 258
B_BOOT = 1000
EXPECTED_ROWS = 3755
EXPECTED_BONAFIDE = 755

# Shriya's published AASIST numbers (results/258_aasist_metrics.csv).
PUBLISHED_AASIST = {'GradTTS': 8.85, 'MatchaTTS': 4.14, 'NaturalSpeech2': 0.37,
                    'PflowTTS': 1.99, 'StyleTTS2': 12.99, 'pooled': 7.81}
TOL = 0.02          # EER points; the published values are rounded to 2 dp

KIND = 'cross-dataset (DFADD), utterance bootstrap'


# ---------------------------------------------------------------------- math
def _eer(tar, non):
    """Same implementation as the rest of the harness. tar = bona fide scores."""
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


def auc_val(pos, neg):
    from scipy.stats import rankdata
    if not len(pos) or not len(neg):
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def utt_bootstrap(bona, spoof, b=B_BOOT, seed=SEED):
    """95% CI on EER, resampling utterances within each class."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        v = _eer(bona[rng.integers(0, bona.size, bona.size)],
                 spoof[rng.integers(0, spoof.size, spoof.size)])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def check(label, ok, detail=''):
    print(f'    [{"PASS" if ok else "FAIL"}] {label}{("  " + detail) if detail else ""}')
    return bool(ok)


# ------------------------------------------------------------------- PART 0
def locate():
    print('PART 0. LOCATE THE SCORE FILES')
    print('==============================')
    ok = True
    for lab, p in [('AASIST DFADD scores', AASIST_DFADD), ('DFADD protocol', DFADD_PROTO)]:
        ok &= check(lab, os.path.exists(p), p)
    if not ok:
        print('\n  Remount Drive and check SHRIYA_DIR at the top of this script.')
        return None, None

    ssl = None
    if os.path.isdir(SSL_SCORES):
        hits = sorted(glob.glob(os.path.join(SSL_SCORES, '*dfadd*'))
                      + glob.glob(os.path.join(SSL_SCORES, '*DFADD*')))
        hits = [h for h in dict.fromkeys(hits) if h.endswith(('.tsv', '.csv'))]
        ensemble = [h for h in hits if 'seed' not in os.path.basename(h).lower()]
        if hits:
            ssl = (ensemble or hits)[0]
            print(f'    [PASS] XLS-R DFADD scores  {ssl}'
                  + ('' if ensemble else '   [NOTE] per-seed file, not the ensemble'))
    if ssl is None:
        print(f'    [NOTE] no XLS-R DFADD score file in {SSL_SCORES}')
        print('           XLS-R rows will be written as "not run".')
    return AASIST_DFADD, ssl


# ------------------------------------------------------------------- PART 1
def read_scores(path):
    raw = pd.read_csv(path, sep='\t', dtype={'filename': str})
    cand = ['cm-score', 'cm_score', 'score', 'logit_bonafide']
    col = next((c for c in cand if c in raw.columns), None)
    if col is None:
        num = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        col = num[0] if num else None
    return raw, col


def load(aasist_path, ssl_path):
    print('\n\nPART 1. LOAD AND VALIDATE')
    print('==========================')
    proto = pd.read_csv(DFADD_PROTO, sep='\t', dtype=str)
    print(f'  protocol columns: {list(proto.columns)}')
    need = {'SPEAKER_ID', 'filename', 'GENERATOR', 'KEY'}
    if not need.issubset(proto.columns):
        print(f'  [FAIL] protocol is missing {sorted(need - set(proto.columns))}. Stop and paste this.')
        return None
    print('  files per generator and speaker:')
    print(proto.groupby(['GENERATOR', 'SPEAKER_ID']).size().unstack(fill_value=0).to_string())

    out = {}
    for name, path in [('AASIST', aasist_path), ('XLS-R', ssl_path)]:
        if path is None:
            continue
        raw, col = read_scores(path)
        print(f'\n  {name}')
        print(f'    columns: {list(raw.columns)}')
        if col != 'cm-score':
            print(f'    [NOTE] using "{col}" as the score column')
        df = raw[['filename', col]].rename(columns={col: 'score'}).merge(
            proto[['filename', 'KEY', 'SPEAKER_ID', 'GENERATOR']], on='filename')
        good = True
        good &= check('row count', len(raw) == EXPECTED_ROWS, f'{len(raw):,}')
        good &= check('no duplicates', not raw.filename.duplicated().any())
        good &= check('join coverage 100%', len(df) == len(raw), f'{len(df):,} of {len(raw):,}')
        good &= check('scores finite', bool(np.isfinite(df.score).all()))
        good &= check('bona fide count', int((df.KEY == 'bonafide').sum()) == EXPECTED_BONAFIDE,
                      str(int((df.KEY == 'bonafide').sum())))
        if not good:
            print(f'    {name} failed validation and is left out. Paste this output.')
            continue
        out[name] = df

    if 'AASIST' in out:
        print('\n  Reproducing the published AASIST DFADD EERs:')
        df = out['AASIST']
        bona = df.loc[df.KEY == 'bonafide', 'score'].to_numpy()
        all_ok = True
        for gen, pub in PUBLISHED_AASIST.items():
            sp = df[df.KEY == 'spoof'] if gen == 'pooled' else df[(df.KEY == 'spoof') & (df.GENERATOR == gen)]
            got = _eer(bona, sp.score.to_numpy())
            all_ok &= check(f'{gen:<15} published {pub:6.2f}  ours {got:6.2f}', abs(got - pub) <= TOL)
        if not all_ok:
            print('    [STOP] our EER does not match the published one. Do not use the table until this is explained.')
    return out


# ------------------------------------------------------------------- PART 2
def per_generator(data):
    print('\n\nPART 2. PER-GENERATOR RESULTS')
    print('=============================')
    print(f'  intervals: utterance bootstrap, B = {B_BOOT}, seed {SEED}\n')
    rows = []
    for name, df in data.items():
        bona = df.loc[df.KEY == 'bonafide', 'score'].to_numpy()
        gens = sorted(g for g in df.loc[df.KEY == 'spoof', 'GENERATOR'].unique())
        for gen in gens + ['pooled']:
            sp = df[df.KEY == 'spoof'] if gen == 'pooled' else df[(df.KEY == 'spoof') & (df.GENERATOR == gen)]
            x = sp.score.to_numpy()
            lo, hi = utt_bootstrap(bona, x)
            rows.append({'detector': name,
                         'slice': f'DFADD {gen}' if gen != 'pooled' else 'DFADD pooled (5 generators)',
                         'kind': KIND, 'n_spoof': len(x), 'n_bonafide': len(bona),
                         'EER_%': round(_eer(bona, x), 2), 'lo': round(lo, 2), 'hi': round(hi, 2),
                         'AUC': round(auc_val(bona, x), 4)})
    tab = pd.DataFrame(rows)
    for name in data:
        print(f'  {name}')
        for _, r in tab[tab.detector == name].iterrows():
            flag = '   [INVERTED: AUC < 0.5]' if r['AUC'] < 0.5 else ''
            print(f'    {r["slice"]:<30} EER {r["EER_%"]:6.2f}  [{r["lo"]:6.2f}, {r["hi"]:6.2f}]'
                  f'   AUC {r["AUC"]:.4f}   n_spoof {r["n_spoof"]:,}{flag}')
        print()
    return tab


# ------------------------------------------------------------------- PART 3
def bonafide_check(data):
    print('\nPART 3. BONA FIDE CHECK  (AASIST only: needs the dev scores)')
    print('============================================================')
    print('  DFADD real speech is VCTK studio audio; ASVspoof 5 real speech is LibriVox.')
    print('  If the detector scores VCTK lower, part of any DFADD error is the detector')
    print('  rejecting real speech, not missing fakes.\n')
    if 'AASIST' not in data or not (os.path.exists(AASIST_DEV) and os.path.exists(DEV_KEYS)):
        print('  [SKIP] AASIST dev scores or dev keys not found.')
        return None
    raw, col = read_scores(AASIST_DEV)
    keys = pd.read_csv(DEV_KEYS, sep='\t', dtype=str)
    kcol = next((c for c in ['cm-label', 'cm_label', 'KEY', 'label'] if c in keys.columns), None)
    if kcol is None or 'filename' not in keys.columns:
        print(f'  [SKIP] could not read the dev keys. Columns: {list(keys.columns)}')
        return None
    dev = raw[['filename', col]].rename(columns={col: 'score'}).merge(
        keys[['filename', kcol]].rename(columns={kcol: 'KEY'}), on='filename')
    dev_bona = dev.loc[dev.KEY == 'bonafide', 'score'].to_numpy()
    dfa = data['AASIST']
    vctk = dfa.loc[dfa.KEY == 'bonafide', 'score'].to_numpy()
    rows = []
    for lab, x in [('ASVspoof 5 dev bona fide', dev_bona), ('DFADD bona fide (VCTK)', vctk)]:
        rows.append({'set': lab, 'n': len(x), 'median': round(float(np.median(x)), 2),
                     'p10': round(float(np.percentile(x, 10)), 2),
                     'p90': round(float(np.percentile(x, 90)), 2)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    a = auc_val(dev_bona, vctk)
    print(f'\n  P(a dev bona fide file outscores a VCTK bona fide file) = {a:.3f}')
    print('  0.5 means the two sets of real speech are scored alike; well above 0.5')
    print('  means VCTK real speech looks less real to AASIST.')
    return t


# ------------------------------------------------------------------- PART 4
def save(tab, data):
    print('\n\nPART 4. SAVE')
    print('============')
    if 'XLS-R' not in data:
        aas = tab[tab.detector == 'AASIST']
        nr = aas.assign(detector='XLS-R', **{'EER_%': np.nan, 'lo': np.nan, 'hi': np.nan, 'AUC': np.nan})
        nr['kind'] = 'cross-dataset (DFADD), not run'
        tab = pd.concat([tab, nr], ignore_index=True)
        print('  XLS-R DFADD rows written as "not run" (no score file).')
    tab.to_csv(DFADD_TABLE, index=False)
    print(f'  saved: {DFADD_TABLE}')

    if not os.path.exists(MAIN_TABLE):
        print(f'  [NOTE] {MAIN_TABLE} not found, so the main table was not updated.')
        return tab
    main = pd.read_csv(MAIN_TABLE)
    before = len(main)
    main = main[~main['slice'].astype(str).str.startswith('DFADD')]      # safe to rerun
    add = tab[[c for c in main.columns if c in tab.columns]]
    merged = pd.concat([main, add], ignore_index=True)
    merged.to_csv(MAIN_TABLE, index=False)
    print(f'  main table: {before} rows -> {len(merged)} rows ({len(add)} DFADD rows)')
    print(f'  saved: {MAIN_TABLE}')
    return tab


def main():
    aasist_path, ssl_path = locate()
    if aasist_path is None:
        return None
    data = load(aasist_path, ssl_path)
    if not data:
        return None
    tab = per_generator(data)
    bonafide_check(data)
    tab = save(tab, data)
    print('\n\nDONE. Paste the output back.')
    return tab


if __name__ == '__main__':
    dfadd_table = main()
