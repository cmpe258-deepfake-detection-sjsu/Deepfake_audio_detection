"""
CMPE 258 - error rates at a fixed, dev-selected threshold
==========================================================

Every number in the main table is an EER. EER re-picks the threshold for each
slice, so it answers "how separable are the two classes here?" It cannot answer
"what happens to a deployed detector?", because a deployed detector has ONE
threshold, chosen in advance.

This script freezes one threshold per detector on dev (the dev EER point) and
then reports, on every eval slice:

  FRR  false rejection rate: real speech called fake   (bona fide score <  threshold)
  FAR  false acceptance rate: fake speech called real  (spoof score     >= threshold)

That splits each slice's error by class, which EER cannot do.

Run it in the same Colab notebook as the harness (Drive mounted).

  PART 0  locate dev and eval score files for both detectors
  PART 1  load and validate
  PART 2  pick the threshold on dev and freeze it
  PART 3  FRR per codec condition (bona fide), with the paired change from uncoded
  PART 4  FAR per attack on uncoded audio
  PART 5  FAR per codec condition (spoof), with the paired change from uncoded
  PART 6  DFADD at the same frozen threshold (AASIST only)
  PART 7  save

Intervals are speaker bootstraps on eval (the threshold stays fixed, it is not
re-selected inside the bootstrap). DFADD uses an utterance bootstrap because it
has two speakers.

Thresholds are selected on dev only and never touched after that.
"""

import json
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------- config
SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'
ZAHIRA_DIR = '/content/drive/MyDrive/XLS-R'
SSL_SCORES = f'{ZAHIRA_DIR}/ssl_detector/scores'

AASIST_DEV = f'{SHRIYA_DIR}/scores/aasist_dev.tsv'
AASIST_EVAL = f'{SHRIYA_DIR}/scores/aasist_eval.tsv'
AASIST_DFADD = f'{SHRIYA_DIR}/scores/aasist_dfadd.tsv'
DEV_KEYS = f'{SHRIYA_DIR}/keys/dev_cm_keys.tsv'
PROTO_EVAL = f'{SHRIYA_DIR}/keys/ASVspoof5.eval.track_1.tsv'
DFADD_PROTO = f'{SHRIYA_DIR}/dfadd/dfadd_test_protocol.tsv'

OUT_TABLE = '/content/drive/MyDrive/258_fixed_threshold.csv'
OUT_META = '/content/drive/MyDrive/258_fixed_threshold_meta.json'

PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
EXPECTED_EVAL = 680774
EXPECTED_DEV = 140950
PUBLISHED_AASIST_DEV_EER = 15.20

UNCODED = '-'
ATTACK = 'ATTACK_LABEL'
SEED = 258
B_BOOT = 1000


# ---------------------------------------------------------------------- math
def _eer(tar, non):
    """Same implementation as the rest of the harness. tar = bona fide scores."""
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


def pick_threshold(bona, spoof):
    """Threshold where FRR and FAR are closest on dev. Decision: real if score >= t."""
    cand = np.unique(np.concatenate((bona, spoof)))
    b = np.sort(bona)
    s = np.sort(spoof)
    frr = np.searchsorted(b, cand, side='left') / b.size          # bona fide below t
    far = 1.0 - np.searchsorted(s, cand, side='left') / s.size    # spoof at or above t
    i = int(np.argmin(np.abs(frr - far)))
    return float(cand[i]), 100 * float(frr[i]), 100 * float(far[i])


def err_flags(df, thr):
    """1 where the file is misclassified at the frozen threshold."""
    isb = (df.KEY == 'bonafide').to_numpy()
    sc = df['score'].to_numpy()
    return np.where(isb, sc < thr, sc >= thr).astype(float)


def spk_counts(spk, err, speakers):
    """Per-speaker (errors, files) aligned to a fixed speaker list."""
    idx = pd.Index(speakers).get_indexer(spk)
    e = np.bincount(idx, weights=err, minlength=len(speakers))
    n = np.bincount(idx, minlength=len(speakers)).astype(float)
    return e, n


def rate_ci(e, n, draws):
    """Rate in percent with a speaker-bootstrap CI. draws: (B, S) speaker indices."""
    if n.sum() == 0:
        return np.nan, np.nan, np.nan
    num = e[draws].sum(axis=1)
    den = n[draws].sum(axis=1)
    v = 100 * num[den > 0] / den[den > 0]
    return 100 * e.sum() / n.sum(), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def diff_ci(e1, n1, e0, n0, draws):
    """(rate1 - rate0) in points, same speaker draws for both."""
    if n1.sum() == 0 or n0.sum() == 0:
        return np.nan, np.nan, np.nan
    d1, d0 = n1[draws].sum(axis=1), n0[draws].sum(axis=1)
    ok = (d1 > 0) & (d0 > 0)
    v = 100 * (e1[draws].sum(axis=1)[ok] / d1[ok] - e0[draws].sum(axis=1)[ok] / d0[ok])
    return (100 * (e1.sum() / n1.sum() - e0.sum() / n0.sum()),
            float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def check(label, ok, detail=''):
    print(f'    [{"PASS" if ok else "FAIL"}] {label}{("  " + detail) if detail else ""}')
    return bool(ok)


def read_scores(path):
    raw = pd.read_csv(path, sep='\t', dtype={'filename': str})
    cand = ['cm-score', 'cm_score', 'score', 'logit_bonafide']
    col = next((c for c in cand if c in raw.columns), None)
    if col is None:
        num = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        col = num[0] if num else None
    return raw[['filename', col]].rename(columns={col: 'score'}), col


# ------------------------------------------------------------------- PART 0
def find_ssl(tag, expected_rows):
    """First XLS-R score file whose name contains `tag` and has the expected row count.
    Ensemble files (no 'seed' in the name) are tried before per-seed files."""
    if not os.path.isdir(SSL_SCORES):
        return None
    files = sorted(f for f in os.listdir(SSL_SCORES)
                   if tag in f.lower() and f.endswith(('.tsv', '.csv')))
    files = [f for f in files if 'seed' not in f.lower()] + [f for f in files if 'seed' in f.lower()]
    for f in files:
        p = os.path.join(SSL_SCORES, f)
        try:
            with open(p) as fh:
                rows = sum(1 for _ in fh) - 1
        except OSError:
            continue
        if rows == expected_rows:
            return p
        print(f'      skipped {f}: {rows:,} rows, expected {expected_rows:,}')
    return None


def locate():
    print('PART 0. LOCATE THE SCORE FILES')
    print('==============================')
    ok = True
    for lab, p in [('AASIST dev scores', AASIST_DEV), ('AASIST eval scores', AASIST_EVAL),
                   ('dev keys', DEV_KEYS), ('eval protocol', PROTO_EVAL)]:
        ok &= check(lab, os.path.exists(p), p)
    if not ok:
        print('\n  Remount Drive and check SHRIYA_DIR at the top of this script.')
        return None
    paths = {'AASIST': (AASIST_DEV, AASIST_EVAL)}
    sd, se = find_ssl('dev', EXPECTED_DEV), find_ssl('eval', EXPECTED_EVAL)
    if sd and se:
        check('XLS-R dev scores', True, sd)
        check('XLS-R eval scores', True, se)
        if ('seed' in os.path.basename(sd).lower()) != ('seed' in os.path.basename(se).lower()):
            print('    [WARN] dev and eval are not the same kind of file (ensemble vs per-seed).')
            print('           The threshold must come from the same model that scored eval. Stop and check.')
        paths['XLS-R'] = (sd, se)
    else:
        print(f'    [NOTE] XLS-R dev or eval scores not found in {SSL_SCORES}; running AASIST only.')
        if os.path.isdir(SSL_SCORES):
            print('           folder contains:', sorted(os.listdir(SSL_SCORES)))
    return paths


# ------------------------------------------------------------------- PART 1
def load(paths):
    print('\n\nPART 1. LOAD AND VALIDATE')
    print('==========================')
    proto = pd.read_csv(PROTO_EVAL, sep=r'\s+', header=None, names=PROTO_COLS, dtype=str)
    proto['filename'] = proto['FLAC_FILE_NAME']
    keys = pd.read_csv(DEV_KEYS, sep='\t', dtype=str)
    kcol = next((c for c in ['cm-label', 'cm_label', 'KEY', 'label'] if c in keys.columns), None)
    if kcol is None or 'filename' not in keys.columns:
        print(f'  [FAIL] cannot read dev keys. Columns: {list(keys.columns)}. Stop and paste this.')
        return None
    keys = keys[['filename', kcol]].rename(columns={kcol: 'KEY'})
    print(f'  eval protocol {len(proto):,} rows; dev keys {len(keys):,} rows')

    data = {}
    for name, (pdev, peval) in paths.items():
        print(f'\n  {name}')
        dv, c1 = read_scores(pdev)
        ev, c2 = read_scores(peval)
        if c1 != 'cm-score' or c2 != 'cm-score':
            print(f'    [NOTE] score columns used: dev "{c1}", eval "{c2}"')
        dev = dv.merge(keys, on='filename')
        evl = ev.merge(proto[['filename', 'KEY', 'SPEAKER_ID', 'CODEC', ATTACK]], on='filename')
        good = True
        good &= check('dev row count', len(dv) == EXPECTED_DEV, f'{len(dv):,}')
        good &= check('eval row count', len(ev) == EXPECTED_EVAL, f'{len(ev):,}')
        good &= check('dev join coverage 100%', len(dev) == len(dv))
        good &= check('eval join coverage 100%', len(evl) == len(ev))
        good &= check('scores finite', bool(np.isfinite(dev.score).all() and np.isfinite(evl.score).all()))
        if not good:
            print(f'    {name} failed validation and is left out. Paste this output.')
            continue
        data[name] = (dev, evl)
    return data


# ------------------------------------------------------------------- PART 2
def thresholds(data):
    print('\n\nPART 2. FREEZE THE THRESHOLD ON DEV')
    print('===================================')
    print('  Decision rule: call a file real if score >= threshold.\n')
    thr = {}
    for name, (dev, evl) in data.items():
        b = dev.loc[dev.KEY == 'bonafide', 'score'].to_numpy()
        s = dev.loc[dev.KEY == 'spoof', 'score'].to_numpy()
        t, frr, far = pick_threshold(b, s)
        eer = _eer(b, s)
        print(f'  {name}')
        print(f'    threshold {t:.4f}    dev FRR {frr:.2f}%   dev FAR {far:.2f}%   dev EER {eer:.2f}%')
        check('FRR and FAR agree with the dev EER', abs((frr + far) / 2 - eer) < 0.05)
        if name == 'AASIST':
            check(f'dev EER matches the published {PUBLISHED_AASIST_DEV_EER:.2f}%',
                  abs(eer - PUBLISHED_AASIST_DEV_EER) < 0.02)
        thr[name] = {'threshold': t, 'dev_FRR_%': round(frr, 2), 'dev_FAR_%': round(far, 2),
                     'dev_EER_%': round(eer, 2)}
    return thr


# -------------------------------------------------------------- PARTS 3 to 5
def eval_rates(data, thr):
    rows = []
    for name, (dev, evl) in data.items():
        t = thr[name]['threshold']
        evl = evl.assign(err=err_flags(evl, t))
        speakers = np.sort(evl.SPEAKER_ID.unique())
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(speakers), (B_BOOT, len(speakers)))

        def cnt(mask):
            sub = evl[mask]
            return spk_counts(sub.SPEAKER_ID.to_numpy(), sub.err.to_numpy(), speakers)

        isb = evl.KEY == 'bonafide'
        unc = evl.CODEC == UNCODED
        codecs = sorted(c for c in evl.CODEC.unique() if c != UNCODED)

        def add(slice_, kind, cls, e, n, ref=None):
            r, lo, hi = rate_ci(e, n, draws)
            row = {'detector': name, 'slice': slice_, 'kind': kind,
                   'class': cls, 'metric': 'FRR' if cls == 'bona fide' else 'FAR',
                   'n': int(n.sum()), 'rate_%': round(r, 2), 'lo': round(lo, 2), 'hi': round(hi, 2),
                   'delta_vs_uncoded': np.nan, 'd_lo': np.nan, 'd_hi': np.nan}
            if ref is not None:
                d, dlo, dhi = diff_ci(e, n, ref[0], ref[1], draws)
                row.update({'delta_vs_uncoded': round(d, 2), 'd_lo': round(dlo, 2), 'd_hi': round(dhi, 2)})
            rows.append(row)

        # bona fide by codec condition
        b0 = cnt(isb & unc)
        add('uncoded', 'codec', 'bona fide', *b0)
        for c in codecs:
            add(f'codec {c}', 'codec', 'bona fide', *cnt(isb & (evl.CODEC == c)), ref=b0)

        # spoof by attack, uncoded only
        s0 = cnt(~isb & unc)
        add('ALL attacks, uncoded', 'attack', 'spoof', *s0)
        for a in sorted(evl.loc[~isb, ATTACK].unique()):
            add(f'attack {a}', 'attack', 'spoof', *cnt(~isb & unc & (evl[ATTACK] == a)))

        # spoof by codec condition
        add('uncoded', 'codec', 'spoof', *s0)
        for c in codecs:
            add(f'codec {c}', 'codec', 'spoof', *cnt(~isb & (evl.CODEC == c)), ref=s0)
    return pd.DataFrame(rows)


def show(tab, thr):
    dets = list(thr)

    def block(title, note, kind, cls, with_delta):
        print(f'\n\n{title}')
        print('=' * len(title))
        print(note + '\n')
        sub = tab[(tab.kind == kind) & (tab['class'] == cls)]
        head = f'  {"slice":<22}'
        for d in dets:
            head += f'{d + " rate [95% CI]":>30}' + (f'{"change vs uncoded":>28}' if with_delta else '')
        print(head)
        for sl in sub[sub.detector == dets[0]]['slice']:
            line = f'  {sl:<22}'
            for d in dets:
                r = sub[(sub.detector == d) & (sub['slice'] == sl)]
                if r.empty:
                    line += f'{"-":>30}' + (f'{"":>28}' if with_delta else '')
                    continue
                r = r.iloc[0]
                line += f'{r["rate_%"]:>10.2f} [{r["lo"]:6.2f}, {r["hi"]:6.2f}]  '
                if with_delta:
                    if np.isfinite(r['delta_vs_uncoded']):
                        star = ' ' if r['d_lo'] <= 0 <= r['d_hi'] else '*'
                        line += f'{r["delta_vs_uncoded"]:>+9.2f} [{r["d_lo"]:+6.2f}, {r["d_hi"]:+6.2f}]{star}'
                    else:
                        line += f'{"(reference)":>28}'
            print(line)
        if with_delta:
            print('\n  * = the interval on the change excludes zero')

    block('PART 3. FRR BY CODEC  (real speech called fake)',
          '  How much real speech does each codec push below the frozen threshold?',
          'codec', 'bona fide', True)
    block('PART 4. FAR BY ATTACK, UNCODED  (fake speech called real)',
          '  Above 50% means the detector accepts most of that attack as real.',
          'attack', 'spoof', False)
    block('PART 5. FAR BY CODEC  (fake speech called real)',
          '  If codecs mostly raise FRR and leave FAR flat or lower, the codec error is on the real-speech side.',
          'codec', 'spoof', True)


# ------------------------------------------------------------------- PART 6
def dfadd(thr):
    print('\n\nPART 6. DFADD AT THE SAME FROZEN THRESHOLD  (AASIST only)')
    print('=========================================================')
    if 'AASIST' not in thr or not (os.path.exists(AASIST_DFADD) and os.path.exists(DFADD_PROTO)):
        print('  [SKIP] AASIST DFADD scores or protocol not found.')
        return pd.DataFrame()
    t = thr['AASIST']['threshold']
    sc, _ = read_scores(AASIST_DFADD)
    proto = pd.read_csv(DFADD_PROTO, sep='\t', dtype=str)
    df = sc.merge(proto[['filename', 'KEY', 'GENERATOR']], on='filename')
    df = df.assign(err=err_flags(df, t))
    rng = np.random.default_rng(SEED)
    rows = []

    def add(label, cls, x):
        x = x.to_numpy()
        boot = 100 * x[rng.integers(0, x.size, (B_BOOT, x.size))].mean(axis=1)
        rows.append({'detector': 'AASIST', 'slice': label, 'kind': 'DFADD (utterance bootstrap)',
                     'class': cls, 'metric': 'FRR' if cls == 'bona fide' else 'FAR', 'n': int(x.size),
                     'rate_%': round(100 * x.mean(), 2),
                     'lo': round(float(np.percentile(boot, 2.5)), 2),
                     'hi': round(float(np.percentile(boot, 97.5)), 2),
                     'delta_vs_uncoded': np.nan, 'd_lo': np.nan, 'd_hi': np.nan})

    add('DFADD bona fide (VCTK)', 'bona fide', df.loc[df.KEY == 'bonafide', 'err'])
    for g in sorted(df.loc[df.KEY == 'spoof', 'GENERATOR'].unique()):
        add(f'DFADD {g}', 'spoof', df.loc[(df.KEY == 'spoof') & (df.GENERATOR == g), 'err'])
    add('DFADD pooled (5 generators)', 'spoof', df.loc[df.KEY == 'spoof', 'err'])
    out = pd.DataFrame(rows)
    print(f'  threshold {t:.4f}, selected on ASVspoof 5 dev\n')
    for _, r in out.iterrows():
        print(f'  {r["slice"]:<30} {r["metric"]} {r["rate_%"]:6.2f}%  [{r["lo"]:6.2f}, {r["hi"]:6.2f}]   n={r["n"]:,}')
    return out


def main():
    paths = locate()
    if paths is None:
        return None
    data = load(paths)
    if not data:
        return None
    thr = thresholds(data)
    tab = eval_rates(data, thr)
    show(tab, thr)
    tab = pd.concat([tab, dfadd(thr)], ignore_index=True)

    print('\n\nPART 7. SAVE')
    print('============')
    tab.to_csv(OUT_TABLE, index=False)
    meta = {'rule': 'real if score >= threshold; threshold = dev EER point, frozen',
            'seed': SEED, 'bootstrap_draws': B_BOOT,
            'detectors': {k: dict(v, dev_file=os.path.basename(paths[k][0]),
                                  eval_file=os.path.basename(paths[k][1])) for k, v in thr.items()}}
    with open(OUT_META, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'  saved: {OUT_TABLE}  ({len(tab)} rows)')
    print(f'  saved: {OUT_META}')
    print('\n\nDONE. Paste the output back.')
    return tab


if __name__ == '__main__':
    fixed_threshold_table = main()
