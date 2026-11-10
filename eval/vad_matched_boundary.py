"""
CMPE 258 - VAD control on the BOUNDARY-MATCHED subset
=====================================================

Why this exists
---------------
Shriya's VAD was run independently on each coded version rather than once on
the uncoded original with the boundaries reused. For C04 and C07 that made no
difference (median lead-silence difference from uncoded: 0.000 s, mean under
1 ms). For C08 and C10 it did: the VAD cuts on average 62-82 ms MORE leading
silence and 105-133 ms more trailing silence than on the uncoded original.

So on those two conditions the trimmed file is not "the uncoded file minus
silence", it is that minus ~200 ms of additional content. The detectors are
known to be unstable on sub-100 ms windows, so a positive difference-in-
differences on exactly those two conditions cannot be attributed to the codec.

Fixing it properly means re-trimming with shared boundaries and rescoring.
This is the cheap rescue: keep only the seeds whose VAD boundaries already
agree across all five conditions, and run the same DiD there. No new audio,
no GPU, uses the scores that already exist.

Caveat to report: selecting on boundary agreement is not random. Utterances
whose VAD agrees under an 8 kHz codec may be cleaner or louder than average.
The subset result is matched but not a random sample of the original, so it
is evidence the effect is not silence-driven, not a replacement estimate of
the effect size. Both are stated in the output.

Set MODEL to 'xlsr' or 'aasist'. Read-only. A few minutes.
"""

import os

import numpy as np
import pandas as pd

MODEL = 'xlsr'                 # 'xlsr' or 'aasist'
DRIVE = '/content/drive/MyDrive'
TOL_S = 0.020                  # boundary agreement tolerance, seconds
REPORT_TOLS = [0.005, 0.010, 0.020, 0.050]

OVERRIDE = {
    # 'sel':   '/content/drive/MyDrive/.../vad_selection.tsv',
    # 'proto': '/content/drive/MyDrive/.../ASVspoof5.eval.track_1.tsv',
}

SCORES = {
    'xlsr':   ('xlsr_eval_vad_trimmed.tsv', 'xlsr_eval_vad_untrimmed.tsv'),
    'aasist': ('aasist_eval_vad_trimmed.tsv', 'aasist_eval.tsv'),
}

PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
CODECS = ['C04', 'C07', 'C08', 'C10']
BOUND_COLS = ['lead_sil_s', 'trail_sil_s']
B_BOOT = 1000
SEED = 258


def find_files(root=None):
    """Walk the Drive for what we need. followlinks=True because a shared
    folder appears as a symlink into .shortcut-targets-by-id and os.walk will
    not descend into it otherwise."""
    root = root or DRIVE
    trimmed_name, untrimmed_name = SCORES[MODEL]
    want = {'sel': 'vad_selection.tsv', 'trimlog': 'vad_trim_log.tsv',
            'trimmed': trimmed_name, 'untrimmed': untrimmed_name,
            'proto': 'ASVspoof5.eval.track_1.tsv'}
    print('FINDING FILES')
    print('=============')
    found = {k: v for k, v in OVERRIDE.items() if v}
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for k, n in want.items():
            if k not in found and n in filenames:
                found[k] = os.path.join(dirpath, n)
    for k in want:
        print(f'  {k:<10} {found.get(k, "NOT FOUND")}')
    missing = [k for k in want if k not in found]
    if missing:
        raise SystemExit(f'\n  Required file(s) not found: {missing}\n'
                         f'  Fill in OVERRIDE at the top and rerun.')
    return found


def score_col(path):
    head = pd.read_csv(path, sep='\t', nrows=5)
    for c in ('cm-score', 'score', 'cm_score', 'llr', 'margin'):
        if c in head.columns:
            return c
    numeric = [c for c in head.columns if pd.api.types.is_numeric_dtype(head[c])]
    if len(numeric) == 1:
        return numeric[0]
    raise SystemExit(f'  Cannot tell which column is the score in {path}')


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


def matched_seeds(log, tol):
    """Seeds whose VAD boundaries agree with the uncoded original, within tol
    seconds, under EVERY condition."""
    ok = None
    for col in BOUND_COLS:
        p = log.pivot_table(index='seed', columns='condition', values=col, aggfunc='first')
        if 'uncoded' not in p.columns:
            raise SystemExit('  trim log has no uncoded condition')
        d = p.sub(p['uncoded'], axis=0).drop(columns='uncoded').abs()
        good = (d <= tol).all(axis=1) & d.notna().all(axis=1)
        ok = good if ok is None else (ok & good)
    return set(ok[ok].index)


def load(paths):
    print('\n\nLOADING')
    print('=======')
    sel = pd.read_csv(paths['sel'], sep='\t', dtype=str)
    proto = pd.read_csv(paths['proto'], sep=r'\s+', header=None,
                        names=PROTO_COLS, dtype=str)
    proto['filename'] = proto['FLAC_FILE_NAME']

    tcol, ucol = score_col(paths['trimmed']), score_col(paths['untrimmed'])
    tr = pd.read_csv(paths['trimmed'], sep='\t', dtype={'filename': str})
    tr = tr[['filename', tcol]].rename(columns={tcol: 'trimmed'})
    un = pd.read_csv(paths['untrimmed'], sep='\t', dtype={'filename': str})
    un = un[['filename', ucol]].rename(columns={ucol: 'untrimmed'})

    df = (sel[['seed', 'condition', 'filename', 'KEY']]
          .merge(proto[['filename', 'SPEAKER_ID']], on='filename')
          .merge(un, on='filename').merge(tr, on='filename'))
    print(f'  model          {MODEL}')
    print(f'  joined         {len(df):>7,} rows   '
          f'{df.seed.nunique():,} originals, {df.SPEAKER_ID.nunique()} speakers')

    log = pd.read_csv(paths['trimlog'], sep='\t')
    return df, log


def boundary_report(log, df):
    print('\n\nHOW BADLY THE BOUNDARIES DISAGREE')
    print('==================================')
    for col in BOUND_COLS:
        p = log.pivot_table(index='seed', columns='condition', values=col, aggfunc='first')
        d = p.sub(p['uncoded'], axis=0).drop(columns='uncoded')
        t = pd.DataFrame({'median': d.median(), 'mean': d.mean(),
                          'p95_abs': d.abs().quantile(.95)}).round(4)
        print(f'\n  {col}, difference from uncoded (seconds)')
        print('    ' + t.to_string().replace('\n', '\n    '))

    print('\n\n  seeds surviving each tolerance (all conditions, both edges)')
    scored = set(df.seed.unique())
    for tol in REPORT_TOLS:
        keep = matched_seeds(log, tol) & scored
        print(f'    +/- {tol * 1000:>5.0f} ms   {len(keep):>6,} of {len(scored):,} '
              f'scored originals  ({100 * len(keep) / max(len(scored), 1):.1f}%)')


def _packs(df, speakers):
    idx = {s: i for i, s in enumerate(speakers)}
    out = {}
    for cond in ['uncoded'] + CODECS:
        sub = df[df.condition == cond]
        for state in ['untrimmed', 'trimmed']:
            tar = [[] for _ in speakers]
            non = [[] for _ in speakers]
            for spk, key, sc in zip(sub.SPEAKER_ID.to_numpy(), sub.KEY.to_numpy(),
                                    sub[state].to_numpy()):
                i = idx.get(spk)
                if i is not None:
                    (tar if key == 'bonafide' else non)[i].append(sc)
            out[(cond, state)] = ([np.asarray(a, float) for a in tar],
                                  [np.asarray(a, float) for a in non])
    return out


def analyse(df, label):
    print(f'\n\n{label}')
    print('=' * len(label))
    print(f'  {df.seed.nunique():,} originals, {len(df):,} files, '
          f'{df.SPEAKER_ID.nunique()} speakers, B={B_BOOT}\n')
    if df.seed.nunique() < 200:
        print('  [warn] small subset -- intervals will be wide; treat as indicative\n')

    speakers = sorted(df.SPEAKER_ID.unique())
    packs = _packs(df, speakers)

    def eer_of(cond, state, pick=None):
        t, n_ = packs[(cond, state)]
        if pick is None:
            return _eer(np.concatenate(t), np.concatenate(n_))
        return _eer(np.concatenate([t[i] for i in pick]),
                    np.concatenate([n_[i] for i in pick]))

    base_un, base_tr = eer_of('uncoded', 'untrimmed'), eer_of('uncoded', 'trimmed')
    print(f'  uncoded baseline   untrimmed {base_un:6.2f}%   trimmed {base_tr:6.2f}%\n')

    rng = np.random.default_rng(SEED)
    did = {c: [] for c in CODECS}
    for _ in range(B_BOOT):
        pick = rng.integers(0, len(speakers), len(speakers))
        bu, bt = eer_of('uncoded', 'untrimmed', pick), eer_of('uncoded', 'trimmed', pick)
        for c in CODECS:
            d = (eer_of(c, 'trimmed', pick) - bt) - (eer_of(c, 'untrimmed', pick) - bu)
            if np.isfinite(d):
                did[c].append(d)

    rows = []
    for c in CODECS:
        eu, et = eer_of(c, 'untrimmed') - base_un, eer_of(c, 'trimmed') - base_tr
        lo, hi = np.percentile(did[c], [2.5, 97.5])
        rows.append({'codec': c, 'effect_untrimmed': round(eu, 2),
                     'effect_trimmed': round(et, 2), 'DiD': round(et - eu, 2),
                     'lo': round(lo, 2), 'hi': round(hi, 2),
                     'verdict': 'survives' if lo <= 0 <= hi else 'CHANGED'})
    tab = pd.DataFrame(rows)

    print(f'  {"codec":<6} {"effect":>8} {"effect":>8} {"change":>8} '
          f'{"95% CI on change":>20}  verdict')
    print(f'  {"":<6} {"untrim":>8} {"trimmed":>8} {"(DiD)":>8}')
    for _, r in tab.iterrows():
        print(f'  {r.codec:<6} {r.effect_untrimmed:>8.2f} {r.effect_trimmed:>8.2f} '
              f'{r.DiD:>8.2f} {f"[{r.lo:.2f}, {r.hi:.2f}]":>20}  {r.verdict}')
    n = int((tab.verdict == 'survives').sum())
    print(f'\n  {n} of {len(tab)} show no detectable change under trimming.')
    return tab


def main(out_dir=None):
    out_dir = out_dir or DRIVE
    paths = find_files()
    df, log = load(paths)
    boundary_report(log, df)

    keep = matched_seeds(log, TOL_S)
    sub = df[df.seed.isin(keep)]
    if sub.seed.nunique() < 50:
        print(f'\n\n  STOPPING: only {sub.seed.nunique()} originals match within '
              f'+/- {TOL_S * 1000:.0f} ms.')
        print('  Raise TOL_S to a tolerance the table above shows is workable, or')
        print('  accept that the control cannot be rescued from the existing scores')
        print('  and has to be rerun with shared boundaries.')
        return None
    matched = analyse(sub, f'BOUNDARY-MATCHED SUBSET (+/- {TOL_S * 1000:.0f} ms)')
    allrows = analyse(df, 'ALL SCORED ORIGINALS (confounded on C08/C10 -- for contrast)')

    print('\n\nREADING THIS')
    print('============')
    print('  C04 and C07 are a valid test in both tables: their boundaries already')
    print('  agree with uncoded. Compare the two tables on C08 and C10. If the')
    print('  change shrinks toward zero on the matched subset, the positive DiD in')
    print('  the full table was the VAD cutting extra content, not the codec.')
    print('  If it holds up at similar size, the effect is real and the boundary')
    print('  disagreement was not driving it.')
    print('\n  Either way the matched subset is selected, not random: report it as')
    print('  evidence about direction, not as the effect-size estimate.')

    matched['subset'] = f'matched_{int(TOL_S * 1000)}ms'
    allrows['subset'] = 'all_scored'
    out = pd.concat([matched, allrows])
    out['model'] = MODEL
    out.to_csv(f'{out_dir}/258_vad_matched_{MODEL}.csv', index=False)
    print(f'\n  saved: {out_dir}/258_vad_matched_{MODEL}.csv')
    print('\n\nDONE. Paste the output back.')
    return out


if __name__ == '__main__':
    vad_matched = main()
