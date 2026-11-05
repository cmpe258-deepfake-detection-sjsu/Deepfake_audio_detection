"""
CMPE 258 - VAD control: paired bootstrap on the trimmed/untrimmed comparison
============================================================================

Shriya's summary gives point estimates. The question the control actually asks
is whether TRIMMING CHANGED THE CODEC EFFECT, so the statistic is a
difference-in-differences:

    effect_untrimmed = EER(codec, untrimmed) - EER(uncoded, untrimmed)
    effect_trimmed   = EER(codec, trimmed)   - EER(uncoded, trimmed)
    DiD              = effect_trimmed - effect_untrimmed

A CI on DiD that includes zero means trimming did not detectably change the
codec effect, i.e. the effect is not an artefact of leading/trailing silence.
A CI that excludes zero and sits well below it would mean the effect was partly
silence-driven.

All four EERs are recomputed on the SAME speaker draw, because the conditions
share source utterances and ignoring that correlation would inflate the
interval badly.

Runs both the full selection and the 93-original exclusion Shriya used, so the
two are directly comparable.

Read-only, no audio, no GPU. A few minutes.
"""

import os

import numpy as np
import pandas as pd

SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'

SEL = f'{SHRIYA_DIR}/vad/vad_selection.tsv'
TRIMLOG = f'{SHRIYA_DIR}/vad/vad_trim_log.tsv'
TRIMMED = f'{SHRIYA_DIR}/scores/aasist_eval_vad_trimmed.tsv'
UNTRIMMED = f'{SHRIYA_DIR}/scores/aasist_eval.tsv'
PROTO = f'{SHRIYA_DIR}/keys/ASVspoof5.eval.track_1.tsv'

PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
CODECS = ['C04', 'C07', 'C08', 'C10']
B_BOOT = 1000          # cheap here: 25k rows, not 680k
SEED = 258


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


def load():
    print('LOADING')
    print('=======')
    sel = pd.read_csv(SEL, sep='\t', dtype=str)
    proto = pd.read_csv(PROTO, sep=r'\s+', header=None, names=PROTO_COLS, dtype=str)
    proto['filename'] = proto['FLAC_FILE_NAME']

    tr = pd.read_csv(TRIMMED, sep='\t', dtype={'filename': str})[['filename', 'cm-score']]
    tr = tr.rename(columns={'cm-score': 'trimmed'})
    un = pd.read_csv(UNTRIMMED, sep='\t', dtype={'filename': str})[['filename', 'cm-score']]
    un = un.rename(columns={'cm-score': 'untrimmed'})

    df = (sel[['seed', 'condition', 'filename', 'KEY']]
          .merge(proto[['filename', 'SPEAKER_ID']], on='filename')
          .merge(un, on='filename')
          .merge(tr, on='filename'))
    print(f'  selection      {len(sel):>7,} rows')
    print(f'  joined         {len(df):>7,} rows   '
          f'{df.seed.nunique():,} originals, {df.SPEAKER_ID.nunique()} speakers')
    print(f'  conditions     {sorted(df.condition.unique())}')
    if len(df) != len(sel):
        print(f'  [warn] {len(sel) - len(df):,} selected files did not join')

    log = pd.read_csv(TRIMLOG, sep='\t', dtype=str)
    no_speech = set(log.loc[log.speech_found.astype(str).str.lower() == 'false', 'seed'])
    print(f'  originals with any no-speech file: {len(no_speech)}')
    return df, no_speech


def _packs(df, speakers):
    """Pre-split every (condition, trim-state) into per-speaker score arrays,
    so one speaker draw can be applied to all of them at once."""
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
                if i is None:
                    continue
                (tar if key == 'bonafide' else non)[i].append(sc)
            out[(cond, state)] = ([np.asarray(a, float) for a in tar],
                                  [np.asarray(a, float) for a in non])
    return out


def analyse(df, label):
    print(f'\n\n{label}')
    print('=' * len(label))
    print(f'  {df.seed.nunique():,} originals, {len(df):,} files, '
          f'{df.SPEAKER_ID.nunique()} speakers, bootstrap B={B_BOOT}\n')

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

    # One pass over draws, baselines computed once per draw rather than once
    # per codec: the same resampled speakers feed every condition.
    rng = np.random.default_rng(SEED)
    did = {c: [] for c in CODECS}
    for _ in range(B_BOOT):
        pick = rng.integers(0, len(speakers), len(speakers))
        bu = eer_of('uncoded', 'untrimmed', pick)
        bt = eer_of('uncoded', 'trimmed', pick)
        for c in CODECS:
            d = (eer_of(c, 'trimmed', pick) - bt) - (eer_of(c, 'untrimmed', pick) - bu)
            if np.isfinite(d):
                did[c].append(d)

    rows = []
    for c in CODECS:
        eu = eer_of(c, 'untrimmed') - base_un
        et = eer_of(c, 'trimmed') - base_tr
        lo, hi = np.percentile(did[c], [2.5, 97.5])
        survives = lo <= 0 <= hi
        rows.append({'codec': c,
                     'effect_untrimmed': round(eu, 2), 'effect_trimmed': round(et, 2),
                     'DiD': round(et - eu, 2), 'lo': round(lo, 2), 'hi': round(hi, 2),
                     'pct_of_effect': round(100 * (et - eu) / eu, 1) if eu else np.nan,
                     'verdict': 'survives' if survives else 'CHANGED'})
    tab = pd.DataFrame(rows)

    print(f'  {"codec":<6} {"effect":>8} {"effect":>8} {"change":>8} {"95% CI on change":>20} '
          f'{"% of":>7}  verdict')
    print(f'  {"":<6} {"untrim":>8} {"trimmed":>8} {"(DiD)":>8} {"":>20} {"effect":>7}')
    for _, r in tab.iterrows():
        print(f'  {r.codec:<6} {r.effect_untrimmed:>8.2f} {r.effect_trimmed:>8.2f} '
              f'{r.DiD:>8.2f} {f"[{r.lo:.2f}, {r.hi:.2f}]":>20} {r.pct_of_effect:>6.1f}%  '
              f'{r.verdict}')

    n_sur = int((tab.verdict == 'survives').sum())
    print(f'\n  {n_sur} of {len(tab)} codec effects show no detectable change under trimming.')
    if n_sur == len(tab):
        print('  The pre-registered falsification condition is not triggered: the codec')
        print('  effect is not explained by leading/trailing silence.')
    return tab


def class_shift(df):
    """What did trimming do to the scores themselves, per class? The codec effect
    is bona-fide-driven, so it matters that trimming did not move one class
    selectively."""
    print('\n\nWHAT TRIMMING DID TO THE SCORES')
    print('================================')
    print('  median (trimmed - untrimmed), per condition and class\n')
    df = df.copy()
    df['delta'] = df.trimmed - df.untrimmed
    piv = df.pivot_table(index='condition', columns='KEY', values='delta',
                         aggfunc='median').round(3)
    piv = piv.reindex(['uncoded'] + CODECS)
    print(piv.to_string())
    print('\n  Near-zero and similar across classes means trimming was close to a')
    print('  no-op, which is what the small amounts of edge silence imply.')
    return piv


def main(out_dir='/content/drive/MyDrive'):
    df, no_speech = load()
    full = analyse(df, f'ALL {df.seed.nunique():,} ORIGINALS')
    excl = analyse(df[~df.seed.isin(no_speech)],
                   f'EXCLUDING {len(no_speech)} ORIGINALS WITH ANY NO-SPEECH FILE')
    class_shift(df)

    full['subset'] = 'all'
    excl['subset'] = 'excl_nospeech'
    out = pd.concat([full, excl])
    out.to_csv(f'{out_dir}/258_vad_bootstrap.csv', index=False)
    print(f'\n  saved: {out_dir}/258_vad_bootstrap.csv')
    print('\n\nDONE. Paste the output back.')
    return out


if __name__ == '__main__':
    vad_boot = main()
