"""
CMPE 258 - shift-decomposed evaluation harness
STEP 1: load, validate, explore.

Paste this into a Colab cell (with Shriya's folder mounted) and run it.
It touches no audio and needs no GPU. Runtime is a couple of minutes.

What it does:
  1. Loads the eval protocol and the AASIST scores from Shriya's Drive folder.
  2. Validates them: row counts, join coverage, duplicates, score convention.
  3. Implements EER in numpy and checks it against Shriya's official-package
     numbers. This matters: the bootstrap needs ~1000 EER calls per slice, and
     the official package is too slow for that. Once our fast EER reproduces
     her 28.11% exactly, we can use it everywhere and still claim official
     metrics.
  4. Prints what is actually in the protocol - codec values, attack tags, the
     attack x codec grid - so the slice definitions come from the data rather
     than from the proposal.

Nothing here is written back to Drive. It is read-only.
"""

import json
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths. SHRIYA_DIR is her shared folder; in Colab, "Shared with me" folders
# need a shortcut added to your own Drive before they appear under MyDrive.
# Right-click the folder in Drive -> Organize -> Add shortcut -> My Drive.
# --------------------------------------------------------------------------
SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'

PROTO_EVAL = f'{SHRIYA_DIR}/keys/ASVspoof5.eval.track_1.tsv'
SCORES_EVAL = f'{SHRIYA_DIR}/scores/aasist_eval.tsv'
SCORES_DEV = f'{SHRIYA_DIR}/scores/aasist_dev.tsv'
SCORES_DFADD = f'{SHRIYA_DIR}/scores/aasist_dfadd.tsv'
SCORES_REF = f'{SHRIYA_DIR}/scores/aasist_train_heldout_ref.tsv'
METRICS_JSON = f'{SHRIYA_DIR}/metrics/aasist_metrics.json'

# Shriya's protocol column names and read settings - keep identical to hers.
PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']

EXPECTED_EVAL_ROWS = 680774
EXPECTED_EVAL_BONAFIDE = 138688


# ==========================================================================
# EER, the standard ASVspoof implementation. Scores are bona fide logits,
# so higher means more likely real: bona fide are the target class.
# ==========================================================================
def det_curve(target_scores, nontarget_scores):
    n = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    order = np.argsort(all_scores, kind='mergesort')
    labels = labels[order]
    tar_sums = np.cumsum(labels)
    non_sums = nontarget_scores.size - (np.arange(1, n + 1) - tar_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), non_sums / nontarget_scores.size))
    thr = np.concatenate((np.atleast_1d(all_scores[order[0]] - 0.001), all_scores[order]))
    return frr, far, thr


def compute_eer(target_scores, nontarget_scores):
    """Returns EER as a fraction (multiply by 100 for a percentage)."""
    if target_scores.size == 0 or nontarget_scores.size == 0:
        return np.nan
    frr, far, _ = det_curve(target_scores, nontarget_scores)
    i = np.argmin(np.abs(frr - far))
    return float(np.mean((frr[i], far[i])))


def eer_from_frame(df, score_col='cm-score', key_col='KEY'):
    tar = df.loc[df[key_col] == 'bonafide', score_col].to_numpy()
    non = df.loc[df[key_col] == 'spoof', score_col].to_numpy()
    return compute_eer(tar, non)


# ==========================================================================
def load():
    print('LOADING')
    print('-------')
    proto = pd.read_csv(PROTO_EVAL, sep=r'\s+', header=None,
                        names=PROTO_COLS, dtype=str)
    proto['filename'] = proto['FLAC_FILE_NAME']
    print(f'  protocol   {len(proto):>8,} rows')

    scores = pd.read_csv(SCORES_EVAL, sep='\t', dtype={'filename': str})
    print(f'  scores     {len(scores):>8,} rows   columns: {list(scores.columns)}')
    return proto, scores


def validate(proto, scores):
    print('\nVALIDATION')
    print('----------')
    ok = True

    def check(label, condition, detail=''):
        nonlocal ok
        mark = 'PASS' if condition else 'FAIL'
        if not condition:
            ok = False
        print(f'  [{mark}] {label}{("  " + detail) if detail else ""}')

    check('protocol row count', len(proto) == EXPECTED_EVAL_ROWS,
          f'{len(proto):,} (expected {EXPECTED_EVAL_ROWS:,})')
    check('score row count', len(scores) == EXPECTED_EVAL_ROWS,
          f'{len(scores):,}')
    check('no duplicate filenames in scores',
          not scores.filename.duplicated().any(),
          f'{int(scores.filename.duplicated().sum())} duplicates')
    check('no duplicate filenames in protocol',
          not proto.filename.duplicated().any(),
          f'{int(proto.filename.duplicated().sum())} duplicates')

    # The claim in Shriya's README: cm-score IS the bona fide logit.
    if 'logit_bonafide' in scores.columns:
        same = np.allclose(scores['cm-score'], scores['logit_bonafide'])
        check('cm-score == logit_bonafide', same,
              'convention confirmed' if same else 'THESE DIFFER - stop and check')

    merged = scores.merge(proto[['filename', 'KEY', 'SPEAKER_ID', 'CODEC',
                                 'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL']],
                          on='filename', how='inner')
    coverage = len(merged) / len(scores) if len(scores) else 0
    check('join coverage is 100%', len(merged) == len(scores),
          f'{coverage:.4%} ({len(merged):,} of {len(scores):,})')
    check('bona fide count', int((merged.KEY == 'bonafide').sum()) == EXPECTED_EVAL_BONAFIDE,
          f'{int((merged.KEY == "bonafide").sum()):,}')
    check('scores are finite', np.isfinite(merged['cm-score']).all())

    return merged, ok


def self_check(merged):
    """Our numpy EER must reproduce Shriya's official-package number."""
    print('\nSELF-CHECK  (our EER vs. the official evaluation package)')
    print('---------------------------------------------------------')
    ours = 100 * eer_from_frame(merged)
    print(f'  our EER, eval pooled      {ours:6.2f}%')

    if os.path.exists(METRICS_JSON):
        official = json.load(open(METRICS_JSON))
        theirs = official.get('eval_pooled', {}).get('EER_%')
        if theirs is not None:
            print(f'  Shriya, official package  {theirs:6.2f}%')
            delta = abs(ours - theirs)
            if delta < 0.02:
                print('  [PASS] match. Use the fast EER for the bootstrap.')
            elif delta < 0.2:
                print(f'  [WARN] differ by {delta:.3f} points - small enough to be')
                print('         threshold-grid granularity, but report the official')
                print('         package numbers in the paper, not these.')
            else:
                print(f'  [FAIL] differ by {delta:.3f} points. Something is wrong with')
                print('         the join or the score column. Resolve before slicing.')
        else:
            print('  (eval_pooled not in the metrics file)')
    else:
        print('  (metrics file not found; skipping comparison)')
    return ours


def explore(merged):
    print('\nWHAT IS ACTUALLY IN THE EVAL PROTOCOL')
    print('------------------------------------')
    print(f'  speakers          {merged.SPEAKER_ID.nunique()}')
    print(f'  codec values      {sorted(merged.CODEC.unique())}')
    print(f'  attack labels     {sorted(merged.ATTACK_LABEL.unique())}')
    n_tags = merged.ATTACK_TAG.nunique()
    print(f'  attack tags       {n_tags} distinct')

    print('\n  files per codec')
    for codec, n in merged.CODEC.value_counts().sort_index().items():
        bona = int((merged.loc[merged.CODEC == codec, 'KEY'] == 'bonafide').sum())
        print(f'    {codec:<12} {n:>8,}   ({bona:,} bona fide)')

    print('\n  files per attack (spoof only, top 40 by count)')
    spoof = merged[merged.KEY == 'spoof']
    for tag, n in spoof.ATTACK_TAG.value_counts().head(40).items():
        print(f'    {tag:<12} {n:>8,}')

    print('\n  attack x codec grid (spoof only, first 12 attacks)')
    grid = pd.crosstab(spoof.ATTACK_TAG, spoof.CODEC)
    print(grid.head(12).to_string())

    print('\n  CODEC_SEED - this is what pairs a coded utterance with its C00 original')
    print(f'    distinct values   {merged.CODEC_SEED.nunique():,}')
    counts = merged.CODEC_SEED.value_counts()
    print(f'    files per seed    min {counts.min()}, median {int(counts.median())}, max {counts.max()}')
    print('    (a seed shared across codec conditions = the same source utterance)')


def per_attack_table(merged, top=15):
    """EER per attack, each against the same bona fide pool. A first look at
    which generators hurt, before slices are formally defined."""
    if 'C00' not in set(merged.CODEC):
        print('\n  (no C00 in this protocol - codec labels are '
              f'{sorted(merged.CODEC.unique())[:6]}; skipping the per-attack table)')
        return None
    print('\n  EER PER ATTACK  (vs. all bona fide at C00, uncoded)')
    c00 = merged[merged.CODEC == 'C00']
    bona = c00[c00.KEY == 'bonafide']
    rows = []
    for tag in sorted(c00.loc[c00.KEY == 'spoof', 'ATTACK_TAG'].unique()):
        sub = pd.concat([bona, c00[(c00.ATTACK_TAG == tag) & (c00.KEY == 'spoof')]])
        rows.append({'attack': tag, 'n_spoof': int((sub.KEY == 'spoof').sum()),
                     'EER_%': round(100 * eer_from_frame(sub), 2)})
    out = pd.DataFrame(rows).sort_values('EER_%', ascending=False)
    print(out.head(top).to_string(index=False))
    return out


def main():
    proto, scores = load()
    merged, ok = validate(proto, scores)
    if not ok:
        print('\nSome checks failed. Fix those before building slices.')
    self_check(merged)
    explore(merged)
    per_attack_table(merged)
    print('\nDONE. Paste this whole output back and we build the slices from it.')
    return merged


if __name__ == '__main__':
    merged = main()
