"""
CMPE 258 - is the A19 result real, and does the "unit selection is recombined
real speech" explanation hold?

Run in the same notebook after step 1. Needs `merged`.

EER above 50% says the detector is worse than chance on A19, but EER alone does
not say WHY. Three different things produce a high EER and only one of them
supports the explanation:

  (a) A19 scores sit ABOVE bona fide        -> systematic inversion, story holds
  (b) A19 scores overlap bona fide heavily  -> the detector is just blind to it
  (c) a subset of speakers or files drives  -> an artifact, not a property of A19

The decisive number is AUC = P(a bona fide file scores higher than an A19 file).
Above 0.5 the detector has the right idea; at 0.5 it is guessing; below 0.5 it
is actively inverted. The per-speaker breakdown separates (a) from (c).
"""

import numpy as np
import pandas as pd

UNCODED = '-'
ATTACK = 'ATTACK_LABEL'
CONTEXT = ['A19', 'A18', 'A26', 'A17', 'A29']   # worst -> best, from step 3

# validated categorical hues (slots 1-3), all-pairs clean in light mode
C_BONA, C_A19, C_EASY = '#2a78d6', '#eb6834', '#1baf7a'


def auc_pos_over_neg(pos, neg):
    """P(a random `pos` value exceeds a random `neg` value), ties counted half."""
    from scipy.stats import rankdata
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def describe(name, x):
    q = np.percentile(x, [5, 25, 50, 75, 95])
    print(f'  {name:<22} n={len(x):>7,}   '
          f'p5 {q[0]:>7.2f}  p25 {q[1]:>7.2f}  median {q[2]:>7.2f}  '
          f'p75 {q[3]:>7.2f}  p95 {q[4]:>7.2f}')


def main(m, make_plot=True, save_to=None):
    unc = m[m.CODEC == UNCODED]
    bona = unc.loc[unc.KEY == 'bonafide', 'cm-score'].to_numpy()

    print('SCORE DISTRIBUTIONS, UNCODED  (higher = detector thinks more real)')
    print('===================================================================')
    describe('bona fide', bona)
    for a in CONTEXT:
        x = unc.loc[(unc[ATTACK] == a) & (unc.KEY == 'spoof'), 'cm-score'].to_numpy()
        if len(x):
            describe(f'{a} (spoof)', x)

    print('\n\nAUC: P(bona fide scores above the attack)')
    print('=========================================')
    print('  > 0.5 detector separates correctly | = 0.5 chance | < 0.5 INVERTED')
    rows = []
    for a in CONTEXT:
        x = unc.loc[(unc[ATTACK] == a) & (unc.KEY == 'spoof'), 'cm-score'].to_numpy()
        if not len(x):
            continue
        a_auc = auc_pos_over_neg(bona, x)
        shift = float(np.median(x) - np.median(bona))
        verdict = ('INVERTED' if a_auc < 0.45 else
                   'near chance' if a_auc < 0.55 else 'correct')
        rows.append({'attack': a, 'AUC': round(a_auc, 4),
                     'median_shift_vs_bonafide': round(shift, 3),
                     'verdict': verdict})
    print(pd.DataFrame(rows).to_string(index=False))
    print('\n  median_shift > 0 means the attack scores HIGHER than real speech.')

    print('\n\nIS IT EVERY SPEAKER, OR A FEW?')
    print('===============================')
    a19 = unc[(unc[ATTACK] == 'A19') & (unc.KEY == 'spoof')]
    bona_rows = unc[unc.KEY == 'bonafide']
    per = []
    for spk, grp in a19.groupby('SPEAKER_ID'):
        b = bona_rows.loc[bona_rows.SPEAKER_ID == spk, 'cm-score'].to_numpy()
        if len(b) < 3 or len(grp) < 3:
            continue
        per.append({'speaker': spk,
                    'auc': auc_pos_over_neg(b, grp['cm-score'].to_numpy()),
                    'n_bona': len(b), 'n_a19': len(grp)})
    per = pd.DataFrame(per)
    if len(per):
        inverted = (per.auc < 0.5).mean()
        print(f'  speakers with enough files     {len(per)}')
        print(f'  speakers where A19 outscores   {inverted:.1%}')
        print(f'  per-speaker AUC  median {per.auc.median():.3f}   '
              f'p10 {per.auc.quantile(.10):.3f}   p90 {per.auc.quantile(.90):.3f}')
        if inverted > 0.9:
            print('  -> systematic across speakers. Not an artifact of a few files.')
        elif inverted > 0.6:
            print('  -> mostly systematic, but check the speakers that disagree.')
        else:
            print('  -> NOT systematic. A subset drives it; do not claim a')
            print('     property of unit selection until you know which files.')

    print('\n\nDOES IT SURVIVE CODECS?')
    print('========================')
    print('  If A19 stays inverted once coded, the cue is in the speech itself,')
    print('  not in a clean-recording artifact that compression destroys.')
    rows = []
    for c in sorted(m.CODEC.unique()):
        sub = m[m.CODEC == c]
        b = sub.loc[sub.KEY == 'bonafide', 'cm-score'].to_numpy()
        x = sub.loc[(sub[ATTACK] == 'A19') & (sub.KEY == 'spoof'), 'cm-score'].to_numpy()
        if len(b) and len(x):
            rows.append({'codec': c, 'n_a19': len(x),
                         'AUC': round(auc_pos_over_neg(b, x), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    if make_plot:
        plot(unc, bona, save_to)

    return per


def plot(unc, bona, save_to=None):
    import matplotlib.pyplot as plt

    a19 = unc.loc[(unc[ATTACK] == 'A19') & (unc.KEY == 'spoof'), 'cm-score'].to_numpy()
    easy = unc.loc[(unc[ATTACK] == 'A29') & (unc.KEY == 'spoof'), 'cm-score'].to_numpy()

    lo, hi = np.percentile(np.concatenate([bona, a19, easy]), [0.2, 99.8])
    bins = np.linspace(lo, hi, 90)

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')

    series = [('Bona fide', bona, C_BONA),
              ('A19  unit selection', a19, C_A19),
              ('A29  easiest attack', easy, C_EASY)]
    peaks, top = [], 0.0
    for label, x, color in series:
        ax.hist(x, bins=bins, density=True, histtype='step',
                linewidth=2, color=color, label=label)
        h, e = np.histogram(x, bins=bins, density=True)
        peaks.append((label, float(e[np.argmax(h)]), color))
        top = max(top, float(h.max()))

    # Direct labels sit in a band above the curves, never on them (relief rule).
    # Stagger any two whose x positions are close enough to collide.
    ax.set_ylim(0, top * 1.26)
    span = bins[-1] - bins[0]
    placed = []
    for label, x, color in sorted(peaks, key=lambda p: p[1]):
        level = 0
        while any(abs(x - px) < 0.22 * span and lv == level for px, lv in placed):
            level += 1
        placed.append((x, level))
        ax.annotate(label, xy=(x, top * (1.19 - 0.08 * level)),
                    ha='center', va='bottom', fontsize=9,
                    color=color, fontweight='600')

    ax.set_xlabel('AASIST score  (bona fide logit - higher = judged more real)',
                  fontsize=10, color='#52514e')
    ax.set_ylabel('density', fontsize=10, color='#52514e')
    # Neutral title: rewrite it to state the finding once the numbers confirm it.
    ax.set_title('AASIST score distributions, uncoded: real speech vs. two attacks',
                 fontsize=12, color='#0b0b0b', loc='left', pad=14)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.grid(axis='y', color='#e5e4e0', linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e', labelsize=9)
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'\n  figure saved: {save_to}')
    plt.show()


if __name__ == '__main__':
    try:
        per_speaker = main(merged,  # noqa: F821
                           save_to='/content/drive/MyDrive/258_fig_a19_scores.png')
    except NameError:
        print('Run step 1 first in this notebook - this script needs `merged`.')
