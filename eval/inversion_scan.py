"""
CMPE 258 - inversion scan on dev, both detectors
================================================

A19 (eval) and A12 (dev) both push AASIST above 50% EER. Above chance means
the detector is not merely weak, it is backwards: it scores those spoofs as
MORE bona fide than real speech. One case is an anecdote. A pattern across
attacks, present in one architecture and absent in the other, is the RQ2
result.

What this does:
  1. SCAN      AUC for every dev attack under both detectors. AUC = P(a bona
               fide file outscores a spoof file). Below 0.5 is inverted.
  2. DEEP DIVE on the worst inverted attack: score quantiles, and whether the
               inversion holds per speaker or is driven by a subset.
  3. FIGURE    the two detectors side by side on that attack.

Run after dev_stage_check.py in the same notebook (reuses nothing; it reloads).
No audio, no GPU.
"""

import glob
import os

import numpy as np
import pandas as pd

SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'
ZAHIRA_DIR = '/content/drive/MyDrive/XLS-R'     # add the trailing space if not renamed
PROTO_DIR = '/content/asvspoof5_protocols'

DETECTORS = {'AASIST': f'{SHRIYA_DIR}/scores/aasist_dev.tsv',
             'XLS-R':  f'{ZAHIRA_DIR}/scores/xlsr_dev.tsv'}

PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']

C_BONA, C_ATK = '#2a78d6', '#eb6834'      # validated categorical slots 1 and 2


def auc(pos, neg):
    """P(a random pos outscores a random neg), ties half."""
    from scipy.stats import rankdata
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def load():
    hits = glob.glob(os.path.join(PROTO_DIR, '**', 'ASVspoof5.dev.track_1.tsv'),
                     recursive=True)
    if not hits:
        raise FileNotFoundError('dev protocol missing - run dev_stage_check.py first')
    proto = pd.read_csv(hits[0], sep=r'\s+', header=None, names=PROTO_COLS, dtype=str)
    proto['filename'] = proto['FLAC_FILE_NAME']

    out = {}
    for name, path in DETECTORS.items():
        s = pd.read_csv(path, sep='\t', dtype={'filename': str})
        col = 'cm-score' if 'cm-score' in s.columns else s.columns[1]
        out[name] = s[['filename', col]].rename(columns={col: 'score'}).merge(
            proto[['filename', 'SPEAKER_ID', 'ATTACK_LABEL', 'KEY']], on='filename')
    return out


def scan(data):
    print('1. INVERSION SCAN - AUC per attack, both detectors')
    print('===================================================')
    print('  AUC = P(bona fide outscores the attack). < 0.5 means INVERTED.')
    rows = []
    for name, df in data.items():
        bona = df.loc[df.KEY == 'bonafide', 'score'].to_numpy()
        bona_med = float(np.median(bona))
        for a in sorted(df.loc[df.KEY == 'spoof', 'ATTACK_LABEL'].unique()):
            x = df.loc[(df.ATTACK_LABEL == a) & (df.KEY == 'spoof'), 'score'].to_numpy()
            rows.append({'detector': name, 'attack': a, 'n': len(x),
                         'AUC': round(auc(bona, x), 4),
                         'median_shift': round(float(np.median(x)) - bona_med, 2)})
    tab = pd.DataFrame(rows)

    wide = tab.pivot(index='attack', columns='detector', values='AUC')
    print()
    print(wide.to_string())

    inv = tab[tab.AUC < 0.5]
    print('\n  INVERTED (AUC below 0.5):')
    if len(inv):
        for _, r in inv.sort_values('AUC').iterrows():
            print(f'    {r.detector:<8} {r.attack}   AUC {r.AUC:.4f}   '
                  f'scores {r.median_shift:+.2f} vs bona fide median   n={r.n:,}')
    else:
        print('    none')

    near = tab[(tab.AUC >= 0.5) & (tab.AUC < 0.6)]
    if len(near):
        print('\n  NEAR CHANCE (0.5 - 0.6, no usable signal):')
        for _, r in near.sort_values('AUC').iterrows():
            print(f'    {r.detector:<8} {r.attack}   AUC {r.AUC:.4f}   n={r.n:,}')

    print('\n  Count of inverted attacks per detector:')
    for name in data:
        n = int(((tab.detector == name) & (tab.AUC < 0.5)).sum())
        total = int((tab.detector == name).sum())
        print(f'    {name:<8} {n} of {total}')
    print('\n  One architecture inverting where the other does not is a')
    print('  qualitative difference, not a difference of degree. That is the')
    print('  RQ2 claim worth making.')
    return tab


def deep_dive(data, attack):
    print(f'\n\n2. DEEP DIVE ON {attack}')
    print('=' * (18 + len(attack)))
    for name, df in data.items():
        bona = df.loc[df.KEY == 'bonafide', 'score'].to_numpy()
        x = df.loc[(df.ATTACK_LABEL == attack) & (df.KEY == 'spoof'), 'score'].to_numpy()
        if not len(x):
            continue
        a = auc(bona, x)
        print(f'\n  {name}   AUC {a:.4f}'
              + ('   INVERTED' if a < 0.5 else ''))
        for lab, v in [('bona fide', bona), (attack, x)]:
            q = np.percentile(v, [5, 25, 50, 75, 95])
            print(f'    {lab:<12} n={len(v):>7,}   p5 {q[0]:>7.2f}  p25 {q[1]:>7.2f}  '
                  f'median {q[2]:>7.2f}  p75 {q[3]:>7.2f}  p95 {q[4]:>7.2f}')

        # per speaker
        sub = df[(df.ATTACK_LABEL == attack) & (df.KEY == 'spoof')]
        bona_rows = df[df.KEY == 'bonafide']
        per = []
        for spk, grp in sub.groupby('SPEAKER_ID'):
            b = bona_rows.loc[bona_rows.SPEAKER_ID == spk, 'score'].to_numpy()
            if len(b) >= 3 and len(grp) >= 3:
                per.append(auc(b, grp['score'].to_numpy()))
        if per:
            per = np.array(per)
            print(f'    speakers checked {len(per)};  '
                  f'{(per < 0.5).mean():.1%} show the inversion;  '
                  f'median speaker AUC {np.median(per):.3f}')


def plot(data, attack, save_to=None):
    import matplotlib.pyplot as plt

    names = list(data.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(11, 4.1), dpi=140)
    fig.patch.set_facecolor('#fcfcfb')
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        df = data[name]
        bona = df.loc[df.KEY == 'bonafide', 'score'].to_numpy()
        x = df.loc[(df.ATTACK_LABEL == attack) & (df.KEY == 'spoof'), 'score'].to_numpy()
        lo, hi = np.percentile(np.concatenate([bona, x]), [0.3, 99.7])
        bins = np.linspace(lo, hi, 70)
        a = auc(bona, x)

        ax.set_facecolor('#fcfcfb')
        top = 0
        for lab, v, c in [('bona fide', bona, C_BONA), (attack, x, C_ATK)]:
            ax.hist(v, bins=bins, density=True, histtype='step', linewidth=2,
                    color=c, label=lab)
            h, _ = np.histogram(v, bins=bins, density=True)
            top = max(top, float(h.max()))
        ax.set_ylim(0, top * 1.22)
        ax.set_title(f'{name}   AUC {a:.3f}'
                     + ('   (inverted)' if a < 0.5 else ''),
                     fontsize=11, loc='left', color='#0b0b0b', pad=10)
        ax.set_xlabel('score  (higher = judged more real)', fontsize=9.5, color='#52514e')
        ax.grid(axis='y', color='#e5e4e0', linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#c3c2b7')
        ax.tick_params(colors='#52514e', labelsize=9)
        ax.legend(frameon=False, fontsize=9, loc='upper left')
    axes[0].set_ylabel('density', fontsize=9.5, color='#52514e')
    fig.suptitle(f'{attack} against real speech, both detectors, dev set',
                 fontsize=12.5, x=0.012, ha='left', color='#0b0b0b')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_to:
        fig.savefig(save_to, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'\n  figure saved: {save_to}')
    plt.show()


def main(attack='A12', save_to='/content/drive/MyDrive/258_fig_a12_inversion.png'):
    data = load()
    tab = scan(data)
    deep_dive(data, attack)
    plot(data, attack, save_to)
    print('\n\nDONE. Paste the output back.')
    return tab


if __name__ == '__main__':
    inversion_table = main()
