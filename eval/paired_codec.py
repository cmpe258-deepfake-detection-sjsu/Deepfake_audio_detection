"""
CMPE 258 - paired per-utterance codec analysis
==============================================

The decomposition shows C04 (Encodec) and C07 (mp3+Encodec) are XLS-R's worst
codecs, and C08/C09/C10 (8 kHz DSP) are AASIST's. It does not show WHY, because
EER conflates two different failures:

    spoof scores rise      -> the codec makes fakes look more real
    bona fide scores fall  -> the codec makes real speech look fake

Those have opposite mechanistic readings. The claim about neural codecs is
specifically the second one: Encodec re-synthesises its input through a learned
decoder, so genuine speech comes out carrying generative signatures and the SSL
representation stops being able to tell the classes apart.

`CODEC_SEED` names the uncoded source file, so every coded file can be paired
with its own original. This measures, per utterance:

    delta = score(coded) - score(uncoded original)

split by class. A codec that works by dragging bona fide down shows a large
negative bona fide delta and a small spoof delta. One that works by lifting
spoofs shows the reverse.

Paired deltas are also a much tighter test than comparing two EERs, because
speaker and utterance are held constant by construction.

Run after both_detectors.py in the same notebook. Read-only, no audio, no GPU.
"""

import os

import numpy as np
import pandas as pd

SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'
ZAHIRA_DIR = '/content/drive/MyDrive/XLS-R'

DETECTORS = {'AASIST': f'{SHRIYA_DIR}/scores/aasist_eval.tsv',
             'XLS-R': f'{ZAHIRA_DIR}/ssl_detector/scores/xlsr_eval.tsv'}
PROTO_EVAL = f'{SHRIYA_DIR}/keys/ASVspoof5.eval.track_1.tsv'

PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
UNCODED = '-'
SEED = 258
B_BOOT = 200

# From the ASVspoof 5 paper. Verify against Table 3 before this goes in the report.
CODEC_TYPE = {
    'C01': 'DSP 16k (opus)', 'C02': 'DSP 16k (amr)', 'C03': 'DSP 16k (speex)',
    'C04': 'NEURAL (Encodec)', 'C05': 'DSP 16k (mp3)', 'C06': 'DSP 16k (m4a)',
    'C07': 'NEURAL (mp3+Encodec)', 'C08': 'DSP 8k (opus)', 'C09': 'DSP 8k (amr)',
    'C10': 'DSP 8k (speex)', 'C11': 'mixed 8k',
}

C_BONA, C_SPOOF = '#2a78d6', '#eb6834'      # validated categorical slots 1 and 2

# Fixed row order, grouped by codec family, so the two panels line up and the
# pattern is visible across them rather than each panel sorting itself.
ROW_ORDER = ['C04', 'C07',                                  # neural
             'C01', 'C02', 'C03', 'C05', 'C06',             # 16 kHz DSP
             'C08', 'C09', 'C10', 'C11']                    # 8 kHz


def paired_frame():
    """One row per coded file, carrying its own uncoded original's score."""
    proto = pd.read_csv(PROTO_EVAL, sep=r'\s+', header=None, names=PROTO_COLS, dtype=str)
    proto['filename'] = proto['FLAC_FILE_NAME']

    out = {}
    for name, path in DETECTORS.items():
        raw = pd.read_csv(path, sep='\t', dtype={'filename': str})
        col = 'cm-score' if 'cm-score' in raw.columns else raw.columns[1]
        df = raw[['filename', col]].rename(columns={col: 'score'}).merge(
            proto[['filename', 'KEY', 'SPEAKER_ID', 'CODEC', 'CODEC_SEED', 'ATTACK_LABEL']],
            on='filename')

        src = (df[df.CODEC == UNCODED]
               .set_index('filename')[['score']]
               .rename(columns={'score': 'score_uncoded'}))
        coded = df[df.CODEC != UNCODED].join(src, on='CODEC_SEED', how='inner')
        coded['delta'] = coded.score - coded.score_uncoded
        out[name] = coded
        print(f'  {name:<8} {len(coded):>8,} coded files paired to their original '
              f'({len(coded) / max(1, len(df[df.CODEC != UNCODED])):.1%} of coded)')
    return out


def boot_median(x, spk, b=B_BOOT, seed=SEED):
    """95% CI on a median delta, resampling speakers."""
    uniq, inv = np.unique(spk, return_inverse=True)
    by = [x[inv == i] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        p = rng.integers(0, len(uniq), len(uniq))
        vals.append(float(np.median(np.concatenate([by[i] for i in p]))))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def table(data):
    print('\n\nMEDIAN SCORE CHANGE CAUSED BY EACH CODEC')
    print('=========================================')
    print('  delta = score(coded) - score(same utterance, uncoded)')
    print('  negative = the codec pushes scores DOWN (looks more fake)\n')
    rows = []
    for name, df in data.items():
        for c in sorted(df.CODEC.unique()):
            sub = df[df.CODEC == c]
            r = {'detector': name, 'codec': c, 'type': CODEC_TYPE.get(c, '?')}
            for cls, key in [('bona', 'bonafide'), ('spoof', 'spoof')]:
                s = sub[sub.KEY == key]
                if len(s):
                    med = float(np.median(s.delta))
                    lo, hi = boot_median(s.delta.to_numpy(), s.SPEAKER_ID.to_numpy())
                    r[f'{cls}_delta'] = round(med, 3)
                    r[f'{cls}_ci'] = f'[{lo:.2f},{hi:.2f}]'
                    r[f'{cls}_n'] = len(s)
            r['gap_closed'] = round(r.get('spoof_delta', 0) - r.get('bona_delta', 0), 3)
            rows.append(r)
    tab = pd.DataFrame(rows)

    for name in data:
        print(f'\n{name}')
        part = tab[tab.detector == name].sort_values('bona_delta')
        print(f'  {"codec":<6} {"type":<22} {"bona Δ":>9} {"CI":>16} '
              f'{"spoof Δ":>9} {"gap closed":>11}')
        for _, r in part.iterrows():
            print(f'  {r.codec:<6} {r["type"]:<22} {r.bona_delta:>9.3f} '
                  f'{r.bona_ci:>16} {r.spoof_delta:>9.3f} {r.gap_closed:>11.3f}')
    print('\n  "gap closed" = spoof delta minus bona fide delta. Positive means the')
    print('  codec moves the two classes toward each other, which is what raises EER.')
    print('  Look at WHICH side moves: a big negative bona fide delta with a small')
    print('  spoof delta means the codec makes real speech look fake, not fakes look real.')
    return tab


def verdict(tab):
    print('\n\nWHAT DRIVES EACH DETECTOR\'S WORST CODECS')
    print('=========================================')
    worst = {'AASIST': ['C08', 'C09', 'C10'], 'XLS-R': ['C04', 'C07']}
    for det, codecs in worst.items():
        part = tab[(tab.detector == det) & (tab.codec.isin(codecs))]
        if not len(part):
            continue
        b = part.bona_delta.mean()
        s = part.spoof_delta.mean()
        print(f'\n  {det}  (worst codecs: {", ".join(codecs)})')
        print(f'    mean bona fide delta  {b:+.3f}')
        print(f'    mean spoof delta      {s:+.3f}')
        if abs(b) > 2 * abs(s):
            print('    -> driven mainly by REAL SPEECH being pushed down.')
            print('       The codec makes genuine audio look synthetic.')
        elif abs(s) > 2 * abs(b):
            print('    -> driven mainly by SPOOFS being pushed up.')
            print('       The codec hides the artifacts the detector relies on.')
        else:
            print('    -> both classes move comparably; the codec compresses the')
            print('       whole score range rather than favouring one class.')


def plot(tab, save_to=None):
    import matplotlib.pyplot as plt
    dets = list(tab.detector.unique())
    fig, axes = plt.subplots(1, len(dets), figsize=(11.5, 5.0), dpi=140, sharex=False)
    fig.patch.set_facecolor('#fcfcfb')
    if len(dets) == 1:
        axes = [axes]

    for ax, det in zip(axes, dets):
        part = tab[tab.detector == det].copy()
        part['_o'] = part.codec.map({c: i for i, c in enumerate(ROW_ORDER)})
        part = part.sort_values('_o', ascending=False)
        y = np.arange(len(part))
        ax.set_facecolor('#fcfcfb')
        ax.axvline(0, color='#c3c2b7', linewidth=1.2, zorder=1)
        ax.scatter(part.bona_delta, y, s=46, color=C_BONA, zorder=3,
                   edgecolor='#fcfcfb', linewidth=1.2, label='bona fide')
        ax.scatter(part.spoof_delta, y, s=46, color=C_SPOOF, zorder=3,
                   edgecolor='#fcfcfb', linewidth=1.2, label='spoof')
        for yi, (_, r) in zip(y, part.iterrows()):
            ax.plot([r.bona_delta, r.spoof_delta], [yi, yi],
                    color='#d3d2cc', linewidth=1.4, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{r.codec}  {r["type"]}' for _, r in part.iterrows()],
                           fontsize=8.5)
        ax.set_title(det, fontsize=11.5, loc='left', color='#0b0b0b', pad=9)
        ax.set_xlabel('median score change vs. the same uncoded utterance',
                      fontsize=9.5, color='#52514e')
        ax.grid(axis='x', color='#e5e4e0', linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#c3c2b7')
        ax.tick_params(colors='#52514e', labelsize=9)
        if ax is axes[0]:                       # one legend for the figure
            ax.legend(frameon=False, fontsize=9, loc='upper left',
                      bbox_to_anchor=(0.0, -0.13), ncol=2)

    fig.suptitle('Which class each codec moves, per utterance',
                 fontsize=12.5, x=0.012, ha='left', color='#0b0b0b')
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    if save_to:
        fig.savefig(save_to, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'\n  figure saved: {save_to}')
    plt.show()


def main(save_fig='/content/drive/MyDrive/258_fig_codec_deltas.png',
         save_csv='/content/drive/MyDrive/258_codec_deltas.csv'):
    print('PAIRING')
    print('=======')
    data = paired_frame()
    tab = table(data)
    verdict(tab)
    plot(tab, save_fig)
    if save_csv:
        tab.to_csv(save_csv, index=False)
        print(f'  table saved: {save_csv}')
    print('\n\nDONE. Paste the output back.')
    return tab


if __name__ == '__main__':
    codec_delta_table = main()
