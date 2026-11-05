"""
CMPE 258 - matched contrasts, rebuilt
=====================================

Replaces the max-minus-min "spread" headline, which was sensitive to how many
conditions sat in each group and how they were pooled.

The three shift axes are not commensurable, so this stops pretending they are:

  CODEC        a genuine matched degradation. The same source utterance, coded
               vs uncoded. Reported as a paired difference with a speaker
               bootstrap on the difference itself.
  ADVERSARIAL  also matched. A18 and A30 are Malafide filters applied to A17,
               so A17 -> A18 holds generator, partition and codec fixed.
  GENERATOR    NOT a matched degradation. A seen-vs-unseen comparison would need
               A01-A08 as a baseline, which exists only in the training
               partition - contaminated for AASIST, partition-confounded for
               XLS-R. Reported as between-family VARIATION, never as shift.

Also here: all three XLS-R seeds beside the ensemble (so no seed is chosen
after seeing results), and the attack x codec heatmap.

Run in the same notebook. Read-only, no audio, no GPU. Allow a few minutes
for the paired bootstraps.
"""

import os

import numpy as np
import pandas as pd

SHRIYA_DIR = '/content/drive/MyDrive/shriya_aasist'
ZAHIRA_DIR = '/content/drive/MyDrive/XLS-R'
SSL_SCORES = f'{ZAHIRA_DIR}/ssl_detector/scores'
PROTO_EVAL = f'{SHRIYA_DIR}/keys/ASVspoof5.eval.track_1.tsv'

SYSTEMS = {
    'AASIST':        f'{SHRIYA_DIR}/scores/aasist_eval.tsv',
    'XLS-R (ens)':   f'{SSL_SCORES}/xlsr_eval.tsv',
    'XLS-R s258':    f'{SSL_SCORES}/xlsr_eval_seed_258.tsv',
    'XLS-R s259':    f'{SSL_SCORES}/xlsr_eval_seed_259.tsv',
    'XLS-R s260':    f'{SSL_SCORES}/xlsr_eval_seed_260.tsv',
}
HEADLINE = ['AASIST', 'XLS-R (ens)']        # the pair used for the contrasts

PROTO_COLS = ['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q',
              'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
UNCODED = '-'
ATTACK = 'ATTACK_LABEL'
SEED = 258
B_BOOT = 200

FAMILIES = {
    'legacy unit selection': ['A19'],
    'zero-shot TTS':         ['A17', 'A21', 'A22', 'A28', 'A29'],
    'voice conversion':      ['A24', 'A25', 'A26'],
}
# Malafide pairs whose SOURCE attack is also in the evaluation partition.
# A20 (from A12) and A23 (from A09) have their sources in dev, so no matched
# contrast is possible for them here.
ADV_PAIRS = [('A17', 'A18'), ('A17', 'A30')]

CODEC_TYPE = {
    'C01': 'DSP 16k opus', 'C02': 'DSP 16k amr', 'C03': 'DSP 16k speex',
    'C04': 'NEURAL Encodec', 'C05': 'DSP 16k mp3', 'C06': 'DSP 16k m4a',
    'C07': 'NEURAL mp3+Encodec', 'C08': 'DSP 8k opus', 'C09': 'DSP 8k amr',
    'C10': 'DSP 8k speex', 'C11': 'mixed 8k',
}


# ----------------------------------------------------------------- EER core
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


def _by_speaker(df, speakers):
    """Pre-split one condition into per-speaker target/nontarget arrays."""
    idx = {s: i for i, s in enumerate(speakers)}
    tar = [[] for _ in speakers]
    non = [[] for _ in speakers]
    for spk, key, sc in zip(df.SPEAKER_ID.to_numpy(), df.KEY.to_numpy(),
                            df['score'].to_numpy()):
        i = idx.get(spk)
        if i is None:
            continue
        (tar if key == 'bonafide' else non)[i].append(sc)
    return ([np.asarray(a, dtype=float) for a in tar],
            [np.asarray(a, dtype=float) for a in non])


def paired_contrast(ref_df, cond_df, b=B_BOOT, seed=SEED):
    """EER(cond) - EER(ref), with a CI on the DIFFERENCE.

    One draw of speakers is applied to BOTH conditions, so the shared-speaker
    correlation is preserved and the interval is on the quantity we claim,
    not on each EER separately."""
    speakers = sorted(set(ref_df.SPEAKER_ID) & set(cond_df.SPEAKER_ID))
    if not speakers:
        print('      [warn] no speakers shared between the two conditions')
        return np.nan, np.nan, np.nan, 0
    rt, rn = _by_speaker(ref_df, speakers)
    ct, cn = _by_speaker(cond_df, speakers)
    # A shared speaker who contributes only one class to one condition makes the
    # pooled arrays degenerate. Say so instead of returning a silent NaN.
    for lab, t, n_ in (('reference', rt, rn), ('condition', ct, cn)):
        if sum(a.size for a in t) == 0 or sum(a.size for a in n_) == 0:
            print(f'      [warn] shared speakers carry only one class in the '
                  f'{lab}; no paired contrast possible')
            return np.nan, np.nan, np.nan, len(speakers)
    point = _eer(np.concatenate(ct), np.concatenate(cn)) - \
            _eer(np.concatenate(rt), np.concatenate(rn))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        p = rng.integers(0, len(speakers), len(speakers))
        d = (_eer(np.concatenate([ct[i] for i in p]), np.concatenate([cn[i] for i in p])) -
             _eer(np.concatenate([rt[i] for i in p]), np.concatenate([rn[i] for i in p])))
        if np.isfinite(d):
            vals.append(d)
    if not vals:
        return point, np.nan, np.nan, len(speakers)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(speakers)


# -------------------------------------------------------------------- load
def load():
    print('LOADING')
    print('=======')
    proto = pd.read_csv(PROTO_EVAL, sep=r'\s+', header=None, names=PROTO_COLS, dtype=str)
    proto['filename'] = proto['FLAC_FILE_NAME']
    keep = ['filename', 'KEY', 'SPEAKER_ID', 'CODEC', 'CODEC_SEED', ATTACK]
    out = {}
    for name, path in SYSTEMS.items():
        if not os.path.exists(path):
            print(f'  [skip] {name}: not found at {path}')
            continue
        raw = pd.read_csv(path, sep='\t', dtype={'filename': str})
        col = 'cm-score' if 'cm-score' in raw.columns else raw.columns[1]
        out[name] = raw[['filename', col]].rename(columns={col: 'score'}).merge(
            proto[keep], on='filename')
        print(f'  {name:<14} {len(out[name]):>8,} rows   pooled eval EER '
              f'{eer_pct(out[name]):6.2f}%')
    return out


# ------------------------------------------------------------------ PART A
def seeds(data):
    print('\n\nPART A. ALL SEEDS BESIDE THE ENSEMBLE')
    print('======================================')
    print('  Reported together so no single seed is chosen after seeing results.')
    print('  Ensembling may or may not help; it is not guaranteed to.\n')
    rows = [{'system': n, 'pooled_EER_%': round(eer_pct(d), 3)}
            for n, d in data.items()]
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    seed_only = tab[tab.system.str.contains('s2')]
    if len(seed_only) >= 2:
        lo, hi = seed_only['pooled_EER_%'].min(), seed_only['pooled_EER_%'].max()
        print(f'\n  individual seeds span {lo:.3f} - {hi:.3f} '
              f'({hi - lo:.3f} points)')
    return tab


# ------------------------------------------------------------------ PART B
def adversarial(data):
    print('\n\nPART B. ADVERSARIAL - MATCHED DEGRADATION')
    print('==========================================')
    print('  A18 and A30 are Malafide filters applied to A17, so generator,')
    print('  partition and codec are held fixed. Difference in EER points,')
    print('  with a speaker bootstrap on the difference.')
    print('  (A20 and A23 are excluded: their source attacks live in dev.)\n')
    rows = []
    for name in HEADLINE:
        df = data.get(name)
        if df is None:
            continue
        unc = df[df.CODEC == UNCODED]
        bona = unc[unc.KEY == 'bonafide']
        for src, atk in ADV_PAIRS:
            a = pd.concat([bona, unc[(unc[ATTACK] == src) & (unc.KEY == 'spoof')]])
            c = pd.concat([bona, unc[(unc[ATTACK] == atk) & (unc.KEY == 'spoof')]])
            d, lo, hi, nspk = paired_contrast(a, c)
            rows.append({'detector': name, 'contrast': f'{src} -> {atk}',
                         'before_%': round(eer_pct(a), 2), 'after_%': round(eer_pct(c), 2),
                         'delta_pts': round(d, 2), 'ci': f'[{lo:.2f},{hi:.2f}]',
                         'speakers': nspk})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print('\n  Read delta_pts in absolute percentage points. Relative factors are')
    print('  dominated by whichever system starts nearer zero and should not be')
    print('  used to compare robustness across detectors.')
    return tab


# ------------------------------------------------------------------ PART C
def codec(data):
    print('\n\nPART C. CODEC - MATCHED DEGRADATION')
    print('====================================')
    print('  Each codec is compared against THE SAME SOURCE UTTERANCES uncoded,')
    print('  not against the whole uncoded partition. Paired bootstrap on the')
    print('  difference.\n')
    rows = []
    for name in HEADLINE:
        df = data.get(name)
        if df is None:
            continue
        coded = df[df.CODEC != UNCODED]
        n_codecs = coded.CODEC.nunique()
        sizes = coded.CODEC_SEED.value_counts()
        crossed_ids = set(sizes[sizes == n_codecs].index)
        ref = df[(df.CODEC == UNCODED) & (df.filename.isin(crossed_ids))]
        print(f'  {name}: reference is {len(ref):,} uncoded source utterances '
              f'(EER {eer_pct(ref):.2f}%)')
        sub = coded[coded.CODEC_SEED.isin(crossed_ids)]
        for c in sorted(sub.CODEC.unique()):
            cond = sub[sub.CODEC == c]
            d, lo, hi, nspk = paired_contrast(ref, cond)
            rows.append({'detector': name, 'codec': c, 'type': CODEC_TYPE.get(c, '?'),
                         'ref_eer': round(eer_pct(ref), 2),
                         'coded_eer': round(eer_pct(cond), 2),
                         'delta_pts': round(d, 2), 'ci': f'[{lo:.2f},{hi:.2f}]'})
    tab = pd.DataFrame(rows)
    print()
    for name in tab.detector.unique():
        print(f'\n  {name}')
        part = tab[tab.detector == name].sort_values('delta_pts', ascending=False)
        print(f'    {"codec":<6} {"type":<20} {"ref":>7} {"coded":>7} {"delta":>8} {"95% CI":>18}')
        for _, r in part.iterrows():
            print(f'    {r["codec"]:<6} {r["type"]:<20} {r["ref_eer"]:>7.2f} '
                  f'{r["coded_eer"]:>7.2f} {r["delta_pts"]:>8.2f} {r["ci"]:>18}')
    print('\n  A CI excluding zero means the codec changed EER for that detector.')
    return tab


# ------------------------------------------------------------------ PART D
def generator(data):
    print('\n\nPART D. GENERATOR - BETWEEN-FAMILY VARIATION')
    print('=============================================')
    print('  NOT a shift degradation. Measuring degradation from seen to unseen')
    print('  generators would need A01-A08 as a baseline, which exists only in')
    print('  the training partition: contaminated for AASIST (the released')
    print('  checkpoint saw every training speaker) and partition-confounded for')
    print('  XLS-R (different speakers and recording conditions). What follows is')
    print('  how far performance varies ACROSS unseen families, nothing more.\n')
    rows = []
    for name in HEADLINE:
        df = data.get(name)
        if df is None:
            continue
        unc = df[df.CODEC == UNCODED]
        bona = unc[unc.KEY == 'bonafide']
        for fam, atks in FAMILIES.items():
            sel = unc[(unc[ATTACK].isin(atks)) & (unc.KEY == 'spoof')]
            if len(sel):
                rows.append({'detector': name, 'family': fam,
                             'attacks': '+'.join(atks),
                             'EER_%': round(eer_pct(pd.concat([bona, sel])), 2)})
    tab = pd.DataFrame(rows)
    print(tab.pivot(index='family', columns='detector', values='EER_%').to_string())
    print('\n  Describe this as variation across families, not as degradation.')
    print('  Note also that the legacy-unit-selection row is the inversion')
    print('  finding, not a generalisation result.')
    return tab


# ------------------------------------------------------------------ PART E
def heatmap(data, save_to=None):
    print('\n\nPART E. ATTACK x CODEC GRID')
    print('============================')
    grids = {}
    for name in HEADLINE:
        df = data.get(name)
        if df is None:
            continue
        atks = sorted(df.loc[df.KEY == 'spoof', ATTACK].unique())
        cods = sorted(df.CODEC.unique(), key=lambda c: (c != UNCODED, c))
        g = pd.DataFrame(index=atks, columns=cods, dtype=float)
        for c in cods:
            sub = df[df.CODEC == c]
            bona = sub[sub.KEY == 'bonafide']
            for a in atks:
                sel = sub[(sub[ATTACK] == a) & (sub.KEY == 'spoof')]
                if len(sel) and len(bona):
                    g.loc[a, c] = eer_pct(pd.concat([bona, sel]))
        grids[name] = g
        print(f'\n  {name}  (EER %, bona fide taken from the same codec condition)')
        print(g.round(1).to_string())
    if save_to:
        draw_heatmap(grids, save_to)
    return grids


def draw_heatmap(grids, save_to):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    names = list(grids)
    fig, axes = plt.subplots(1, len(names), figsize=(13, 6.2), dpi=140)
    fig.patch.set_facecolor('#fcfcfb')
    if len(names) == 1:
        axes = [axes]
    vmax = float(np.nanmax([g.to_numpy(dtype=float).max() for g in grids.values()]))
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap('Blues')                  # sequential, one hue
    for ax, name in zip(axes, names):
        g = grids[name].astype(float)
        ax.imshow(g.to_numpy(), cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(g.columns)))
        ax.set_xticklabels(['none' if c == UNCODED else c for c in g.columns],
                           fontsize=8, rotation=45, ha='right')
        ax.set_yticks(range(len(g.index)))
        ax.set_yticklabels(g.index, fontsize=8)
        for i in range(len(g.index)):
            for j in range(len(g.columns)):
                v = g.iat[i, j]
                if not np.isfinite(v):
                    continue
                inverted = v > 50
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=7.2,
                        color='#b03a35' if inverted else
                              ('#fcfcfb' if v > vmax * 0.58 else '#14171c'),
                        fontweight='700' if inverted else '400')
        ax.set_title(name, fontsize=11.5, loc='left', color='#0b0b0b', pad=9)
        ax.tick_params(colors='#52514e', length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle('EER % by attack and codec condition — red bold marks EER above 50 (inverted)',
                 fontsize=11.5, x=0.012, ha='left', color='#0b0b0b')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_to, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\n  figure saved: {save_to}')
    plt.show()


def main(out_dir='/content/drive/MyDrive'):
    data = load()
    a = seeds(data)
    b = adversarial(data)
    c = codec(data)
    d = generator(data)
    g = heatmap(data, f'{out_dir}/258_fig_attack_codec_heatmap.png')
    for nm, t in [('seeds', a), ('adversarial', b), ('codec', c), ('generator', d)]:
        t.to_csv(f'{out_dir}/258_matched_{nm}.csv', index=False)
    print(f'\n  tables saved to {out_dir}/258_matched_*.csv')
    print('\n\nDONE. Paste the output back.')
    return a, b, c, d, g


if __name__ == '__main__':
    seed_tab, adv_tab, codec_tab, gen_tab, grids = main()
