"""
CMPE 258 - evaluation harness
STEP 3: resolve the pairing question, build the formal slices, and put
        speaker-level bootstrap intervals on everything.

Run in the same notebook after steps 1 and 2. Needs `merged`.
Read-only on Drive; writes nothing. No audio, no GPU.

Part A  Is CODEC_SEED the filename of the uncoded source utterance?
        Step 2 compared coded seeds against uncoded SEEDS (all '-'), which was
        the wrong comparison. The right one is coded seed vs uncoded FILENAME.
Part B  The three slices, each against a matched reference.
Part C  Speaker-level bootstrap, then the results table.
"""

import numpy as np
import pandas as pd

UNCODED = '-'
ATTACK = 'ATTACK_LABEL'
SEED = 258
B_BOOT = 200        # raise to 1000 for the final run; 200 is fine while iterating

# Attack families, from the project README. A18/A20/A23/A30 are Malafide and
# A27/A31/A32 are Malacopula, so they are adversarial, not generator families.
FAMILIES = {
    'legacy unit selection': ['A19'],
    'zero-shot TTS':         ['A17', 'A21', 'A22', 'A28', 'A29'],
    'voice conversion':      ['A24', 'A25', 'A26'],
}
MALAFIDE = {'A18': 'A17', 'A30': 'A17', 'A20': None, 'A23': None}  # None = source is in dev
MALACOPULA = ['A27', 'A31', 'A32']


# ------------------------------------------------------------------ EER
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
    return _eer(df.loc[df.KEY == 'bonafide', 'cm-score'].to_numpy(),
                df.loc[df.KEY == 'spoof', 'cm-score'].to_numpy())


# ------------------------------------------------------------------ Part A
def resolve_pairing(m):
    print('PART A. IS CODEC_SEED THE SOURCE UTTERANCE FILENAME?')
    print('=====================================================')
    uncoded = m[m.CODEC == UNCODED]
    coded = m[m.CODEC != UNCODED]

    uncoded_names = set(uncoded.filename)
    coded_seeds = set(coded.CODEC_SEED.unique())
    hit = coded_seeds & uncoded_names
    print(f'  distinct coded seeds              {len(coded_seeds):>9,}')
    print(f'  of those that are an uncoded file {len(hit):>9,}  '
          f'({len(hit) / max(1, len(coded_seeds)):.1%})')

    if len(hit) / max(1, len(coded_seeds)) > 0.5:
        print('\n  CONFIRMED: CODEC_SEED names the uncoded source utterance.')
        print('  True utterance-level pairing is available.')
        paired = True
    else:
        print('\n  NOT a filename link. Falling back to the fully-crossed codec')
        print('  subset, which still pairs codecs against each other.')
        paired = False

    if paired:
        # sanity: does a coded file agree with its source on speaker and attack?
        src = uncoded.set_index('filename')[['SPEAKER_ID', ATTACK, 'KEY', 'cm-score']]
        src.columns = ['src_speaker', 'src_attack', 'src_key', 'src_score']
        j = coded.join(src, on='CODEC_SEED', how='inner')
        print(f'\n  coded files linked to a source     {len(j):>9,} of {len(coded):,}')
        agree_spk = (j.SPEAKER_ID == j.src_speaker).mean()
        agree_att = (j[ATTACK] == j.src_attack).mean()
        print(f'  speaker agrees with source         {agree_spk:.2%}')
        print(f'  attack agrees with source          {agree_att:.2%}')
        if agree_spk > 0.99 and agree_att > 0.99:
            print('  -> consistent. Safe to use for paired codec analysis.')
        else:
            print('  -> INCONSISTENT. Do not use paired analysis; investigate.')
        return paired, j

    return paired, None


def crossed_subset(m):
    """The 11-codec fully crossed subset: source utterances encoded under
    every codec. Cleanest possible codec comparison."""
    coded = m[m.CODEC != UNCODED]
    n_codecs = coded.CODEC.nunique()
    sizes = coded.CODEC_SEED.value_counts()
    full = sizes[sizes == n_codecs].index
    sub = coded[coded.CODEC_SEED.isin(full)]
    print(f'\n  FULLY CROSSED SUBSET: {len(full):,} source utterances x '
          f'{n_codecs} codecs = {len(sub):,} files')
    return sub


# ------------------------------------------------------------------ Part C
def bootstrap_eer(df, b=B_BOOT, seed=SEED):
    """95% CI by resampling SPEAKERS with replacement (not utterances:
    utterances from one speaker are not independent)."""
    spk = df.SPEAKER_ID.to_numpy()
    sc = df['cm-score'].to_numpy()
    isb = (df.KEY == 'bonafide').to_numpy()
    uniq, inv = np.unique(spk, return_inverse=True)
    tar_by = [sc[(inv == i) & isb] for i in range(len(uniq))]
    non_by = [sc[(inv == i) & ~isb] for i in range(len(uniq))]

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.integers(0, len(uniq), len(uniq))
        t = np.concatenate([tar_by[i] for i in pick])
        n_ = np.concatenate([non_by[i] for i in pick])
        v = _eer(t, n_)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def row(name, df, kind):
    lo, hi = bootstrap_eer(df)
    return {'slice': name, 'kind': kind,
            'n_spoof': int((df.KEY == 'spoof').sum()),
            'speakers': df.SPEAKER_ID.nunique(),
            'EER_%': round(eer_pct(df), 2),
            'ci_lo': round(lo, 2), 'ci_hi': round(hi, 2)}


# ------------------------------------------------------------------ Part B
def build_slices(m, crossed):
    print('\n\nPART B/C. SLICES WITH SPEAKER-LEVEL BOOTSTRAP')
    print('==============================================')
    print(f'  (bootstrap B={B_BOOT}; raise to 1000 for the final table)')
    unc = m[m.CODEC == UNCODED]
    bona = unc[unc.KEY == 'bonafide']
    out = []

    print('\n  reference...')
    out.append(row('ALL attacks, uncoded', unc, 'reference'))

    print('  generator families...')
    for fam, attacks in FAMILIES.items():
        sel = unc[(unc[ATTACK].isin(attacks)) & (unc.KEY == 'spoof')]
        if len(sel):
            out.append(row(f'{fam} ({"+".join(attacks)})',
                           pd.concat([bona, sel]), 'generator'))

    print('  adversarial...')
    for atk, src in MALAFIDE.items():
        sel = unc[(unc[ATTACK] == atk) & (unc.KEY == 'spoof')]
        if len(sel):
            label = f'Malafide {atk}' + (f' (from {src})' if src else ' (source in dev)')
            out.append(row(label, pd.concat([bona, sel]), 'adversarial'))
    sel = unc[(unc[ATTACK].isin(MALACOPULA)) & (unc.KEY == 'spoof')]
    if len(sel):
        out.append(row(f'Malacopula ({"+".join(MALACOPULA)})',
                       pd.concat([bona, sel]), 'adversarial'))

    print('  adversarial, source-matched (the clean comparison)...')
    for atk, src in MALAFIDE.items():
        if src is None:
            continue
        s_sel = unc[(unc[ATTACK] == src) & (unc.KEY == 'spoof')]
        a_sel = unc[(unc[ATTACK] == atk) & (unc.KEY == 'spoof')]
        if len(s_sel) and len(a_sel):
            out.append(row(f'  source {src} alone', pd.concat([bona, s_sel]), 'adv-paired'))
            out.append(row(f'  -> {atk} (adversarial)', pd.concat([bona, a_sel]), 'adv-paired'))

    print('  codec, on the fully crossed subset...')
    cb = crossed  # bona fide and spoof both coded, within this subset
    for c in sorted(cb.CODEC.unique()):
        out.append(row(f'codec {c}', cb[cb.CODEC == c], 'codec'))

    return pd.DataFrame(out)


def report(tab):
    print('\n\nRESULTS  (EER %, 95% speaker-level bootstrap CI)')
    print('=================================================')
    for kind in ['reference', 'generator', 'adversarial', 'adv-paired', 'codec']:
        part = tab[tab['kind'] == kind]
        if not len(part):
            continue
        print(f'\n{kind.upper()}')
        for _, r in part.iterrows():
            print(f"  {r['slice']:<40} {r['EER_%']:>6.2f}  "
                  f"[{r['ci_lo']:>6.2f}, {r['ci_hi']:>6.2f}]   "
                  f"n={r['n_spoof']:>7,}  spk={r['speakers']}")

    ref = tab[tab['kind'] == 'reference']['EER_%'].iloc[0] if len(tab[tab['kind'] == 'reference']) else np.nan
    print(f'\n  Reference (all attacks, uncoded): {ref:.2f}%')
    print('  A slice whose CI excludes the reference is a real effect.')
    print('  Compare the SPREAD within each shift type, not just the worst cell:')
    print('  that spread is what answers "which shift breaks detectors".')


def main(m):
    paired, joined = resolve_pairing(m)
    crossed = crossed_subset(m)
    tab = build_slices(m, crossed)
    report(tab)
    print('\n\nDONE. Paste the output back.')
    return tab, paired, joined


if __name__ == '__main__':
    try:
        results, is_paired, joined_df = main(merged)   # noqa: F821
    except NameError:
        print('Run step 1 first in this notebook - this script needs `merged`.')
