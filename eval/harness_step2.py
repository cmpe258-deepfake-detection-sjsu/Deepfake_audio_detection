"""
CMPE 258 - evaluation harness
STEP 2: find out how the eval protocol really groups files, then take a first
        look at the generator and codec shifts.

Run this in the same notebook, after step 1 (it reuses `merged`).
Read-only, no audio, no GPU.

Three questions it answers:
  Q1. What are AC1 / AC2 / AC3?  (ATTACK_TAG vs ATTACK_LABEL)
  Q2. Can a coded file be paired with its uncoded original?  The proposal
      assumed CODEC_SEED does this. Step 1 showed median group size 1, so it
      probably does not. This checks every candidate pairing key and says
      plainly whether paired analysis is possible.
  Q3. What do the two shifts look like before any formal slicing?
"""

import numpy as np
import pandas as pd

UNCODED = '-'          # step 1 showed the uncoded condition is '-', not 'C00'
ATTACK = 'ATTACK_LABEL'  # the real attack identifier; ATTACK_TAG is something else


# ---------------------------------------------------------------- Q1
def what_are_the_tags(m):
    print('Q1. ATTACK_TAG vs ATTACK_LABEL')
    print('==============================')
    ct = pd.crosstab(m[ATTACK], m.ATTACK_TAG)
    print(ct.to_string())
    print('\n  Read this as: which attacks fall under each AC group. If each')
    print('  attack sits in exactly one AC, the ACs are a coarser grouping of')
    print('  attacks and we can ignore them. If an attack is spread across')
    print('  several ACs, ATTACK_TAG encodes something else (likely a codec or')
    print('  condition grouping) and is worth understanding before slicing.')

    spread = (ct > 0).sum(axis=1)
    multi = spread[spread > 1]
    if len(multi):
        print(f'\n  {len(multi)} attack(s) appear under more than one AC:')
        print('   ', dict(multi))
    else:
        print('\n  Every attack sits in exactly one AC group.')


# ---------------------------------------------------------------- Q2
def can_we_pair(m):
    print('\n\nQ2. CAN CODED FILES BE PAIRED WITH THEIR UNCODED ORIGINAL?')
    print('==========================================================')

    uncoded = m[m.CODEC == UNCODED]
    coded = m[m.CODEC != UNCODED]
    print(f'  uncoded files   {len(uncoded):>8,}')
    print(f'  coded files     {len(coded):>8,}')
    print(f'  ratio           {len(coded) / max(1, len(uncoded)):.2f} coded per uncoded')

    # --- CODEC_SEED, the key the proposal assumed ---
    print('\n  CODEC_SEED group sizes among CODED files only:')
    sizes = coded.CODEC_SEED.value_counts()
    dist = sizes.value_counts().sort_index()
    for size, n in dist.head(12).items():
        print(f'    {n:>8,} seeds cover exactly {size} file(s)')
    print(f'    distinct seeds among coded files: {coded.CODEC_SEED.nunique():,}')

    shared = sizes[sizes > 1]
    if len(shared):
        print(f'\n  {len(shared):,} seeds cover more than one coded file. Example:')
        example = shared.index[0]
        cols = ['filename', 'SPEAKER_ID', 'CODEC', ATTACK, 'KEY', 'CODEC_SEED']
        print(coded[coded.CODEC_SEED == example][cols].to_string(index=False))
        print('\n  If those rows are the SAME utterance under DIFFERENT codecs,')
        print('  CODEC_SEED pairs coded files with each other (but still not with')
        print('  the uncoded original, which carries a different seed value).')
    else:
        print('\n  No coded seed covers more than one file: CODEC_SEED is unique')
        print('  per file and cannot pair anything.')

    # --- does any seed appear in BOTH coded and uncoded? ---
    overlap = set(coded.CODEC_SEED.unique()) & set(uncoded.CODEC_SEED.unique())
    print(f'\n  seeds shared between coded and uncoded files: {len(overlap)}')
    if not overlap:
        print('  -> CODEC_SEED cannot link a coded file to its uncoded original.')

    # --- filenames: do coded and uncoded share a stem? ---
    print('\n  SAMPLE FILENAMES')
    print(f'    uncoded : {list(uncoded.filename.head(5))}')
    for c in sorted(coded.CODEC.unique())[:3]:
        print(f'    {c:<8}: {list(coded[coded.CODEC == c].filename.head(5))}')
    print('\n  If coded names are a transformation of an uncoded name (same stem,')
    print('  added suffix), we can pair on the stem. If they are unrelated IDs,')
    print('  no utterance-level pairing exists and the codec slice must be')
    print('  compared at the group level instead.')

    # --- speaker + attack as a coarser matched comparison ---
    print('\n  FALLBACK: speaker x attack coverage')
    key = ['SPEAKER_ID', ATTACK]
    u_keys = set(map(tuple, uncoded[key].drop_duplicates().to_numpy()))
    c_keys = set(map(tuple, coded[key].drop_duplicates().to_numpy()))
    both = u_keys & c_keys
    print(f'    speaker x attack cells, uncoded : {len(u_keys):,}')
    print(f'    speaker x attack cells, coded   : {len(c_keys):,}')
    print(f'    present in both                 : {len(both):,} '
          f'({len(both) / max(1, len(u_keys)):.1%} of uncoded cells)')
    print('    -> if this is high, the codec effect can be measured within')
    print('       speaker and attack even without utterance-level pairing.')


# ---------------------------------------------------------------- EER helpers
def det_curve(t, n_):
    n = t.size + n_.size
    s = np.concatenate((t, n_))
    lab = np.concatenate((np.ones(t.size), np.zeros(n_.size)))
    o = np.argsort(s, kind='mergesort')
    lab = lab[o]
    ts = np.cumsum(lab)
    ns = n_.size - (np.arange(1, n + 1) - ts)
    frr = np.concatenate((np.atleast_1d(0), ts / t.size))
    far = np.concatenate((np.atleast_1d(1), ns / n_.size))
    return frr, far


def eer_pct(df):
    t = df.loc[df.KEY == 'bonafide', 'cm-score'].to_numpy()
    n_ = df.loc[df.KEY == 'spoof', 'cm-score'].to_numpy()
    if t.size == 0 or n_.size == 0:
        return np.nan
    frr, far = det_curve(t, n_)
    i = np.argmin(np.abs(frr - far))
    return 100 * float(np.mean((frr[i], far[i])))


# ---------------------------------------------------------------- Q3
def first_look(m):
    print('\n\nQ3. FIRST LOOK AT THE TWO SHIFTS')
    print('================================')

    unc = m[m.CODEC == UNCODED]
    bona_unc = unc[unc.KEY == 'bonafide']

    print('\n  GENERATOR SHIFT - EER per attack, uncoded only,')
    print('  each attack scored against the same uncoded bona fide pool')
    rows = []
    for a in sorted(unc.loc[unc.KEY == 'spoof', ATTACK].unique()):
        sub = pd.concat([bona_unc, unc[(unc[ATTACK] == a) & (unc.KEY == 'spoof')]])
        rows.append({'attack': a,
                     'n_spoof': int((sub.KEY == 'spoof').sum()),
                     'speakers': sub.SPEAKER_ID.nunique(),
                     'EER_%': round(eer_pct(sub), 2)})
    gen = pd.DataFrame(rows).sort_values('EER_%', ascending=False)
    print(gen.to_string(index=False))
    print(f'\n  pooled uncoded EER: {eer_pct(unc):.2f}%')

    print('\n  CODEC SHIFT - EER per codec condition')
    print('  (each codec scored on its own files: bona fide and spoof both coded)')
    rows = []
    for c in sorted(m.CODEC.unique()):
        sub = m[m.CODEC == c]
        rows.append({'codec': c, 'n_files': len(sub),
                     'n_bonafide': int((sub.KEY == 'bonafide').sum()),
                     'EER_%': round(eer_pct(sub), 2)})
    cod = pd.DataFrame(rows)
    print(cod.to_string(index=False))

    print('\n  ADVERSARIAL - the Malafide and Malacopula attacks named in the')
    print('  proposal, uncoded, against the uncoded bona fide pool')
    adversarial = {'A18': 'Malafide (from A17)', 'A20': 'Malafide (from A12)',
                   'A23': 'Malafide (from A09)', 'A30': 'Malafide (from A17)',
                   'A27': 'Malacopula', 'A31': 'Malacopula', 'A32': 'Malacopula'}
    rows = []
    for a, what in adversarial.items():
        sel = unc[(unc[ATTACK] == a) & (unc.KEY == 'spoof')]
        if not len(sel):
            rows.append({'attack': a, 'kind': what, 'n_spoof': 0, 'EER_%': np.nan})
            continue
        sub = pd.concat([bona_unc, sel])
        rows.append({'attack': a, 'kind': what, 'n_spoof': len(sel),
                     'EER_%': round(eer_pct(sub), 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n  ATTACK x CODEC coverage (rows = attacks, columns = codecs)')
    print(pd.crosstab(m[ATTACK], m.CODEC).to_string())

    return gen, cod


def main(m):
    what_are_the_tags(m)
    can_we_pair(m)
    gen, cod = first_look(m)
    print('\n\nDONE. Paste the output back; next step is the formal slices and')
    print('the speaker-level bootstrap.')
    return gen, cod


if __name__ == '__main__':
    try:
        gen_table, codec_table = main(merged)   # noqa: F821  (from step 1)
    except NameError:
        print('Run step 1 first in this notebook - this script needs `merged`.')
