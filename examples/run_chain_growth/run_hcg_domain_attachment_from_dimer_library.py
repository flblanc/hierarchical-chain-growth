#!/usr/bin/env python3

"""
run_hcg_domain_attachment_from_dimer_library
-----------------------------------------------
run hierarchical chain growth for an IDR that attaches to a folded domain at one
terminus (N-ter or C-ter), using the pre-sampled dimer fragment library (see
`run_hcg_from_dimer_library.py`) for the bulk of the IDR's fragments, combined with a
single purpose-prepared fragment at the domain junction (as in
`run_hcg_domain_attachment.py`).

Why the junction can't also come from the dimer library
---------------------------------------------------------
`get_residue_indices_for_assembly` has a known limitation, documented in its own
source comment: a 1-residue overlap only produces a valid alignment window when a
capping-group residue sits right next to it to serve as a second alignment anchor
("having 1 overlapping residue ... does work only with headgroup"). The domain
junction is always assembled with no capping group (the domain's real terminus has no
synthetic cap), so the only overlap a bare 2-residue dimer could support there without
discarding its entire contribution -- domain_overlap=1 -- hits exactly this
unsupported case. Using domain_overlap=2 instead avoids that, but then needs a
domain-adjacent fragment with at least 3 real residues (2 shared with the domain, at
least 1 newly contributed) -- more than a dimer has. So the domain-adjacent fragment
must be purpose-prepared (as in `run_hcg_domain_attachment.py`), not sourced from the
dimer library.

Every other IDR fragment doesn't touch the domain at all, so it's an ordinary
overlap=1/capping_groups=True join and is sourced from the dimer library exactly as in
`run_hcg_from_dimer_library.py`. The dimer-library-grown part is joined to the
junction fragment's own free (non-domain-facing) end the same ordinary way -- that
end keeps its normal synthetic cap, used as the alignment anchor for that join.

Sequence bookkeeping
---------------------
`sequence_f` below must cover only the part of the IDR grown from the dimer library --
i.e. everything *after* the residues already present in the junction fragment's own
new (non-domain-shared) contribution. Splitting the full IDR sequence at the right
residue is the same fragment-library design step `run_hcg_domain_attachment.py`
already asks for: it happens outside this code, when the junction fragment is
prepared (e.g. from a short simulation, or sliced from a longer existing fragment).
"""
import os

from chain_growth.hcg_list import make_hcl_l
from chain_growth.fragment_list import generate_fragment_list, prepare_domain_fragment
from chain_growth.hcg_fct import hierarchical_chain_growth

################
## prepare HCG
################

## path to the unpacked dimer library (400 folders "0".."399", each with fragment.pdb/fragment.xtc)
dimer_library = '/data2/hcg-fragment-library/dimerLibrary'
# sequence covering only the dimer-library-grown part of the IDR (see module docstring)
sequence_f = 'sequence.fasta'
# path to store assembled models in
path = 'domain_attachment_from_dimer_library/'
# where the per-fragment MDfragments/<id>/pair0.pdb + pair.xtc (symlinks for ordinary
# fragments, real files for the domain and the junction fragment) are created
path0 = 'MDfragments_from_dimer_library/'

# amino-acid order used to index the dimer library: folder index = 20*i + j, where i, j
# are this list's positions of the fragment's first and second residue, respectively
DIMER_ORDER = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'THR', 'SER', 'CYS', 'GLN', 'ASN',
               'GLU', 'ASP', 'LYS', 'TRP', 'ARG', 'TYR', 'PHE', 'HIS', 'PRO', 'MET']

## fragment construction: dimer library fragments are 2 residues long with 1 residue
## overlap between consecutive windows (see run_hcg_from_dimer_library.py)
fragment_length = 2
overlap = 1

## generate the ordinary (dimer-library) IDR fragment list and overlaps_d as usual
fragment_l, overlaps_d = generate_fragment_list(sequence_f, fragment_length, overlap)

## for each ordinary IDR fragment, symlink its MDfragments folder to the matching dimer
os.makedirs(os.path.join(path0, 'MDfragments'), exist_ok=True)
for i, (res_a, res_b) in enumerate(fragment_l):
    dimer_index = 20 * DIMER_ORDER.index(res_a) + DIMER_ORDER.index(res_b)
    dst = os.path.join(path0, 'MDfragments', str(i))
    os.makedirs(dst, exist_ok=True)
    src = os.path.join(dimer_library, str(dimer_index))
    for fname, linkname in (('fragment.pdb', 'pair0.pdb'), ('fragment.xtc', 'pair.xtc')):
        link = os.path.join(dst, linkname)
        if not os.path.exists(link):
            os.symlink(os.path.join(src, fname), link)

## the folded domain: attach the IDR to the domain's N-terminus ('N') or C-terminus ('C')
domain_id = 'domain'
terminus = 'C'

# prepare the domain as a rigid, single-frame MD-fragment folder, matching the
# "pair0.pdb" + "pair.xtc" convention every ordinary MD fragment folder uses
prepare_domain_fragment('folded_domain.pdb', '{}/MDfragments/{}'.format(path0, domain_id))

# the one purpose-prepared fragment bridging the domain and the dimer-library-grown
# IDR (see module docstring). Unlike domain_id, there's no library or simple structure
# file to convert here -- it needs its own short simulation (or slicing from a longer
# existing fragment), so its "pair0.pdb"/"pair.xtc" must already exist under
# path0/MDfragments/<junction_id> before this script runs; prepare_domain_fragment
# can still write it there once you have that single-conformation (or trajectory)
# structure, e.g. prepare_domain_fragment('domain_junction.pdb', '{}/MDfragments/{}'
# .format(path0, junction_id))
junction_id = 'domain_junction'
# residues the domain shares with junction_id's own overlap region (>=2; see module
# docstring for why domain_overlap=1 isn't supported at this junction)
domain_overlap = 2

## lists for the HCG: [ordinary dimer-library fragments..., junction, domain]
## (attaching at the domain's N-terminus) or [domain, junction, ordinary
## dimer-library fragments...] (attaching at the domain's C-terminus)
n_ordinary = len(fragment_l)
if terminus == 'N':
    fragment_ids = list(range(n_ordinary)) + [junction_id, domain_id]
elif terminus == 'C':
    fragment_ids = [domain_id, junction_id] + list(range(n_ordinary))
else:
    raise ValueError("terminus must be 'N' or 'C'")
hcg_l, promo_l = make_hcl_l(len(fragment_ids), fragment_ids=fragment_ids)

# number of full-length conformers to assemble
kmax = 200
# dimer library fragments (and the junction fragment's free end) are ACE/NME-capped
capping_groups = True

###########
## run HCG
###########
# strip_cap_nterm/strip_cap_cterm: the end where the domain attaches keeps its real
# terminal residue (False); the free end keeps the usual synthetic-cap stripping (True).
# For terminus='N' instead, swap these two (strip_cap_nterm=True, strip_cap_cterm=False).
hierarchical_chain_growth(hcg_l, promo_l, overlaps_d, path0, path, kmax=kmax,
        capping_groups=capping_groups, domain_id=domain_id, domain_overlap=domain_overlap,
        strip_cap_nterm=False, strip_cap_cterm=True) #, verbose=True)
