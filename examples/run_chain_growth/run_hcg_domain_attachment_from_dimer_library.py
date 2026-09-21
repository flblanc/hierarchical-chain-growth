#!/usr/bin/env python3

"""
run_hcg_domain_attachment_from_dimer_library
-----------------------------------------------
run hierarchical chain growth for an IDR that attaches to a folded domain at one
terminus (N-ter or C-ter), using the pre-sampled dimer fragment library (see
`run_hcg_from_dimer_library.py`) for *every* fragment, including the one bridging the
domain and the rest of the IDR -- no new simulation needed anywhere.

Why the junction needs special handling
------------------------------------------
`get_residue_indices_for_assembly` has a known limitation, documented in its own
source comment: a 1-residue overlap only produces a valid alignment window when a
capping-group residue sits right next to it to serve as a second alignment anchor
("having 1 overlapping residue ... does work only with headgroup"). The domain
junction is always assembled with no capping group (the domain's real terminus has no
synthetic cap), so the only overlap a bare 2-residue dimer could support there without
discarding its entire contribution -- domain_overlap=1 -- hits exactly this
unsupported case. `domain_overlap=2` avoids it, but then needs a domain-adjacent
fragment with at least 3 real residues (2 shared with the domain, at least 1 newly
contributed) -- more than any single dimer-library entry has.

`chain_growth.fragment_list.build_domain_junction_fragment` builds that 3-residue
fragment *without* any new simulation: every one of the dimer library's 400
residue-pair combinations already exists in it, so it looks up "[domain's own last 2
residues]" and "[domain's own last residue, the IDR's boundary residue]" (or the
mirror image, for N-terminal attachment) directly from the library, and merges them
with the completely ordinary (non-domain) overlap=1/capping_groups=True mechanism --
the same as any other internal HCG join. See its own docstring for the full
derivation.

Every other IDR fragment doesn't touch the domain at all, so it's an ordinary
overlap=1/capping_groups=True join, sourced from the dimer library exactly as in
`run_hcg_from_dimer_library.py`.
"""
import os

from chain_growth.hcg_list import make_hcl_l
from chain_growth.fragment_list import (build_domain_junction_fragment, dimer_library_fragment_dir,
                                         generate_fragment_list, prepare_domain_fragment)
from chain_growth.hcg_fct import hierarchical_chain_growth

################
## prepare HCG
################

## path to the unpacked dimer library (400 folders "0".."399", each with fragment.pdb/fragment.xtc)
dimer_library = '/data2/hcg-fragment-library/dimerLibrary'
# sequence covering only the dimer-library-grown part of the IDR (i.e. everything
# except the domain itself, which is read separately from folded_domain_pdb below)
sequence_f = 'sequence.fasta'
# path to store assembled models in
path = 'domain_attachment_from_dimer_library/'
# where the per-fragment MDfragments/<id>/pair0.pdb + pair.xtc (symlinks for ordinary
# fragments, real files for the domain and the junction fragment) are created
path0 = 'MDfragments_from_dimer_library/'
# path to the folded domain's PDB structure (a single, rigid conformation)
folded_domain_pdb = 'folded_domain.pdb'

## fragment construction: dimer library fragments are 2 residues long with 1 residue
## overlap between consecutive windows (see run_hcg_from_dimer_library.py)
fragment_length = 2
overlap = 1

## generate the ordinary (dimer-library) IDR fragment list and overlaps_d as usual
fragment_l, overlaps_d = generate_fragment_list(sequence_f, fragment_length, overlap)

## for each ordinary IDR fragment, symlink its MDfragments folder to the matching dimer
os.makedirs(os.path.join(path0, 'MDfragments'), exist_ok=True)
for i, (res_a, res_b) in enumerate(fragment_l):
    dst = os.path.join(path0, 'MDfragments', str(i))
    os.makedirs(dst, exist_ok=True)
    src = dimer_library_fragment_dir(dimer_library, res_a, res_b)
    for fname, linkname in (('fragment.pdb', 'pair0.pdb'), ('fragment.xtc', 'pair.xtc')):
        link = os.path.join(dst, linkname)
        if not os.path.exists(link):
            os.symlink(os.path.join(src, fname), link)

## the folded domain: attach the IDR to the domain's N-terminus ('N') or C-terminus ('C')
domain_id = 'domain'
terminus = 'C'
# residues the domain shares with the junction fragment's overlap region; see
# build_domain_junction_fragment's docstring for why this must be 2, not 1
domain_overlap = 2
# number of full-length conformers to assemble (also used for the junction's own ensemble)
kmax = 200

# prepare the domain as a rigid, single-frame MD-fragment folder, matching the
# "pair0.pdb" + "pair.xtc" convention every ordinary MD fragment folder uses
prepare_domain_fragment(folded_domain_pdb, '{}/MDfragments/{}'.format(path0, domain_id))

# the one fragment bridging the domain and the dimer-library-grown IDR, built purely
# from the dimer library -- no new simulation
junction_id = 'domain_junction'
idr_boundary_residue = fragment_l[0][0] if terminus == 'C' else fragment_l[-1][-1]
build_domain_junction_fragment(
    dimer_library=dimer_library, domain_pdb=folded_domain_pdb,
    idr_boundary_residue=idr_boundary_residue, terminus=terminus,
    path0=path0, junction_id=junction_id, kmax=kmax)

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
