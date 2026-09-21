#!/usr/bin/env python3

"""
run_hcg_domain_attachment
--------------------------
run hierarchical chain growth for an IDR that attaches to a folded domain at one
terminus (N-ter or C-ter), instead of growing a fully free-standing IDR.

This assumes the domain-facing terminal MD fragment of the IDR library was simulated
so that its overlap residues are the domain's own real residues, not a synthetic
ACE/NME cap -- exactly how ordinary IDR fragments already overlap each other (see
`chain_growth.fragment_list.add_domain_to_fragment_list`'s docstring). That fragment
library design happens outside this code, during simulation setup: prepare the
domain-facing fragment's sequence to genuinely include the domain's last (attaching at
the domain's C-terminus, `terminus='C'`) or first (attaching at the domain's
N-terminus, `terminus='N'`) `domain_overlap` residues, rather than capping that end.
"""
from chain_growth.hcg_list import make_hcl_l
from chain_growth.fragment_list import (generate_fragment_list, add_domain_to_fragment_list,
                                         prepare_domain_fragment)
from chain_growth.hcg_fct import hierarchical_chain_growth

################
## prepare HCG
################

## input file and path / output path
# path to MD fragments
path0 = '..'
# path to store assembled models in
path = 'truncated_tauK18_domain/'
# file with sequence, format: "fasta" or "PDB"
sequence_f = '../truncated_tauK18.fasta'

## fragment construction
# length of MD fragments (without the end-capping groups if present)
fragment_length = 5
# length of the residue overlap between subsequent fragments
# == number of the same residues in subsequent fragments
overlap = 2

## generate list of fragments, dictionary of overlaps between fragments
# generating overlaps_d is necessary, since to match the full-length sequence
# the overlap between e.g., the two last fragments can vary
fragment_l, overlaps_d = generate_fragment_list(sequence_f, fragment_length, overlap)

## the folded domain: attach the IDR to the domain's N-terminus ('N') or C-terminus ('C')
domain_id = 'domain'
domain_overlap = 2
terminus = 'C'

# prepare the domain as a rigid, single-frame MD-fragment folder, matching the
# "pair0.pdb" + "pair.xtc" convention every ordinary MD fragment folder uses
prepare_domain_fragment('folded_domain.pdb', '{}/MDfragments/{}'.format(path0, domain_id))

## lists for the HCG, with the domain spliced in at the chosen terminus
# fragment_ids : ordered fragment ids including the domain, in final sequence order
fragment_ids = add_domain_to_fragment_list(len(fragment_l), domain_id, terminus)
# hcg_l : list of paired fragments
# promo_l : list to evaluate if last fragment of level m in hcg_l is promoted to level m+1
hcg_l, promo_l = make_hcl_l(len(fragment_ids), fragment_ids=fragment_ids)

# maximal number of pairs/full-length models to assemble
kmax = 100
# MD fragments are sampled with or without end-capping groups
capping_groups = True

###########
## run HCG
###########
# strip_cap_nterm/strip_cap_cterm: the end where the domain attaches keeps its real
# terminal residue (False); the free end keeps the usual synthetic-cap stripping (True)
hierarchical_chain_growth(hcg_l, promo_l, overlaps_d, path0, path, kmax=kmax,
        capping_groups=capping_groups, domain_id=domain_id, domain_overlap=domain_overlap,
        strip_cap_nterm=False, strip_cap_cterm=True) #, verbose=True)
