#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for `chain_growth.fragment_list.build_domain_junction_fragment`,
which builds the one fragment bridging a rigid folded domain and the pre-sampled,
generic dimer fragment library, with no new simulation -- see its docstring and
`examples/run_chain_growth/run_hcg_domain_attachment_from_dimer_library.py`.

These tests need the actual generic dimer fragment library (400 folders "0".."399",
see README's "Web application of HCG" section), which is a separate, not-checked-into-
this-repo resource -- they are skipped when it isn't present at the conventional path.
"""

import os

import MDAnalysis as mda
import pytest

from chain_growth.fragment_list import (build_domain_junction_fragment, dimer_library_fragment_dir,
                                         generate_fragment_list, get_sequence, prepare_domain_fragment)
from chain_growth.hcg_fct import hierarchical_chain_growth
from chain_growth.hcg_list import make_hcl_l

DIMER_LIBRARY = '/data2/hcg-fragment-library/dimerLibrary'
pytestmark = pytest.mark.skipif(
    not os.path.isdir(DIMER_LIBRARY),
    reason='generic dimer fragment library not present at {}'.format(DIMER_LIBRARY))

test_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(test_dir, '../../examples/')


def _prepare_stand_in_domain(out_pdb):
    '''A real folded domain PDB carries no ACE/NME caps at all; strip both from an
    existing MD fragment's first frame to build a plausible stand-in.'''
    u = mda.Universe(os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'))
    domain_atoms = u.select_atoms('not resname ACE and not resname NME')
    domain_atoms.write(out_pdb)


def _symlink_ordinary_fragments(path0, fragment_l):
    '''symlink each ordinary (non-domain-adjacent) IDR fragment's MDfragments folder
    to its matching entry in the generic dimer library, exactly as
    run_hcg_from_dimer_library.py does.'''
    mdfragments_dir = os.path.join(path0, 'MDfragments')
    for i, (res_a, res_b) in enumerate(fragment_l):
        dst = os.path.join(mdfragments_dir, str(i))
        os.makedirs(dst, exist_ok=True)
        src = dimer_library_fragment_dir(DIMER_LIBRARY, res_a, res_b)
        for fname, linkname in (('fragment.pdb', 'pair0.pdb'), ('fragment.xtc', 'pair.xtc')):
            link = os.path.join(dst, linkname)
            if not os.path.exists(link):
                os.symlink(os.path.join(src, fname), link)


def test_domain_junction_c_terminus(tmp_path):
    domain_pdb = str(tmp_path / 'stand_in_domain.pdb')
    _prepare_stand_in_domain(domain_pdb)
    domain_sequence = get_sequence(domain_pdb, NA=False).get_sequence_list()
    assert domain_sequence == ['LEU', 'GLN', 'THR', 'ALA', 'PRO']

    path0 = str(tmp_path / 'MDfragments_from_dimer_library')
    domain_id = 'domain'
    prepare_domain_fragment(domain_pdb, os.path.join(path0, 'MDfragments', domain_id))

    sequence_f = str(tmp_path / 'sequence.fasta')
    with open(sequence_f, 'w') as f:
        f.write('>test\nGAVLI\n')
    fragment_l, overlaps_d = generate_fragment_list(sequence_f, fragment_length=2, overlap=1)
    _symlink_ordinary_fragments(path0, fragment_l)

    junction_id = 'domain_junction'
    build_domain_junction_fragment(
        dimer_library=DIMER_LIBRARY, domain_pdb=domain_pdb,
        idr_boundary_residue=fragment_l[0][0], terminus='C',
        path0=path0, junction_id=junction_id, kmax=5, rmsd_cut_off=3.0)

    # domain's own last 2 residues (ALA, PRO) plus the IDR's own first residue (GLY);
    # domain-facing end (leading) capless, free end (trailing) keeps its normal cap
    u_junction = mda.Universe(os.path.join(path0, 'MDfragments', junction_id, 'pair0.pdb'),
                               os.path.join(path0, 'MDfragments', junction_id, 'pair.xtc'))
    assert list(u_junction.residues.resnames) == ['ALA', 'PRO', 'GLY', 'NME']
    assert u_junction.trajectory.n_frames == 5

    fragment_ids = [domain_id, junction_id] + list(range(len(fragment_l)))
    hcg_l, promo_l = make_hcl_l(len(fragment_ids), fragment_ids=fragment_ids)
    out_path = str(tmp_path / 'out')
    hierarchical_chain_growth(
        hcg_l, promo_l, overlaps_d, path0, out_path, kmax=5, capping_groups=True,
        domain_id=domain_id, domain_overlap=2, strip_cap_nterm=False, strip_cap_cterm=True,
        rmsd_cut_off=3.0)

    final_level = len(hcg_l)
    final_pdb = os.path.join(out_path, str(final_level), domain_id, 'pair0.pdb')
    assert os.path.exists(final_pdb)
    u_final = mda.Universe(final_pdb)
    # no ACE: the stand-in domain fixture has no caps at all, matching a real domain
    assert list(u_final.residues.resnames) == [
        'LEU', 'GLN', 'THR', 'ALA', 'PRO', 'GLY', 'ALA', 'VAL', 'LEU', 'ILE']


def test_domain_junction_n_terminus(tmp_path):
    domain_pdb = str(tmp_path / 'stand_in_domain.pdb')
    _prepare_stand_in_domain(domain_pdb)

    path0 = str(tmp_path / 'MDfragments_from_dimer_library')
    domain_id = 'domain'
    prepare_domain_fragment(domain_pdb, os.path.join(path0, 'MDfragments', domain_id))

    sequence_f = str(tmp_path / 'sequence.fasta')
    with open(sequence_f, 'w') as f:
        f.write('>test\nGAVLI\n')
    fragment_l, overlaps_d = generate_fragment_list(sequence_f, fragment_length=2, overlap=1)
    _symlink_ordinary_fragments(path0, fragment_l)

    junction_id = 'domain_junction'
    build_domain_junction_fragment(
        dimer_library=DIMER_LIBRARY, domain_pdb=domain_pdb,
        idr_boundary_residue=fragment_l[-1][-1], terminus='N',
        path0=path0, junction_id=junction_id, kmax=5, rmsd_cut_off=3.0)

    # free end (leading, IDR's own last residue) keeps its normal cap; domain-facing
    # end (trailing) is capless, ending in domain's own first 2 residues (LEU, GLN)
    u_junction = mda.Universe(os.path.join(path0, 'MDfragments', junction_id, 'pair0.pdb'))
    assert list(u_junction.residues.resnames) == ['ACE', 'ILE', 'LEU', 'GLN']

    fragment_ids = list(range(len(fragment_l))) + [junction_id, domain_id]
    hcg_l, promo_l = make_hcl_l(len(fragment_ids), fragment_ids=fragment_ids)
    out_path = str(tmp_path / 'out')
    hierarchical_chain_growth(
        hcg_l, promo_l, overlaps_d, path0, out_path, kmax=5, capping_groups=True,
        domain_id=domain_id, domain_overlap=2, strip_cap_nterm=True, strip_cap_cterm=False,
        rmsd_cut_off=3.0)

    final_level = len(hcg_l)
    final_pdb = os.path.join(out_path, str(final_level), '0', 'pair0.pdb')
    assert os.path.exists(final_pdb)
    u_final = mda.Universe(final_pdb)
    assert list(u_final.residues.resnames) == [
        'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'LEU', 'GLN', 'THR', 'ALA', 'PRO']
