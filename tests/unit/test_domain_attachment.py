#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import MDAnalysis as mda
import pytest

from chain_growth.hcg_fct import get_residue_indices_for_assembly
from chain_growth.hcg_list import make_hcl_l
from chain_growth.fragment_list import add_domain_to_fragment_list, prepare_domain_fragment

test_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(test_dir, '../../examples/')


@pytest.mark.parametrize("overlap0", [1, 2])
def test_get_residue_indices_no_cap_offset_for_any_overlap(overlap0):
    '''capping_groups=False must drop the synthetic-cap offset (e=0) regardless of
    overlap0, including overlap0==1 -- a case the original condition
    (`overlap0 > 1 and capping_groups == False`) missed.'''
    index_aln_l, index_clash_l, index_merge_l = get_residue_indices_for_assembly(
        overlap0=overlap0, current_overlap=overlap0, capping_groups=False,
        last_level=False, verbose=False)
    align_begin1, align_end1, align_begin2, align_end2 = index_aln_l
    assert align_begin1 == -overlap0
    assert align_begin2 == 0


@pytest.mark.parametrize("overlap0", [1, 2])
def test_get_residue_indices_cap_offset_when_capped(overlap0):
    '''capping_groups=True keeps the existing e=1 offset unchanged.'''
    index_aln_l, index_clash_l, index_merge_l = get_residue_indices_for_assembly(
        overlap0=overlap0, current_overlap=overlap0, capping_groups=True,
        last_level=False, verbose=False)
    align_begin1, align_end1, align_begin2, align_end2 = index_aln_l
    assert align_begin1 == -(overlap0 + 1)
    assert align_begin2 == 1


def test_strip_cap_independently_controls_each_end():
    '''strip_cap_nterm/strip_cap_cterm must independently gate merge_begin1/merge_end2
    at last_level, so a domain's real terminus can be kept while the free end's
    synthetic cap is still stripped.'''
    _, _, index_merge_l_both = get_residue_indices_for_assembly(
        overlap0=2, current_overlap=2, capping_groups=True, last_level=True,
        verbose=False)
    assert index_merge_l_both[0] == 1   # merge_begin1
    assert index_merge_l_both[-1] == -2  # merge_end2

    _, _, index_merge_l_no_nterm = get_residue_indices_for_assembly(
        overlap0=2, current_overlap=2, capping_groups=True, last_level=True,
        verbose=False, strip_cap_nterm=False, strip_cap_cterm=True)
    assert index_merge_l_no_nterm[0] == 0    # kept: domain's real N-terminal residue
    assert index_merge_l_no_nterm[-1] == -2  # still stripped: free end's synthetic cap

    _, _, index_merge_l_no_cterm = get_residue_indices_for_assembly(
        overlap0=2, current_overlap=2, capping_groups=True, last_level=True,
        verbose=False, strip_cap_nterm=True, strip_cap_cterm=False)
    assert index_merge_l_no_cterm[0] == 1    # still stripped: free end's synthetic cap
    assert index_merge_l_no_cterm[-1] == -1  # kept: domain's real C-terminal residue


def test_strip_cap_defaults_to_capping_groups():
    '''Leaving strip_cap_nterm/strip_cap_cterm at None must reproduce the old,
    capping_groups-only behavior exactly (backward compatibility).'''
    _, _, index_merge_l_default = get_residue_indices_for_assembly(
        overlap0=2, current_overlap=2, capping_groups=True, last_level=True,
        verbose=False)
    _, _, index_merge_l_explicit = get_residue_indices_for_assembly(
        overlap0=2, current_overlap=2, capping_groups=True, last_level=True,
        verbose=False, strip_cap_nterm=True, strip_cap_cterm=True)
    assert index_merge_l_default == index_merge_l_explicit


def test_add_domain_to_fragment_list_n_terminus():
    '''terminus='N' attaches the IDR to the domain's N-terminus, so the domain is
    placed last (after the IDR).'''
    fragment_ids = add_domain_to_fragment_list(n_fragments=4, domain_id='domain', terminus='N')
    assert fragment_ids == [0, 1, 2, 3, 'domain']


def test_add_domain_to_fragment_list_c_terminus():
    '''terminus='C' attaches the IDR to the domain's C-terminus, so the domain is
    placed first (before the IDR).'''
    fragment_ids = add_domain_to_fragment_list(n_fragments=4, domain_id='domain', terminus='C')
    assert fragment_ids == ['domain', 0, 1, 2, 3]


def test_add_domain_to_fragment_list_invalid_terminus():
    with pytest.raises(ValueError):
        add_domain_to_fragment_list(n_fragments=4, domain_id='domain', terminus='middle')


def test_make_hcl_l_with_custom_fragment_ids():
    '''A custom fragment_ids list (e.g. with a domain id spliced in) must be used
    verbatim in place of the default 0..N-1 range, without renumbering.'''
    fragment_ids = add_domain_to_fragment_list(n_fragments=3, domain_id='domain', terminus='C')
    hcg_l, promo_l = make_hcl_l(len(fragment_ids), fragment_ids=fragment_ids)
    # level 1: domain (id 'domain') paired with fragment 0
    assert hcg_l[0][0] == ['domain', 0]


def test_prepare_domain_fragment_rejects_hydrogen_free_pdb(tmp_path):
    '''A folded-domain PDB with no explicit hydrogens (typical of a bare crystal
    structure, cryo-EM model, or structure prediction) must be rejected outright with
    a clear, actionable error -- not silently accepted, which would otherwise surface
    much later as a cryptic MDAnalysis atom-count-mismatch SelectionError deep inside
    a hierarchical_chain_growth run.'''
    u = mda.Universe(os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'))
    no_h_pdb = str(tmp_path / 'no_h_domain.pdb')
    u.select_atoms('not type H').write(no_h_pdb)

    with pytest.raises(ValueError, match='no hydrogen atoms'):
        prepare_domain_fragment(no_h_pdb, str(tmp_path / 'out'))
    assert not (tmp_path / 'out').exists()


def test_prepare_domain_fragment_accepts_hydrogenated_pdb(tmp_path):
    '''Baseline: a normally-hydrogenated structure is still accepted as before.'''
    prepare_domain_fragment(
        os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'), str(tmp_path / 'out'))
    assert (tmp_path / 'out' / 'pair0.pdb').exists()
