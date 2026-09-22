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


def test_prepare_domain_fragment_rejects_nonstandard_hydrogen_naming(tmp_path):
    '''Some protonation tools name hydrogens generically (e.g. sequentially numbered
    "H01", "H02", ...) instead of the standard PDB/AMBER "H" for the backbone amide
    proton that hierarchical_chain_growth's alignment hardcodes. Having *some*
    hydrogens isn't enough -- this must be rejected too, with a message that points at
    the actual naming problem (a real, reproduced case: a domain PDB re-protonated
    this way still crashed with the same atom-count-mismatch SelectionError as having
    no hydrogens at all).'''
    u = mda.Universe(os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'))
    renamed_pdb = str(tmp_path / 'nonstandard_h_domain.pdb')
    hydrogens = u.select_atoms('type H')
    hydrogens.names = ['H{:02d}'.format(i) for i in range(len(hydrogens))]
    u.atoms.write(renamed_pdb)

    with pytest.raises(ValueError, match='none literally named "H"'):
        prepare_domain_fragment(renamed_pdb, str(tmp_path / 'out'))
    assert not (tmp_path / 'out').exists()


def test_prepare_domain_fragment_accepts_hydrogenated_pdb(tmp_path):
    '''Baseline: a normally-hydrogenated structure is still accepted as before.'''
    prepare_domain_fragment(
        os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'), str(tmp_path / 'out'))
    assert (tmp_path / 'out' / 'pair0.pdb').exists()


def test_prepare_domain_fragment_tags_domain_segid(tmp_path):
    '''Every prepared domain fragment is tagged with DOMAIN_SEGID unconditionally
    (not just when surface_mask is given), so find_clashes's domain-surface
    optimization can be enabled later without re-preparing the fragment.'''
    from chain_growth.hcg_fct import DOMAIN_SEGID
    prepare_domain_fragment(
        os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'), str(tmp_path / 'out'))
    u = mda.Universe(str(tmp_path / 'out' / 'pair0.pdb'))
    assert set(u.atoms.segids) == {DOMAIN_SEGID}


def test_prepare_domain_fragment_without_surface_mask_defaults_to_all_kept(tmp_path):
    '''The critical safety property: preparing a domain WITHOUT surface_mask (i.e.
    every existing caller, and the default) must leave every atom's tempfactor at
    its natural default (0.0) -- find_clashes only excludes tempfactor > 0.5, so
    this must mean "nothing excluded", identical to pre-existing behavior. Getting
    the encoding backwards here would make every domain invisible to clash-checking
    whenever this new, optional feature isn't used -- a silent, severe regression.'''
    prepare_domain_fragment(
        os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'), str(tmp_path / 'out'))
    u = mda.Universe(str(tmp_path / 'out' / 'pair0.pdb'))
    assert list(u.atoms.tempfactors) == [0.0] * len(u.atoms)


def test_prepare_domain_fragment_with_surface_mask_bakes_in_tempfactors(tmp_path):
    '''surface_mask=True (exposed) -> tempfactor 0.0 (kept); surface_mask=False
    (buried) -> tempfactor 1.0 (excluded by find_clashes) -- the reverse of the
    boolean, since "buried" must be the non-default value (see the test above).'''
    import numpy as np
    u_orig = mda.Universe(os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'))
    n = len(u_orig.atoms)
    mask = np.array([i % 2 == 0 for i in range(n)])  # alternate exposed/buried

    prepare_domain_fragment(
        os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'), str(tmp_path / 'out'),
        surface_mask=mask)
    u = mda.Universe(str(tmp_path / 'out' / 'pair0.pdb'))
    expected = np.where(mask, 0.0, 1.0)
    assert np.array_equal(np.array(u.atoms.tempfactors), expected)


def test_prepare_domain_fragment_rejects_wrong_length_surface_mask(tmp_path):
    import numpy as np
    with pytest.raises(ValueError, match='surface_mask has'):
        prepare_domain_fragment(
            os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'), str(tmp_path / 'out'),
            surface_mask=np.array([True, False]))  # wrong length


def test_compute_domain_surface_mask_requires_freesasa(monkeypatch):
    '''A clear ImportError, not a cryptic one, when freesasa isn't installed.'''
    import builtins
    from chain_growth.fragment_list import compute_domain_surface_mask
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'freesasa':
            raise ImportError("no module named freesasa")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ImportError, match='pip install freesasa'):
        compute_domain_surface_mask(os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'))


def test_compute_domain_surface_mask_classifies_real_domain():
    '''On a real, compact folded domain, a meaningful fraction of heavy atoms should
    be classified as buried (not all-exposed, not all-buried) -- a basic sanity
    check that the freesasa integration is actually working, not just returning a
    constant.'''
    pytest.importorskip('freesasa')
    from chain_growth.fragment_list import compute_domain_surface_mask
    mask = compute_domain_surface_mask(os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'))
    u = mda.Universe(os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'))
    assert len(mask) == len(u.atoms)
    heavy = u.select_atoms('not (name H* or name [123]H*)')
    heavy_mask = mask[heavy.indices]
    assert 0 < heavy_mask.sum() < len(heavy_mask)  # neither all-exposed nor all-buried
    # hydrogens are never classified buried (SASA is computed on heavy atoms only)
    hydrogens = u.select_atoms('name H* or name [123]H*')
    assert mask[hydrogens.indices].all()


def test_find_clashes_excludes_marked_buried_domain_atoms(tmp_path):
    '''The actual integration point: find_clashes must drop atoms explicitly marked
    buried (segid DOMAIN_SEGID, tempfactor > 0.5), and must be a complete no-op
    (checks every atom, exactly like before this feature existed) for a domain
    prepared without a surface_mask.'''
    import numpy as np
    from chain_growth.hcg_fct import find_clashes

    u1_pdb = os.path.join(examples_dir, 'MDfragments/0/pair0.pdb')
    u2_pdb = os.path.join(examples_dir, 'MDfragments/1/pair0.pdb')

    # unmasked domain: find_clashes result must match a plain (no DOMAIN_SEGID
    # involved) clash check exactly
    prepare_domain_fragment(u1_pdb, str(tmp_path / 'unmasked'))
    u1_unmasked = mda.Universe(str(tmp_path / 'unmasked' / 'pair0.pdb'))
    u2 = mda.Universe(u2_pdb)
    clashes_unmasked = find_clashes(u1_unmasked, u2, index1b=-3, index2e=2)

    u1_plain = mda.Universe(u1_pdb)  # never went through prepare_domain_fragment at all
    u2_plain = mda.Universe(u2_pdb)
    clashes_plain = find_clashes(u1_plain, u2_plain, index1b=-3, index2e=2)
    assert clashes_unmasked == clashes_plain

    # fully-buried mask: every one of u1's own atoms gets excluded, so nothing can
    # possibly clash against u2 regardless of geometry
    n = len(u1_plain.atoms)
    all_buried = np.zeros(n, dtype=bool)
    prepare_domain_fragment(u1_pdb, str(tmp_path / 'all_buried'), surface_mask=all_buried)
    u1_all_buried = mda.Universe(str(tmp_path / 'all_buried' / 'pair0.pdb'))
    clashes_all_buried = find_clashes(u1_all_buried, u2, index1b=-3, index2e=2)
    assert clashes_all_buried == 0


def test_fragment_assembly_raises_on_single_frame_deterministic_failure(monkeypatch):
    '''When both fragments have exactly one frame (e.g. a domain-junction fragment
    built with too small a kmax, joined to the always-rigid, single-frame domain),
    there is only ever one possible alignment/clash trial. If it fails, retrying it
    forever cannot ever succeed -- fragment_assembly must raise immediately instead
    of hanging, rather than spinning in its rejection-sampling loop forever.'''
    import chain_growth.hcg_fct as hcg_fct

    u1 = mda.Universe(os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'))
    u2 = mda.Universe(os.path.join(examples_dir, 'MDfragments/1/pair0.pdb'))
    assert u1.trajectory.n_frames == 1 and u2.trajectory.n_frames == 1

    # deterministically force the single possible trial to fail, regardless of the
    # two fragments' real geometry -- this test is about the guard logic, not
    # whether these two particular fragments happen to clash
    monkeypatch.setattr(hcg_fct, '_attempt_merge', lambda *a, **k: None)

    with pytest.raises(ValueError, match='only one possible alignment/clash trial'):
        hcg_fct.fragment_assembly(u1, u2, dire=str('unused'), select={}, index_clash_l=[0, 0],
                                   index_merge_l=[0, -1, 0, -1], rmsd_cut_off=0.6,
                                   clash_distance=2.0, kmax=1)


def test_reweighted_fragment_assembly_raises_on_single_frame_deterministic_failure(monkeypatch):
    '''Same guard as fragment_assembly (see above), for the reweighted/importance-
    sampling code path -- it has the identical single-possible-trial hang risk.'''
    import numpy as np
    import chain_growth.rhcg_fct as rhcg_fct

    u1 = mda.Universe(os.path.join(examples_dir, 'MDfragments/0/pair0.pdb'))
    u2 = mda.Universe(os.path.join(examples_dir, 'MDfragments/1/pair0.pdb'))
    assert u1.trajectory.n_frames == 1 and u2.trajectory.n_frames == 1

    monkeypatch.setattr(rhcg_fct, '_attempt_merge', lambda *a, **k: None)

    with pytest.raises(ValueError, match='only one possible alignment/clash trial'):
        rhcg_fct.reweighted_fragment_assembly(
            u1, u2, dire=str('unused'), select={}, index_clash_l=[0, 0],
            index_merge_l=[0, -1, 0, -1], rmsd_cut_off=0.6, clash_distance=2.0,
            kmax=1, w_l=[np.array([1.0]), np.array([1.0])])


def test_unify_domain_segid_merges_domain_and_idr_into_one(tmp_path):
    '''Once the domain and the grown IDR are merged, they're a single, covalently
    continuous molecule -- _unify_domain_segid must erase DOMAIN_SEGID's internal
    bookkeeping tag (and the domain's separate chainID) so the output reflects
    that, rather than looking like two separate molecules/chains.'''
    from chain_growth.assembly import DOMAIN_SEGID, _unify_domain_segid, merge_universe

    domain_pdb = os.path.join(examples_dir, 'MDfragments/0/pair0.pdb')
    idr_pdb = os.path.join(examples_dir, 'MDfragments/1/pair0.pdb')
    prepare_domain_fragment(domain_pdb, str(tmp_path / 'domain'))

    u1 = mda.Universe(str(tmp_path / 'domain' / 'pair0.pdb'))
    u2 = mda.Universe(idr_pdb)
    u = merge_universe(u1, u2, 0, -1, 0, -1)

    segids_before = set(u.atoms.segids)
    assert DOMAIN_SEGID in segids_before and len(segids_before) == 2
    assert len(set(u.atoms.chainIDs)) == 2

    _unify_domain_segid(u, DOMAIN_SEGID)

    segids_after = set(u.atoms.segids)
    assert len(segids_after) == 1 and DOMAIN_SEGID not in segids_after
    assert len(set(u.atoms.chainIDs)) == 1


def test_unify_domain_segid_noop_without_domain_atoms():
    '''Called on a universe with no DOMAIN_SEGID-tagged atoms at all (e.g. an
    ordinary, domain-free HCG run), _unify_domain_segid must do nothing rather than
    guess -- there's no "other" value to unify towards.'''
    from chain_growth.assembly import DOMAIN_SEGID, _unify_domain_segid

    u = mda.Universe(os.path.join(examples_dir, 'MDfragments/1/pair0.pdb'))
    segids_before = list(u.atoms.segids)
    chainids_before = list(u.atoms.chainIDs)

    _unify_domain_segid(u, DOMAIN_SEGID)

    assert list(u.atoms.segids) == segids_before
    assert list(u.atoms.chainIDs) == chainids_before


def test_attempt_merge_relabels_domain_segid_only_when_requested(tmp_path):
    '''_attempt_merge must leave DOMAIN_SEGID intact by default (every intermediate
    HCG level relies on it for find_clashes's domain-surface-only optimization),
    and only relabel when explicitly asked via relabel_domain_segid -- the caller's
    signal that this is the truly final merge.'''
    from chain_growth.assembly import (DOMAIN_SEGID, _attempt_merge, get_residue_indices_for_assembly,
                                        translate_concept)

    domain_pdb = os.path.join(examples_dir, 'MDfragments/0/pair0.pdb')
    idr_pdb = os.path.join(examples_dir, 'MDfragments/1/pair0.pdb')
    prepare_domain_fragment(domain_pdb, str(tmp_path / 'domain'))

    index_aln_l, index_clash_l, index_merge_l = get_residue_indices_for_assembly(
        overlap0=2, current_overlap=2, capping_groups=True, last_level=False, verbose=False)

    def fresh_pair():
        u1 = mda.Universe(str(tmp_path / 'domain' / 'pair0.pdb'))
        u2 = mda.Universe(idr_pdb)
        select = translate_concept(u1, u2, False, *index_aln_l)
        return u1, u2, select

    # rmsd_cut_off deliberately huge and clash_distance deliberately tiny, so this
    # trial is guaranteed to pass regardless of these two fragments' real geometry
    # -- this test is about the relabeling behavior, not the accept/reject criteria
    u1, u2, select = fresh_pair()
    u_default = _attempt_merge(u1, u2, select, index_clash_l, index_merge_l, 999.0, 0.001)
    assert DOMAIN_SEGID in set(u_default.atoms.segids)

    u1, u2, select = fresh_pair()
    u_relabeled = _attempt_merge(u1, u2, select, index_clash_l, index_merge_l, 999.0, 0.001,
                                 relabel_domain_segid=DOMAIN_SEGID)
    assert DOMAIN_SEGID not in set(u_relabeled.atoms.segids)


def test_strip_terminal_caps_removes_leading_ace_and_trailing_nme():
    '''The real-world motivation: a leftover cap can end up in a "final" model when
    strip_cap_nterm/strip_cap_cterm (which work by residue position) are configured
    for the wrong physical end -- confirmed in practice for a real domain-attachment
    run, which silently deleted the domain's own real terminal residue while leaving
    the free end's ACE cap in place. _strip_terminal_caps checks residue identity
    instead, so it's unaffected by that kind of mistake.'''
    from chain_growth.assembly import _strip_terminal_caps

    u = mda.Universe(os.path.join(examples_dir, 'MDfragments/1/pair0.pdb'))
    assert list(u.residues.resnames) == ['ACE', 'ALA', 'PRO', 'VAL', 'PRO', 'MET', 'NME']

    u_stripped = _strip_terminal_caps(u)

    assert list(u_stripped.residues.resnames) == ['ALA', 'PRO', 'VAL', 'PRO', 'MET']
    assert list(u_stripped.residues.resids) == [1, 2, 3, 4, 5]


def test_strip_terminal_caps_noop_without_a_cap():
    '''An already-correctly-stripped chain (neither end named ACE/NME, e.g. because
    strip_cap_nterm/strip_cap_cterm already did their job, or a domain's own real
    terminal residue) must be returned untouched.'''
    from chain_growth.assembly import _strip_terminal_caps

    u = mda.Universe(os.path.join(examples_dir, 'MDfragments/1/pair0.pdb'))
    internal_only = mda.core.universe.Merge(u.select_atoms('resid 2:6'))
    resnames_before = list(internal_only.residues.resnames)
    assert 'ACE' not in resnames_before and 'NME' not in resnames_before

    result = _strip_terminal_caps(internal_only)

    assert result is internal_only
    assert list(result.residues.resnames) == resnames_before


def test_attempt_merge_strips_terminal_caps_only_when_requested(tmp_path):
    '''_attempt_merge must leave a leftover cap in place by default, and only strip
    it when explicitly asked via strip_terminal_caps -- the caller's signal that
    this is the truly final merge.'''
    from chain_growth.assembly import (_attempt_merge, get_residue_indices_for_assembly,
                                        translate_concept)

    domain_pdb = os.path.join(examples_dir, 'MDfragments/0/pair0.pdb')
    idr_pdb = os.path.join(examples_dir, 'MDfragments/1/pair0.pdb')
    prepare_domain_fragment(domain_pdb, str(tmp_path / 'domain'))

    index_aln_l, index_clash_l, index_merge_l = get_residue_indices_for_assembly(
        overlap0=2, current_overlap=2, capping_groups=True, last_level=False, verbose=False)

    def fresh_pair():
        u1 = mda.Universe(str(tmp_path / 'domain' / 'pair0.pdb'))
        u2 = mda.Universe(idr_pdb)
        select = translate_concept(u1, u2, False, *index_aln_l)
        return u1, u2, select

    # rmsd_cut_off deliberately huge and clash_distance deliberately tiny, so this
    # trial is guaranteed to pass regardless of these two fragments' real geometry
    u1, u2, select = fresh_pair()
    u_default = _attempt_merge(u1, u2, select, index_clash_l, index_merge_l, 999.0, 0.001)
    assert u_default.residues[0].resname == 'ACE'

    u1, u2, select = fresh_pair()
    u_stripped = _attempt_merge(u1, u2, select, index_clash_l, index_merge_l, 999.0, 0.001,
                                strip_terminal_caps=True)
    assert u_stripped.residues[0].resname != 'ACE'


def test_derive_strip_cap_from_domain_position():
    '''The domain occupying the left (N-terminal) half of hcg_l's top-level pair
    means its real residue is at the very start -- keep it (strip_cap_nterm=False)
    -- and the free end's cap is at the very end -- strip it (strip_cap_cterm=True).
    Domain on the right is the mirror image.'''
    from chain_growth.hcg_list import derive_strip_cap_from_domain_position, make_hcl_l

    hcg_l, _ = make_hcl_l(5, fragment_ids=['domain', 0, 1, 2, 3])
    assert derive_strip_cap_from_domain_position(hcg_l, 'domain') == (False, True)

    hcg_l, _ = make_hcl_l(5, fragment_ids=[0, 1, 2, 3, 'domain'])
    assert derive_strip_cap_from_domain_position(hcg_l, 'domain') == (True, False)


def test_hierarchical_chain_growth_raises_on_conflicting_strip_cap_args(tmp_path):
    '''The exact real bug this derivation prevents: an explicit strip_cap_nterm/
    strip_cap_cterm that disagrees with where domain_id actually ends up must raise
    immediately -- before any assembly work happens -- rather than silently
    stripping the domain's real terminal residue.'''
    from chain_growth.hcg_fct import hierarchical_chain_growth
    from chain_growth.hcg_list import make_hcl_l

    hcg_l, promo_l = make_hcl_l(5, fragment_ids=['domain', 0, 1, 2, 3])
    # domain is first, so strip_cap_nterm should be False -- passing True conflicts
    with pytest.raises(ValueError, match='conflicts with where domain_id'):
        hierarchical_chain_growth(
            hcg_l, promo_l, overlaps_d={0: 2}, path0='unused', path=str(tmp_path / 'out'),
            kmax=1, domain_id='domain', domain_overlap=2, strip_cap_nterm=True)
    assert not (tmp_path / 'out').exists()


def test_hierarchical_chain_growth_leaves_matching_strip_cap_args_untouched(tmp_path):
    '''An explicit value that already matches the derived one must be accepted
    without complaint -- this validation is a safety check, not a ban on passing
    strip_cap_nterm/strip_cap_cterm explicitly.'''
    from chain_growth.hcg_fct import hierarchical_chain_growth
    from chain_growth.hcg_list import derive_strip_cap_from_domain_position, make_hcl_l

    hcg_l, promo_l = make_hcl_l(5, fragment_ids=['domain', 0, 1, 2, 3])
    derived_nterm, derived_cterm = derive_strip_cap_from_domain_position(hcg_l, 'domain')

    domain_pdb = os.path.join(examples_dir, 'MDfragments/0/pair0.pdb')
    prepare_domain_fragment(domain_pdb, str(tmp_path / 'MDfragments' / 'domain'))
    # the run will still fail downstream (no real MDfragments/0..3 set up here) --
    # this test only cares that it gets *past* the strip_cap validation itself
    with pytest.raises(Exception) as exc_info:
        hierarchical_chain_growth(
            hcg_l, promo_l, overlaps_d={0: 2}, path0=str(tmp_path), path=str(tmp_path / 'out'),
            kmax=1, domain_id='domain', domain_overlap=2,
            strip_cap_nterm=derived_nterm, strip_cap_cterm=derived_cterm)
    assert 'conflicts with where domain_id' not in str(exc_info.value)
