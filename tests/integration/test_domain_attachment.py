#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for attaching an IDR to a rigid folded domain, at either terminus.

Since `find_clashes` is reused completely unmodified for the domain's junction (see
`chain_growth.hcg_fct.hierarchical_chain_growth`), and it checks the entire content of
each Universe (not just the local overlap residues), domain-aware clash filtering falls
out automatically at every level once the domain is merged into the growing chain -- a
successful run (every level finds `kmax` clash-free pairs) is itself evidence that this
propagates through the whole hierarchy, not just a final bolt-on step.

There's no independently prepared folded-domain PDB in this repo's example data, so
these tests repurpose existing truncated-tauK18 MD fragments as stand-in rigid domains,
with a synthetic cap removed wherever the approved design calls for real (uncapped)
overlap residues instead, and only the first frame kept (single, rigid conformation).
"""

import os

import MDAnalysis as mda
import pytest

from chain_growth.assembly import DOMAIN_SEGID
from chain_growth.fragment_list import (add_domain_to_fragment_list, generate_fragment_list,
                                         get_sequence, prepare_domain_fragment)
from chain_growth.hcg_fct import hierarchical_chain_growth
from chain_growth.hcg_list import make_hcl_l

test_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(test_dir, '../../examples/')


def _prepare_domain_fixture(out_dir):
    '''Build a rigid, single-frame stand-in "domain" from MD fragment 0's first frame,
    with its trailing NME cap removed so its overlap residues (ALA, PRO) are "real"
    domain residues rather than being followed by a synthetic cap.'''
    fragment0_pdb = os.path.join(examples_dir, 'MDfragments/0/pair0.pdb')
    u = mda.Universe(fragment0_pdb)
    domain_atoms = u.select_atoms('not resname NME')
    os.makedirs(out_dir, exist_ok=True)
    domain_atoms.write(os.path.join(out_dir, 'pair0.pdb'))
    domain_atoms.write(os.path.join(out_dir, 'pair.xtc'), frames='all')


def _build_c_terminus_fragment_tree(tmp_path, domain_id, n_real_fragments=14):
    '''Shared setup for every "domain first, IDR growing off its C-terminal end"
    test below: symlink fragments 2..n_real_fragments from examples/ unmodified,
    and rebuild fragment 1 without its own leading ACE cap (per the approved
    design, the domain-facing terminal fragment has no synthetic cap on the
    domain-facing end). Callers still need to build MDfragments/<domain_id>
    themselves -- how the domain is prepared is what each test is actually about.

    Returns
    -------
    mdfragments_dir, sequence_f, overlaps_d, hcg_l, promo_l
    '''
    mdfragments_dir = tmp_path / 'MDfragments'
    for i in range(2, n_real_fragments + 1):
        src = os.path.join(examples_dir, 'MDfragments', str(i))
        dst = mdfragments_dir / str(i)
        os.makedirs(dst)
        for fname in ('pair0.pdb', 'pair.xtc'):
            os.symlink(os.path.join(src, fname), dst / fname)

    fragment1_dst = mdfragments_dir / '1'
    os.makedirs(fragment1_dst)
    u_fragment1 = mda.Universe(os.path.join(examples_dir, 'MDfragments/1/pair0.pdb'))
    fragment1_atoms = u_fragment1.select_atoms('not resname ACE')
    fragment1_atoms.write(str(fragment1_dst / 'pair0.pdb'))
    fragment1_atoms.write(str(fragment1_dst / 'pair.xtc'), frames='all')

    # overlaps_d for the real (non-domain) fragments; key 0 is only ever read as the
    # run's general default overlap, unrelated to fragment id 0 not being used here
    sequence_f = os.path.join(examples_dir, 'truncated_tauK18.fasta')
    _, overlaps_d = generate_fragment_list(sequence_f, fragment_length=5, overlap=2)

    fragment_ids = [domain_id] + list(range(1, n_real_fragments + 1))
    hcg_l, promo_l = make_hcl_l(len(fragment_ids), fragment_ids=fragment_ids)

    return mdfragments_dir, sequence_f, overlaps_d, hcg_l, promo_l


def test_domain_attachment_c_terminus(tmp_path):
    '''Domain placed first, IDR growing off its C-terminal end (its own N-terminal
    residue, ACE, is the real terminus that's kept).'''
    domain_id = 'domain'
    domain_overlap = 2

    mdfragments_dir, sequence_f, overlaps_d, hcg_l, promo_l = _build_c_terminus_fragment_tree(
        tmp_path, domain_id)
    _prepare_domain_fixture(mdfragments_dir / domain_id)

    out_dir = tmp_path / 'out'
    # strip_terminal_caps=False: this fixture's "domain" is a stand-in built from a
    # real, still-capped MD fragment (see _prepare_domain_fixture), so its own kept
    # terminal residue is literally resname ACE -- a fixture-only coincidence real
    # domains won't have (a true domain terminus is never named ACE/NME). Disabled
    # here so that coincidence doesn't collide with the name-based safety net this
    # test isn't about; strip_terminal_caps gets its own dedicated tests.
    hierarchical_chain_growth(
        hcg_l, promo_l, overlaps_d, str(tmp_path), str(out_dir), kmax=5,
        capping_groups=True, domain_id=domain_id, domain_overlap=domain_overlap,
        strip_cap_nterm=False, strip_cap_cterm=True, strip_terminal_caps=False)

    # level 1: domain (kept: ACE,LEU,GLN,THR) merged with fragment 1 (kept: ALA,PRO,VAL,PRO,MET,NME)
    level1_pdb = out_dir / '1' / domain_id / 'pair0.pdb'
    assert level1_pdb.exists()
    u_level1 = mda.Universe(str(level1_pdb))
    assert list(u_level1.residues.resnames) == [
        'ACE', 'LEU', 'GLN', 'THR', 'ALA', 'PRO', 'VAL', 'PRO', 'MET', 'NME']

    # final level: domain's real N-terminal residue (ACE, standing in for a genuine
    # domain terminus) is kept; the free C-terminal end's synthetic NME cap is stripped
    final_level = len(hcg_l)
    final_pdb = out_dir / str(final_level) / domain_id / 'pair0.pdb'
    assert final_pdb.exists()
    u_final = mda.Universe(str(final_pdb))
    resnames_final = list(u_final.residues.resnames)
    assert resnames_final[0] == 'ACE'
    assert 'NME' not in resnames_final

    # excluding the kept domain "cap" residue, the real sequence must match the
    # reference tauK18 sequence exactly, the same check test_hcg.py's test_sequence uses
    sequence_hcg = get_sequence(str(final_pdb), NA=False).get_sequence_list()[1:]
    sequence_ref = get_sequence(sequence_f, NA=False).get_sequence_list()
    assert sequence_hcg == sequence_ref


def test_domain_attachment_unifies_segid_only_at_final_level(tmp_path):
    '''Once the domain is merged into the growing chain, DOMAIN_SEGID must stay on
    the domain's own atoms at every intermediate level (find_clashes relies on it
    for the domain-surface-only clash-check optimization at every subsequent
    level), and only get erased -- so the final output looks like one single,
    covalently continuous molecule rather than two -- once assembly is actually
    finished.'''
    domain_id = 'domain'
    domain_overlap = 2

    mdfragments_dir, sequence_f, overlaps_d, hcg_l, promo_l = _build_c_terminus_fragment_tree(
        tmp_path, domain_id)
    assert len(hcg_l) > 1  # this test needs a real intermediate level to check

    # unlike _prepare_domain_fixture (used by the other tests in this file), this
    # goes through the real prepare_domain_fragment, so the domain's atoms actually
    # get tagged with DOMAIN_SEGID -- required to exercise the relabeling this test
    # is about
    fragment0_pdb = os.path.join(examples_dir, 'MDfragments/0/pair0.pdb')
    u_fragment0 = mda.Universe(fragment0_pdb)
    domain_src_pdb = tmp_path / 'domain_src.pdb'
    u_fragment0.select_atoms('not resname NME').write(str(domain_src_pdb))
    prepare_domain_fragment(str(domain_src_pdb), str(mdfragments_dir / domain_id))

    out_dir = tmp_path / 'out'
    hierarchical_chain_growth(
        hcg_l, promo_l, overlaps_d, str(tmp_path), str(out_dir), kmax=5,
        capping_groups=True, domain_id=domain_id, domain_overlap=domain_overlap,
        strip_cap_nterm=False, strip_cap_cterm=True)

    level1_pdb = out_dir / '1' / domain_id / 'pair0.pdb'
    u_level1 = mda.Universe(str(level1_pdb))
    assert DOMAIN_SEGID in set(u_level1.atoms.segids)
    assert len(set(u_level1.atoms.segids)) == 2

    final_level = len(hcg_l)
    final_pdb = out_dir / str(final_level) / domain_id / 'pair0.pdb'
    u_final = mda.Universe(str(final_pdb))
    assert DOMAIN_SEGID not in set(u_final.atoms.segids)
    assert len(set(u_final.atoms.segids)) == 1
    assert len(set(u_final.atoms.chainIDs)) == 1


def test_domain_attachment_n_terminus(tmp_path):
    '''Domain placed last, IDR growing off its N-terminal end (its own C-terminal
    residue, NME, is the real terminus that's kept).'''
    domain_id = 'domain'
    domain_overlap = 2
    n_real_fragments = 15  # fragments 0..14 of the truncated tauK18 example

    # isolated MDfragments tree: fragments 0..13 (symlinked from examples/) + a
    # modified fragment 14 + the domain.
    mdfragments_dir = tmp_path / 'MDfragments'
    for i in range(n_real_fragments - 1):
        src = os.path.join(examples_dir, 'MDfragments', str(i))
        dst = mdfragments_dir / str(i)
        os.makedirs(dst)
        for fname in ('pair0.pdb', 'pair.xtc'):
            os.symlink(os.path.join(src, fname), dst / fname)

    # fragment 14 (raw: ACE,SER,ASN,VAL,GLN,SER,NME) is the domain-facing terminal
    # fragment here: its own trailing NME cap is stripped, per the approved design.
    fragment14_pdb = os.path.join(examples_dir, 'MDfragments/14/pair0.pdb')
    u_fragment14 = mda.Universe(fragment14_pdb)
    fragment14_dst = mdfragments_dir / '14'
    os.makedirs(fragment14_dst)
    fragment14_atoms = u_fragment14.select_atoms('not resname NME')
    fragment14_atoms.write(str(fragment14_dst / 'pair0.pdb'))
    fragment14_atoms.write(str(fragment14_dst / 'pair.xtc'), frames='all')

    # the domain: sliced from fragment 14's own last 3 residues (GLN, SER, NME), so its
    # leading overlap residues (GLN, SER) are identical to fragment 14's real trailing
    # residues by construction -- exactly how two real overlapping fragments would share
    # identical residues -- with fragment 14's own leading ACE/SER/ASN/VAL dropped, so
    # there's no leading cap on the domain's overlap-facing end either.
    domain_atoms = u_fragment14.select_atoms('resid 5:7')
    domain_dst = mdfragments_dir / domain_id
    os.makedirs(domain_dst)
    domain_atoms.write(str(domain_dst / 'pair0.pdb'))
    domain_atoms.write(str(domain_dst / 'pair.xtc'), frames='all')

    sequence_f = os.path.join(examples_dir, 'truncated_tauK18.fasta')
    _, overlaps_d = generate_fragment_list(sequence_f, fragment_length=5, overlap=2)

    fragment_ids = list(range(n_real_fragments)) + [domain_id]
    hcg_l, promo_l = make_hcl_l(len(fragment_ids), fragment_ids=fragment_ids)

    out_dir = tmp_path / 'out'
    # strip_terminal_caps=False: this fixture's "domain" is sliced from a real,
    # still-capped MD fragment (see above), so its own kept terminal residue is
    # literally resname NME -- a fixture-only coincidence real domains won't have (a
    # true domain terminus is never named ACE/NME). Disabled here so that
    # coincidence doesn't collide with the name-based safety net this test isn't
    # about; strip_terminal_caps gets its own dedicated tests.
    hierarchical_chain_growth(
        hcg_l, promo_l, overlaps_d, str(tmp_path), str(out_dir), kmax=5,
        capping_groups=True, domain_id=domain_id, domain_overlap=domain_overlap,
        strip_cap_nterm=True, strip_cap_cterm=False, strip_terminal_caps=False)

    # level 1: fragment 14 (kept: ACE,SER,ASN,VAL,GLN) merged with domain (kept: SER,NME)
    # -- together exactly reconstituting fragment 14's original, unmodified sequence
    level1_pdb = out_dir / '1' / '14' / 'pair0.pdb'
    assert level1_pdb.exists()
    u_level1 = mda.Universe(str(level1_pdb))
    assert list(u_level1.residues.resnames) == [
        'ACE', 'SER', 'ASN', 'VAL', 'GLN', 'SER', 'NME']

    # final level: the free N-terminal end's synthetic ACE cap is stripped; the domain's
    # real C-terminal residue (NME, standing in for a genuine domain terminus) is kept
    final_level = len(hcg_l)
    final_pdb = out_dir / str(final_level) / '0' / 'pair0.pdb'
    assert final_pdb.exists()
    u_final = mda.Universe(str(final_pdb))
    resnames_final = list(u_final.residues.resnames)
    assert resnames_final[-1] == 'NME'
    assert 'ACE' not in resnames_final

    # excluding the kept domain "cap" residue, the real sequence must match the
    # reference tauK18 sequence exactly, the same check test_hcg.py's test_sequence uses
    sequence_hcg = get_sequence(str(final_pdb), NA=False).get_sequence_list()[:-1]
    sequence_ref = get_sequence(sequence_f, NA=False).get_sequence_list()
    assert sequence_hcg == sequence_ref


def test_misconfigured_strip_cap_args_raise_immediately(tmp_path):
    '''Reproduces a real bug found in practice: a domain-attachment run with
    strip_cap_nterm/strip_cap_cterm set for the wrong physical end (backwards
    relative to where the domain actually sits in fragment_ids) silently deleted
    the domain's real terminal residue while leaving a real ACE/NME cap in the
    "final" output. hierarchical_chain_growth now derives the correct values from
    hcg_l automatically and raises immediately if an explicit value disagrees,
    catching this exact mistake before any computation runs (rather than relying
    on strip_terminal_caps to clean up a leftover cap afterward, which can't undo
    a real residue wrongly deleted the other way).'''
    domain_id = 'domain'
    domain_overlap = 2

    mdfragments_dir, sequence_f, overlaps_d, hcg_l, promo_l = _build_c_terminus_fragment_tree(
        tmp_path, domain_id)
    _prepare_domain_fixture(mdfragments_dir / domain_id)

    # domain is FIRST here, so the correct values are strip_cap_nterm=False (keep
    # the domain's own real residue) and strip_cap_cterm=True (strip the free
    # end's cap) -- deliberately backwards below, exactly reproducing the real
    # mistake (which used the *other* orientation's values unswapped)
    with pytest.raises(ValueError, match='conflicts with where domain_id'):
        hierarchical_chain_growth(
            hcg_l, promo_l, overlaps_d, str(tmp_path), str(tmp_path / 'out'), kmax=5,
            capping_groups=True, domain_id=domain_id, domain_overlap=domain_overlap,
            strip_cap_nterm=True, strip_cap_cterm=False)
    assert not (tmp_path / 'out').exists()
