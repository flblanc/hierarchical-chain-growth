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

from chain_growth.fragment_list import add_domain_to_fragment_list, generate_fragment_list, get_sequence
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


def test_domain_attachment_c_terminus(tmp_path):
    '''Domain placed first, IDR growing off its C-terminal end (its own N-terminal
    residue, ACE, is the real terminus that's kept).'''
    domain_id = 'domain'
    domain_overlap = 2
    n_real_fragments = 14  # fragments 1..14 of the truncated tauK18 example

    # isolated MDfragments tree: fragments 2..14 (symlinked from examples/) + the domain.
    # Fragment 1 is rebuilt without its own leading ACE cap: per the approved design, the
    # domain-facing terminal fragment has no synthetic cap on the domain-facing end (its
    # overlap residues directly continue the domain's real sequence), so its stand-in
    # here must match that, not the unmodified raw fragment (which still has ACE there).
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

    _prepare_domain_fixture(mdfragments_dir / domain_id)

    # overlaps_d for the real (non-domain) fragments; key 0 is only ever read as the
    # run's general default overlap, unrelated to fragment id 0 not being used here
    sequence_f = os.path.join(examples_dir, 'truncated_tauK18.fasta')
    _, overlaps_d = generate_fragment_list(sequence_f, fragment_length=5, overlap=2)

    fragment_ids = [domain_id] + list(range(1, n_real_fragments + 1))
    hcg_l, promo_l = make_hcl_l(len(fragment_ids), fragment_ids=fragment_ids)

    out_dir = tmp_path / 'out'
    hierarchical_chain_growth(
        hcg_l, promo_l, overlaps_d, str(tmp_path), str(out_dir), kmax=5,
        capping_groups=True, domain_id=domain_id, domain_overlap=domain_overlap,
        strip_cap_nterm=False, strip_cap_cterm=True)

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
    hierarchical_chain_growth(
        hcg_l, promo_l, overlaps_d, str(tmp_path), str(out_dir), kmax=5,
        capping_groups=True, domain_id=domain_id, domain_overlap=domain_overlap,
        strip_cap_nterm=True, strip_cap_cterm=False)

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
