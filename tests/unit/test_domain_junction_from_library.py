#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chain_growth.fragment_list import DIMER_LIBRARY_RESIDUE_ORDER, dimer_library_fragment_dir


def test_dimer_library_fragment_dir_index_formula():
    '''folder index = 20*i + j, i/j = positions of the two residues in
    DIMER_LIBRARY_RESIDUE_ORDER -- e.g. the first two entries (GLY-GLY, GLY-ALA) and
    the start of the second row (ALA-GLY).'''
    assert dimer_library_fragment_dir('lib', 'GLY', 'GLY').endswith('/0')
    assert dimer_library_fragment_dir('lib', 'GLY', 'ALA').endswith('/1')
    assert dimer_library_fragment_dir('lib', 'ALA', 'GLY').endswith('/20')


def test_dimer_library_fragment_dir_covers_all_20x20_pairs():
    '''every ordered pair of the 20 standard amino acids must map to a distinct index
    in [0, 400).'''
    indices = set()
    for res_a in DIMER_LIBRARY_RESIDUE_ORDER:
        for res_b in DIMER_LIBRARY_RESIDUE_ORDER:
            path = dimer_library_fragment_dir('lib', res_a, res_b)
            index = int(path.rsplit('/', 1)[-1])
            assert 0 <= index < 400
            indices.add(index)
    assert len(indices) == 400
