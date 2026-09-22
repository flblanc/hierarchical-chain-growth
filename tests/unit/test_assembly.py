#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chain_growth.hcg_list import make_hcl_l, flatten


def test_make_hcl_l_with_promotion():
    '''Check that the fragment assembly schedule is generated as expected. This test
       is based on Fig. S1
       https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.9b00809/suppl_file/ct9b00809_si_001.pdf
    '''
    n_fragments = 46
    hcg_l, promo_l = make_hcl_l(n_fragments)

    ## hcg_l, promo_l index+1 equals level of assembly m.
    ## index 0, level 1: pairs of fragments (level 0 are fragments)

    ## one promotion going from level 1->2
    assert promo_l[1] == True
    ## one promotion going from level 4->5
    assert promo_l[4] == True

    ## the two unequal fragments at the penultimate level
    assert len(flatten(hcg_l[-2][1])) == 14
    assert len(flatten(hcg_l[-2][0])) == 32

    ## one 46-mer of fragments at the last level
    assert len(flatten(hcg_l[-1][0])) == 46

