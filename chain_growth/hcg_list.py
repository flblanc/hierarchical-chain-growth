#!/usr/bin/env python3

import numpy as np

def pair_fragments(input_ar, verbose = False):
    """ create pairs for each level
    
    Parameter
    ---------
    input_ar : list
        initial list of MD fragments/pairs of paired fragments
        
    Returns
    -------
    new_pairs : list
        new list with paired fragments/pairs of paired fragments
    promotion : boolean
        evaluation if fragment is promoted to the next hierarchy level
        False : two fragments can be paired 
        True : left-over fragment, that cannot be paired, is promoted 
    """
    promotion = False
    ## create list of paired fragments/pairs of paired fragments
    ## as pairs or list of pair for m > 1
    new_pairs = zip(input_ar[::2], input_ar[1::2])
    new_pairs = [list(p) for p in new_pairs]
    
    ## divide len(input_ar) by. If there is remainder (len(input_ar) == uneven) -> promote
    if len(input_ar) % 2:
        if verbose:
            print('promoting')
        ## add fragment/pairs of paired fragments to new_pairs without a partner to pair
        ## to promote it to the next hierarchy level
        promotion = True
        new_pairs.append(input_ar[-1])

    return new_pairs , promotion


def next_power_of_2(N): 
    """ get number of hierarchy levels for chain growth
        -> the next power of 2 from the number of fragments
    
    Parameters
    ----------
    N : integer
        total number of fragments
    
    Returns
    -------
    next power of two for N : integer    
    """
    
    ## calculate number of hierarchy levels m for the number of fragments used
    return 1 if N == 0 else 2**(N - 1).bit_length()

def flatten(pair_list):
    """ flattens list with pairs of fragments / pairs of paired fragments
    
    Parameters
    ----------
    pair_list : list 
        list of lists with paired fragments / pairs of paired fragments
    
    Returns
    -------
    pair_list : list
        flattened lists of pairs
    """
    
    # only recurse into actual nested lists (as built by pair_fragments); checking for
    # collections.abc.Iterable instead would also recurse into a string fragment id
    # (e.g. a folded domain's id) character by character, infinitely
    if isinstance(pair_list, list):
        return [a for i in pair_list for a in flatten(i)]
    else:
        return [pair_list]


def derive_strip_cap_from_domain_position(hcg_l, domain_id):
    """ determine the correct strip_cap_nterm/strip_cap_cterm values for a
    domain-attachment `hierarchical_chain_growth` run, purely from `hcg_l`'s
    structure -- no computation needs to have happened yet.

    `hcg_l[-1]` always collapses to exactly one top-level pair `[left_branch,
    right_branch]` (`make_hcl_l` builds exactly enough levels for a full binary
    reduction), and hierarchical pairing always preserves the original fragment_ids
    order, so `left_branch`/`right_branch` correspond to the N-terminal-most and
    C-terminal-most halves of the assembled chain, respectively. Whichever half
    contains `domain_id` is where the domain's own real terminal residue ends up,
    so that end's strip flag must be False (keep it); the other end is the free
    end's synthetic cap, so its strip flag must be True.

    This removes the need for a caller to manually reason about which physical end
    `domain_id` lands on for a given fragment_ids/terminus choice -- exactly the
    reasoning that went wrong in a real run (see `strip_terminal_caps`'s docstring
    on `hierarchical_chain_growth`).

    Parameters
    ----------
    hcg_l : list
        as returned by `make_hcl_l`
    domain_id : hashable
        the domain's fragment id

    Returns
    -------
    strip_cap_nterm, strip_cap_cterm : boolean, boolean
    """
    left_branch, right_branch = hcg_l[-1][0]
    domain_in_left = domain_id in flatten(left_branch)
    domain_in_right = domain_id in flatten(right_branch)
    if domain_in_left and not domain_in_right:
        return False, True
    elif domain_in_right and not domain_in_left:
        return True, False
    else:
        raise ValueError(
            "could not locate domain_id {!r} on exactly one side of hcg_l's "
            "top-level pair (found in left branch: {}, in right branch: {}) -- "
            "this fragment_ids/hcg_l doesn't look like a valid single domain "
            "attachment.".format(domain_id, domain_in_left, domain_in_right))


def make_hcl_l(N, n_to_c_term = True, fragment_ids = None):
    """ create input list with fragments/ pairs of fragments to assemble in HCG to get the full-length chain

    Parameter
    ---------
    N : integer
        number of fagments
    n_to_c_term : boolean
        direction of growth
        True : from N-terminus to C-terminus
        False : from C-terminus to N-terminus
    fragment_ids : list, optional
        explicit, ordered list of fragment ids (matching MDfragments folder names) to use
        instead of the default `0, 1, ..., N-1`. Must have length N. Lets a fragment with a
        non-consecutive id (e.g. a folded domain attached via `hierarchical_chain_growth`'s
        `domain_id`) be placed at either end without renumbering the other fragments'
        folders. The default is None (uses `0, ..., N-1`, unchanged from before).

    Returns:
    hcg_a[:,0] : list
        list for HCG with lists of fragments assigned to be paired
    hcg_a[:,1] : list
        list with boolean, if here is an promotion in level m (last assembly step)
    """
    next_power2 = next_power_of_2(N)
    # from next power of 2 create max number of assemling levels -> M
    m = np.log2(next_power2).astype(int)
    frag_pair_l = list(fragment_ids) if fragment_ids is not None else list(np.arange(N))
    hcg_l = []
    promo_l =  [] 
    ## loop through number of levels to generate list of paired fragment/ pairs of paired fragments
    ## and get promotion-evaluation
    ## save all pairs + promotion-evaluation per level in hcg_a
    for level_index in range(m):
        ## reverse initial fragment list to grow from C- to N-terminus
        if level_index == 0 and n_to_c_term == False:
            frag_pair_l.reverse()
        frag_pair_l, promo = pair_fragments(frag_pair_l)
        hcg_l.append(frag_pair_l)
        promo_l.append(promo)
        
    return hcg_l, promo_l
    
