#!/usr/bin/env python3
"""
rhcg_fct
--------
reweighted hierarchical chain growth (RHCG): assembles fragments (reweighted
according to experimental data) into the full-length chain, using importance
sampling. See `chain_growth.hcg_fct` for plain (non-reweighted) HCG; the two share
the per-pair geometry/selection helpers in `chain_growth.assembly`.

Unlike `hierarchical_chain_growth`, this always runs serially (not yet
parallelized with multiprocessing.Pool).
"""
import numpy as np
import MDAnalysis as mda
import pathlib, shutil, os
from tqdm import tqdm
from chain_growth.assembly import (
    DOMAIN_SEGID, translate_concept, get_residue_indices_for_assembly, _resolve_old_pairs,
    _resolve_pair_overlap, _is_proline_at_alignment_end, _attempt_merge)


def reweighted_fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
         rmsd_cut_off, clash_distance, kmax, w_l, ri_l=None, draw_indices=True,
         chain_weights_prev_l=None, relabel_domain_segid=None, strip_terminal_caps=False):
    """ assemble the fragments to pairs
    
    Parameters
    ----------
    u1 : universe
        fragment 1 to pair
    u2 : universe
        fragment 1 to pair
    dire : path
        path to store assembled pair in
    select : dictionary 
        dictionary with resids and atoms
        of fragment 1 and 2 for alignment
    index_clash_l : list
        list with residue indices of fragment 1 and 2 for clash detection
        residues that should be excluded from neighbour search
    index_merge_l : list
        list with residue indices of fragment 1 and 2 for the final assemby
    rmsd_cut_off : float
        cut-off for the RMSD of the fragment alignment
    clash_distance : float
        max. allowed distance between atoms
    kmax : integer
        number of pairs that should be assembled in level m_i
    w_l : list or array-like
        list with weight arrays per for the fragments. The weights define how likely it is to observe
        configuration a (frame a) in comparison to the native ensemble. These weights are
        used to perform weighted drawing, i.e., draw weighted random frames.
        Weights per frame come either from ensemble reweighting (in level m == 1)
        or can be uniform (in level m > 1).
    ri_l : array-like
        array with indices for chosing a specific confoormation of a fragment. 
        if None: draw indices randomly
    draw_indices : booolean
        if new random integers == frame indices are drawn or else taken from a input array.
    chain_weights_prev_l : list or array-like
        array with stored product of weights cW1 * cW2 of assembled fragments / pairs from previous level
        eq 5 in stelzl at al JACS Au 2022
        DO NOT USE as fragment weight to draw random frame - not yet normalized!
    relabel_domain_segid : string, optional
        see `chain_growth.assembly._attempt_merge`; pass `DOMAIN_SEGID` only for the
        truly final merge of a domain-attachment run. The default is None.
    strip_terminal_caps : boolean, optional
        see `chain_growth.assembly._attempt_merge`; pass True only for the truly
        final merge. The default is False.


    Returns
    -------
    if draw_indices:
        rs : list of lists
            successful indices drawn for fragment 1 and 2 during fragment assembly
        assembled_chain_weights : array
            array with stored product of weights cW1 * cW2 of assembled fragments / pairs from current level
    else:
        assembled_chain_weights : array
            array with stored product of weights cW1 * cW2 of assembled fragments / pairs from current level
        
    else:
        None
    """
    
    k = 0
    # array to store product of weights cW1 * cW2 of assembled fragments / pairs from current level
    assembled_chain_weights = np.zeros(kmax)
    
    if ri_l is None:
        # array to store random frame indices of successfully assembled fragments / pairs
        rs = np.zeros((kmax, 2))

    # when both fragments have exactly one frame, there is only ever one possible
    # (r1, r2) combination -- every iteration of the loop below would draw frame 0
    # from each and repeat the exact same, deterministic trial forever. If that one
    # trial doesn't pass, retrying cannot ever succeed, so fail fast with a clear
    # error instead of spinning indefinitely (see fragment_assembly in hcg_fct.py
    # for the same guard, and its full rationale)
    if draw_indices and u1.trajectory.n_frames == 1 and u2.trajectory.n_frames == 1:
        if _attempt_merge(u1, u2, select, index_clash_l, index_merge_l,
                          rmsd_cut_off, clash_distance,
                          relabel_domain_segid=relabel_domain_segid,
                          strip_terminal_caps=strip_terminal_caps) is None:
            raise ValueError(
                "reweighted_fragment_assembly: both fragments have only one frame, "
                "so there is only one possible alignment/clash trial, and it failed "
                "(RMSD or clash criterion not met) -- retrying cannot ever succeed. "
                "This typically means one of the two fragments (often a "
                "domain-junction fragment built with too small a kmax) has too few "
                "sampled conformations to have a real chance of avoiding a clash. "
                "Give it a larger kmax (more sampled frames) and try again.")

    writePDB = True
    while k < kmax:
        if draw_indices:
            # random integer to draw random frame
            r1 = np.random.choice(u1.trajectory.n_frames, p=w_l[0])
            r2 = np.random.choice(u2.trajectory.n_frames, p=w_l[1])
        else:
            r1 = int(ri_l[k][0])
            r2 = int(ri_l[k][1])
            # r1 = int(ri_l[0,k])
            # r2 = int(ri_l[1,k])
        # load random frame
        u1.trajectory[r1]
        u2.trajectory[r2]
        
        # rmsd suportimposition of the subsequent fragmenmt, calculate clashes,
        # and assemble the subsequent fragments if both criteria pass
        u = _attempt_merge(u1, u2, select, index_clash_l, index_merge_l,
                           rmsd_cut_off, clash_distance,
                           relabel_domain_segid=relabel_domain_segid,
                           strip_terminal_caps=strip_terminal_caps)
        if u is not None:
            if writePDB :
               # save atom positions + topology for first pair in a pdb file
                u.atoms.write("{}/pair{}.pdb".format(dire,k))
                at = np.zeros((kmax , u.atoms.n_atoms , 3))
                writePDB = False

            # save atom positions of subsequehnt pairs in array
            at[k] = u.atoms.positions

            ## calculate weight for assembled chain/ fragment at step k
            ## unnormalized! except for mdfragments
            assembled_chain_weights[k] = chain_weights_prev_l[0][r1] * chain_weights_prev_l[1][r2]
            if draw_indices:
                rs[k] = [r1,r2]
            k+=1
        
    # load and save universe with pair0 as topology and atom positions 
    # of subsequent paired fragment of this assembly step as trajectory
    # n frames = desired number of pairs in this assembly step
    allPairs = mda.Universe("{}/pair0.pdb".format(dire) , at)
    allPairs.atoms.write("{}/pair.xtc".format(dire) , frames='all')
    
    # return random integer array if none was given as input
    if draw_indices:
        return rs, assembled_chain_weights
    else:
        return assembled_chain_weights


def reweighted_hierarchical_chain_growth(hcg_l, promo_l, overlaps_d, path0, path, kmax,
             rmsd_cut_off=0.6, clash_distance=2.0, capping_groups=True,
             ri_l=None,  path2weights='weights/', theta=10.0,  verbose=False,
             domain_id=None, domain_overlap=None, strip_cap_nterm=None, strip_cap_cterm=None,
             progress=True, strip_terminal_caps=True):
    """ perform reweighted hierarchical chain growth (+ ímportance sampling)
    assemble fragments (reweighted according to experimental data)
                        or pairs of fragments until reaching the full-length chain

    Parameters
    ----------
    hcg_l : list
        list for HCG with lists of fragments assigned to be paired
    promo_l : list
        list with boolean, if here is an promotion in level m (last assembly step)
    overlaps_d : dictionary
        dict of overlaps between subsequent fragments
    path0 : string
        path to the MD fragments
    path : string
        path to folders where the assembled pairs are stored in
    path : path
        path to the MD fragments (and to folders where the assembled pairs are stored in)
    kmax : integer
        number of pairs that should be assembled in level m_i
    rmsd_cut_off : float, optional
        cut-off for the RMSD of the fragment alignment. The default is 0.6
    clash_distance : float, optional
        max. allowed distance between atoms. The default is 2.0
    capping_groups : boolean, optional
        MD fragment are sampled with or without end-capping groups. The default is True
    ri_l : array-like
        array with indices for chosing a specific confoormation of a fragment. The default is None (=0)
    draw_indices : booolean
        if new random integers == frame indices are drawn or else taken from a input array.
    domain_id : hashable, optional
        fragment id of a rigid folded domain to attach the IDR to, see
        `hierarchical_chain_growth`. The default is None (no domain).
    domain_overlap : integer, optional
        see `hierarchical_chain_growth`. Required when `domain_id` is set.
    strip_cap_nterm : boolean, optional
        see `hierarchical_chain_growth`. The default is None (defaults to `capping_groups`).
    strip_cap_cterm : boolean, optional
        see `hierarchical_chain_growth`. The default is None (defaults to `capping_groups`).
    progress : boolean, optional
        whether to print a per-level tqdm progress bar to stderr, tracking how many of
        the current level's fragment pairs have finished. See `hierarchical_chain_growth`.
        The default is True.
    strip_terminal_caps : boolean, optional
        see `hierarchical_chain_growth`. The default is True.

    Returns
    -------
    None.
    """

    last_level = False
    k_max = kmax
    number_hcg_levels = len(hcg_l)

    for m , fragment_l in enumerate(hcg_l):
        # folder to save assembled pair in this level (m+1)
        level = m+1
        # folder to get old pairs from previous level (m)
        previous_level = m
        promotion  = promo_l[m]
        overlap = overlaps_d[0]
        
        if level > 1 :# and weighted:
            chain_weights_prev = np.load("{}/chain_weight_level{}.npy".format(path, previous_level), allow_pickle=True)
        else:
            chain_weights_prev = None
        if ri_l is None:
            draw_indices = True
            r_l = []
        else:
            r_l = ri_l[m]
            draw_indices = False

        assembled_chain_weights = []
        # if MD fragments are sampled with end-capping_groups, 
        # they are removed in the last assembly step 
        if m+1 == len(hcg_l):
            last_level = True
            k_max = kmax

        c1 = 0
        fragment_l_iter = enumerate(fragment_l)
        if progress:
            fragment_l_iter = tqdm(fragment_l_iter, total=len(fragment_l),
                                    desc="Level {}/{}".format(level, number_hcg_levels))
        for m_i , pair_l in fragment_l_iter:
            c2=c1+1
            proline_2nd_posi = False
            old_pair1, old_pair2 = _resolve_old_pairs(pair_l, promotion)

            if m == 0:
                previous_level = 'MDfragments'             
                path2fragment = path0
            else:
                path2fragment = path 
            old_dire1 = '{}/{}/{}'.format(path2fragment, previous_level, old_pair1)
            old_dire2 = '{}/{}/{}'.format(path2fragment, previous_level, old_pair2)
            # create folder/path to store assembled pairs
            dire = "{}/{}/{}".format(path, level, old_pair1)
            pathlib.Path(dire).mkdir(parents=True, exist_ok=True)
            
            # print('level, fragment, old_pairs, counter ', level, m_i, old_pair1, old_pair2, c1, c2)#, chain_weights_prev)
            
            # promotion of unpaired fragment to next higher hierarchy level
            if promotion and m_i == (len(fragment_l)-1):
                print('promotion in level, pair',  level, old_pair1)
                
                
                # store weight of promoted fragment for next level
                if level == 1:
                    w  = np.genfromtxt('{}/{}/wopt/w_theta{}_dat.txt'.format(path2weights, old_pair1, theta))
                else:
                    w = chain_weights_prev[c1]
                assembled_chain_weights.append(w)
                if os.path.exists('{}/pair0.pdb'.format(dire)):
                    # print('already copied promoted fragment ', 
                    #       level, previous_level, old_pair1)
                    continue
                else:
                    shutil.copyfile('{}/pair0.pdb'.format(old_dire1),
                                        '{}/pair0.pdb'.format(dire))
                    shutil.copyfile('{}/pair.xtc'.format(old_dire1),
                                        '{}/pair.xtc'.format(dire))
                    continue
                
            # load universe -> load conformations of fragment 1 and 2
            u1 = mda.Universe('{}/pair0.pdb'.format(old_dire1),
                              '{}/pair.xtc'.format(old_dire1))
            u2 = mda.Universe('{}/pair0.pdb'.format(old_dire2),
                              '{}/pair.xtc'.format(old_dire2))                       
            ## weights 
            if level == 1:
                
                w1  = np.genfromtxt('{}/{}/wopt/w_theta{}_dat.txt'.format(path2weights, old_pair1, theta))
                w2  = np.genfromtxt('{}/{}/wopt/w_theta{}_dat.txt'.format(path2weights, old_pair2, theta))
                chain_weights_prev_l = [w1, w2]
            else:
                w1 = np.ones(u1.trajectory.n_frames)/u1.trajectory.n_frames
                w2 = np.ones(u2.trajectory.n_frames)/u2.trajectory.n_frames
                chain_weights_prev_l = [chain_weights_prev[c1], chain_weights_prev[c2]]
            
            # store w1, w2 in a list as input for fragment_assembly
            w_l = [w1, w2]
            o, pair_overlap0, pair_capping_groups = _resolve_pair_overlap(
                    pair_l, old_pair1, old_pair2, domain_id, domain_overlap,
                    overlap, overlaps_d, capping_groups)
            # once assembly finishes (last_level), the domain and the grown IDR are
            # one single, covalently continuous molecule -- erase DOMAIN_SEGID's
            # internal bookkeeping tag (needed by find_clashes at every earlier
            # level, see its own docstring) from the output instead of leaving it
            # looking like two separate molecules/chains
            relabel_domain_segid = DOMAIN_SEGID if (last_level and domain_id is not None) else None
            # unlike relabel_domain_segid, this is not domain-specific: any run's
            # truly final merge can have a leftover cap (see strip_terminal_caps's
            # docstring on hierarchical_chain_growth)
            strip_caps_now = strip_terminal_caps and last_level

            # get indices for assembly
            index_aln_l, index_clash_l, index_merge_l = get_residue_indices_for_assembly(
                                                    overlap0=pair_overlap0, current_overlap=o,
                                                    capping_groups=pair_capping_groups, last_level=last_level,
                                                    verbose=verbose, strip_cap_nterm=strip_cap_nterm,
                                                    strip_cap_cterm=strip_cap_cterm)

            ## check if for u2 residue 2 that is aligned is ja proline
            proline_2nd_posi = _is_proline_at_alignment_end(u2, index_aln_l)

            # create dictionary with desired residues
            # defined as "mobile" and "ref" you want to superimpose
            select = translate_concept(u1, u2, proline_2nd_posi, *index_aln_l)
            if verbose:
                print('level pair to grow, previous_level, old_pair1, old_pair2, overlap',
                      level, previous_level, old_pair1, old_pair2, o)
                print('fragment 1 ',
                      u1.select_atoms('{}'.format(select['mobile'])).residues.resnames)
                print('fragment 2 ',
                      u2.select_atoms('{}'.format(select['reference'])).residues.resnames)
            
            # assemble the fragments into pairs
            # (or pairs into pairs of pairs)
            if  draw_indices:
                
                rs, acw_mi = reweighted_fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
                                  rmsd_cut_off, clash_distance,  kmax=k_max, w_l=w_l,
                                  chain_weights_prev_l=chain_weights_prev_l,
                                  relabel_domain_segid=relabel_domain_segid,
                                  strip_terminal_caps=strip_caps_now)
                r_l.append(rs)
                assembled_chain_weights.append(acw_mi)
            else:
                rs = r_l[m_i]
                acw_mi = reweighted_fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
                                  rmsd_cut_off, clash_distance,  kmax=k_max, w_l=w_l, ri_l=rs,
                                  draw_indices=draw_indices, chain_weights_prev_l=chain_weights_prev_l,
                                  relabel_domain_segid=relabel_domain_segid,
                                  strip_terminal_caps=strip_caps_now)
                assembled_chain_weights.append(acw_mi)
            c1 += 2
        if draw_indices:
            np.save("{}/confIndex_level{}.npy".format(path, level), r_l)
        # assembled_chain_weights can be ragged (a promoted fragment's weight array has
        # its own native frame count, not kmax) -- dtype=object is required on current
        # numpy, which no longer silently allows saving an inhomogeneous array
        np.save("{}/chain_weight_level{}.npy".format(path, level),
                np.array(assembled_chain_weights, dtype=object))
    return None

