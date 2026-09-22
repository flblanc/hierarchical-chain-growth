#!/usr/bin/env python3
"""
hcg_fct_multiproc
-----------
core functions for hierarchical chain growth

run hcg in parallel - per level -> m_i loop is paralleliiiized
using multipprocessing.Pool
"""
import numpy as np
import MDAnalysis as mda
import pathlib, shutil, os
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from chain_growth.hcg_list import derive_strip_cap_from_domain_position
# the per-pair geometry/selection helpers used to live in this module; they moved to
# chain_growth.assembly for organization, and are re-exported here unchanged so
# existing `from chain_growth.hcg_fct import ...` call sites keep working
from chain_growth.assembly import (
    HYDROGEN_NAME_SELECTION, DOMAIN_SEGID, translate_concept, find_clashes,
    merge_universe, get_residue_indices_for_assembly, _resolve_old_pairs,
    _resolve_pair_overlap, _is_proline_at_alignment_end, _attempt_merge)


def fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
         rmsd_cut_off, clash_distance, kmax, ri_l=None, draw_indices=True,
         relabel_domain_segid=None, strip_terminal_caps=False):
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
    ri_l : array-like
        array with indices for chosing a specific confoormation of a fragment. The default is None
        if None: draw indices randomly
    draw_indices : booolean
        if new random integers == frame indices are drawn or else taken from a input array.
    relabel_domain_segid : string, optional
        see `chain_growth.assembly._attempt_merge`; pass `DOMAIN_SEGID` only for the
        truly final merge of a domain-attachment run. The default is None.
    strip_terminal_caps : boolean, optional
        see `chain_growth.assembly._attempt_merge`; pass True only for the truly
        final merge. The default is False.

    Returns
    -------
    if draw_indices:
        rs = list of lists
        successful indices drawn for fragment 1 and 2 during fragment assembly
    else:
        None
    """

    k = 0

    if ri_l is None:
        # array to store random frame indices of successfully assembled fragments / pairs
        rs = np.zeros((kmax, 2))

    # when both fragments have exactly one frame, there is only ever one possible
    # (r1, r2) combination -- every iteration of the loop below would draw frame 0
    # from each and repeat the exact same, deterministic trial forever. If that one
    # trial doesn't pass, retrying cannot ever succeed, so fail fast with a clear
    # error instead of spinning indefinitely (e.g. a domain-junction fragment built
    # with too small a kmax has no other conformation to fall back on)
    if draw_indices and u1.trajectory.n_frames == 1 and u2.trajectory.n_frames == 1:
        if _attempt_merge(u1, u2, select, index_clash_l, index_merge_l,
                          rmsd_cut_off, clash_distance,
                          relabel_domain_segid=relabel_domain_segid,
                          strip_terminal_caps=strip_terminal_caps) is None:
            raise ValueError(
                "fragment_assembly: both fragments have only one frame, so there is "
                "only one possible alignment/clash trial, and it failed (RMSD or "
                "clash criterion not met) -- retrying cannot ever succeed. This "
                "typically means one of the two fragments (often a domain-junction "
                "fragment built with too small a kmax) has too few sampled "
                "conformations to have a real chance of avoiding a clash. Give it a "
                "larger kmax (more sampled frames) and try again.")

    writePDB = True
    while k < kmax:
        if draw_indices:
            # random integer to draw random frame
            r1 = np.random.randint(u1.trajectory.n_frames)
            r2 = np.random.randint(u2.trajectory.n_frames)
        else:
            r1 = int(ri_l[k][0])
            r2 = int(ri_l[k][1])
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
        return rs
    else:
        return None

def hierarchical_chain_growth(hcg_l, promo_l, overlaps_d, path0, path, kmax,
             dict_to_fragment_folder=None, rmsd_cut_off=0.6, clash_distance=2.0, capping_groups=True,
             ri_l=None, streamlit_progressbar=None, verbose=False, domain_id=None,
             domain_overlap=None, strip_cap_nterm=None, strip_cap_cterm=None,
             num_threads=None, progress=True, strip_terminal_caps=True):
    """ perform hierarchical chain growth 
    assemble fragments/ pairs of fragments until reaching the full-length chain
    by calling _loop_func -> does the inner loop and calls fragment assembly
    takes same arguments as hcg function

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
    dict_to_fragment_folder : dictionary
        dictionary that transaltes between the input sequence - required fragments - 
        and the code. The default is None
    rmsd_cut_off : float, optional
        cut-off for the RMSD of the fragment alignment. The default is 0.6
    clash_distance : float, optional
        max. allowed distance between atoms. The default is 2.0
    capping_groups : boolean, optional
        MD fragment are sampled with or without end-capping groups. The default is True
    ri_l : array-like
        array with indices for chosing a specific confoormation of a fragment. The default is None
    draw_indices : booolean
        if new random integers == frame indices are drawn or else taken from a input array.
    domain_id : hashable, optional
        fragment id (as used in `fragment_list`/the MDfragments folder name) of a rigid folded
        domain to attach the IDR to. When set, the junction where the domain first joins its
        neighboring fragment is treated as capping-group-free (its overlap residues are the
        domain's own real residues, not a synthetic cap), using `domain_overlap` rather than an
        `overlaps_d` lookup. The default is None (no domain).
    domain_overlap : integer, optional
        number of residues the domain shares with its neighboring fragment's overlap region.
        Required when `domain_id` is set; not read from `overlaps_d` (which reserves key `0` for
        the run's general default overlap).
    strip_cap_nterm : boolean, optional
        whether to strip the capping-group residue at the N-terminal-most exposed end of the
        full-length chain at the last assembly level. When `domain_id` is set, leave this as
        None: it's derived automatically from where `domain_id` actually ends up in `hcg_l`
        (see `chain_growth.hcg_list.derive_strip_cap_from_domain_position`), so the domain's
        real terminal residue is always kept rather than risking it being stripped as if it
        were a synthetic cap -- a real run once got this backwards by passing an explicit,
        wrong value, which is exactly what an explicit value that disagrees with the derived
        one now raises a clear error for instead of silently doing the wrong thing. When
        `domain_id` is None, defaults to `capping_groups` when left as None, unchanged from
        before.
    strip_cap_cterm : boolean, optional
        whether to strip the capping-group residue at the C-terminal-most exposed end of the
        full-length chain at the last assembly level. See `strip_cap_nterm` -- the same
        automatic derivation (and validation against an explicit value) applies here too when
        `domain_id` is set.
    num_threads : integer, optional
        number of worker processes to use for the per-level Pool that assembles a level's
        independent fragment pairs in parallel. The default is None, which auto-detects
        `os.cpu_count()` (capped at the number of pairs in a level). Set to 1 to force fully
        serial execution -- e.g. when calling this from a context where starting subprocesses
        is unsafe or undesirable (a plain interactive/REPL session, some notebook setups, or a
        shared/HPC login node), or for debugging.
    progress : boolean, optional
        whether to print a per-level tqdm progress bar to stderr, tracking how many of
        the current level's independent fragment pairs have finished (each pair draws
        `kmax` accepted conformations via rejection sampling, so pairs can take a
        while, especially at a large, real domain junction). The default is True. Set
        to False to suppress it -- e.g. when embedding this in another UI that has its
        own progress reporting (see `streamlit_progressbar`), or to keep captured
        output clean.
    strip_terminal_caps : boolean, optional
        whether to remove a leftover leading ACE and/or trailing NME residue from the
        truly final, full-length assembled chain, checked by residue name rather than
        position -- a safety net independent of (and applied after) strip_cap_nterm/
        strip_cap_cterm, which work by position and are only correct if those are set
        to match which physical end of the chain each one actually governs (easy to
        get backwards, particularly for a domain-attachment run; see
        `chain_growth.assembly._strip_terminal_caps`'s docstring for a real example of
        exactly that mistake silently deleting a domain's real terminal residue). The
        default is True.

    Returns
    -------
    None.
    """

    if domain_id is not None:
        derived_nterm, derived_cterm = derive_strip_cap_from_domain_position(hcg_l, domain_id)
        if strip_cap_nterm is None:
            strip_cap_nterm = derived_nterm
        elif strip_cap_nterm != derived_nterm:
            raise ValueError(
                "strip_cap_nterm={} conflicts with where domain_id={!r} actually ends "
                "up in the assembled chain (derived from hcg_l/fragment_ids): it should "
                "be {}, so that the domain's real terminal residue is kept rather than "
                "stripped as if it were a synthetic cap (this is exactly the mistake a "
                "real run made -- see strip_terminal_caps's docstring). Pass "
                "strip_cap_nterm={} explicitly, or leave it as None to have this derived "
                "automatically.".format(strip_cap_nterm, domain_id, derived_nterm, derived_nterm))
        if strip_cap_cterm is None:
            strip_cap_cterm = derived_cterm
        elif strip_cap_cterm != derived_cterm:
            raise ValueError(
                "strip_cap_cterm={} conflicts with where domain_id={!r} actually ends "
                "up in the assembled chain (derived from hcg_l/fragment_ids): it should "
                "be {}, so that the domain's real terminal residue is kept rather than "
                "stripped as if it were a synthetic cap (this is exactly the mistake a "
                "real run made -- see strip_terminal_caps's docstring). Pass "
                "strip_cap_cterm={} explicitly, or leave it as None to have this derived "
                "automatically.".format(strip_cap_cterm, domain_id, derived_cterm, derived_cterm))

    last_level = False
    k_max = kmax
    number_hcg_levels = hcg_l.__len__()
    i_it = 0

    cpu = os.cpu_count() if num_threads is None else num_threads
    # Pool workers are forked, so they inherit an *identical* copy of this process's
    # global NumPy RNG state; without reseeding, different workers would draw the same
    # "random" frame-index sequence. Derive a SeedSequence from the current global state
    # (so a prior np.random.seed() call still makes the whole run reproducible) and spawn
    # one independent child seed per task below, so every worker reseeds itself before
    # drawing any random frame indices.
    seed_seq = np.random.SeedSequence(np.random.randint(0, 2**32 - 1))
    for m , fragment_l in enumerate(hcg_l):
        # folder to save assembled pair in this level (m+1)
        level = m+1
        # folder to get old pairs from previous level (m)
        previous_level = m
        promotion  = promo_l[m]
 
        ## if no index list is supplied draw new indices (fragment conformations)
        if ri_l is None:
            draw_indices = True
            r_l = []
        else:
            r_l = ri_l[m]
            draw_indices = False
            
        # if MD fragments are sampled with end-capping_groups, 
        # they are removed in the last assembly step 
        if m+1 == number_hcg_levels:
            last_level = True
    
        if len(fragment_l) > cpu:
            level_num_threads = cpu
        else:
            level_num_threads = len(fragment_l)
        child_seeds = seed_seq.spawn(len(fragment_l))
        pairs = [(m_i, pair_l, child_seeds[m_i]) for m_i, pair_l in enumerate(fragment_l)]
        d = {"path0": path0, "path": path, 
             "dict_to_fragment_folder": dict_to_fragment_folder,
             "level": level, 'previous_level': previous_level,
             "last_level": last_level, "fragment_l": fragment_l, "rmsd_cutoff": rmsd_cut_off,
             "clash_distance": clash_distance, "kmax": kmax, "r_l":r_l,
             "overlap_d": overlaps_d, "promotion": promotion, "capping_groups": capping_groups,
             "draw_indices" : draw_indices, "verbose": verbose,
             "domain_id": domain_id, "domain_overlap": domain_overlap,
             "strip_cap_nterm": strip_cap_nterm, "strip_cap_cterm": strip_cap_cterm,
             "strip_terminal_caps": strip_terminal_caps }
        # POOL LOOP application
        with Pool(level_num_threads) as p:
            func = partial(_loop_func, d)
            if progress:
                # imap (not imap_unordered) preserves the same pairs-order guarantee
                # p.map makes, which r_l's draw_indices=False lookup (r_l[m_i]) below
                # relies on -- this only adds a progress bar, nothing else changes
                results = list(tqdm(p.imap(func, pairs), total=len(pairs),
                                     desc="Level {}/{}".format(level, number_hcg_levels)))
            else:
                results = p.map(func, pairs)
        if draw_indices:
            for r in results:
                if r is not None:
                    r_l += [r]
                    
            np.save("{}/confIndex_level{}.npy".format(path, m+1), r_l) 

        if streamlit_progressbar is not None:
            i_it = i_it + 1

            progress = float(i_it) / float(number_hcg_levels)
            #print(i_it, progress)
            streamlit_progressbar.progress(progress)
    return None


def _loop_func(variables, pairs):
    """ does the inner loop of hierarchical chain growth and calls fragment assembly
        
    idea for possible implementation - inspired from discussion with / and 
    realized with input from Hendrik Jung    

    Parameters
    ----------
    variables : dictionary
        variables needed for fragment assembly as defined in hcg function.
    pairs : tuple
        (m_i, pair_l, seed): pair_l of pairs need to be assembled per level, the
        respective pair id, and a per-task numpy SeedSequence used to reseed this
        (forked, and therefore otherwise RNG-state-identical) worker before any random
        frame indices are drawn.

    Returns
    -------
    if draw_indices:
        rs = list of lists
        successful indices drawn for fragment 1 and 2 during fragment assembly
    else:
        None
    """
    
    # print(variables)
    m_i, pair_l, seed = pairs
    # reseed: this worker is a forked copy of the parent process and otherwise starts
    # with an identical global RNG state to every other worker (see hierarchical_chain_growth)
    np.random.seed(seed.generate_state(4))
    path0 = variables["path0"]
    path = variables["path"]
    dict_to_fragment_folder = variables["dict_to_fragment_folder"]
    level = variables["level"]
    previous_level = variables["previous_level"]
    last_level = variables["last_level"]
    fragment_l = variables["fragment_l"]
    rmsd_cut_off = variables["rmsd_cutoff"]
    clash_distance = variables["clash_distance"]
    k_max =  variables["kmax"]
    r_l = variables["r_l"]
    draw_indices = variables["draw_indices"]
    overlaps_d = variables["overlap_d"]
    promotion = variables["promotion"]
    capping_groups = variables["capping_groups"]
    verbose = variables["verbose"]
    domain_id = variables["domain_id"]
    domain_overlap = variables["domain_overlap"]
    strip_cap_nterm = variables["strip_cap_nterm"]
    strip_cap_cterm = variables["strip_cap_cterm"]
    strip_terminal_caps = variables["strip_terminal_caps"]

    overlap = overlaps_d[0]

    # if MD fragments are sampled with end-capping_groups, 
    # they are removed in the last assembly step 

    proline_2nd_posi = False
    old_pair1, old_pair2 = _resolve_old_pairs(pair_l, promotion)

    if previous_level == 0 and dict_to_fragment_folder is not None:
        ## path0 = path to online fragment library
        path2fragment = path0 
        #print(old_pair1, old_pair2)
        old_pair1_fragmentLib = dict_to_fragment_folder[old_pair1]
        old_pair2_fragmentLib = dict_to_fragment_folder[old_pair2]
        #print(old_pair1_fragmentLib, old_pair2_fragmentLib)
        old_dire1 = '{}/{}'.format(path2fragment, old_pair1_fragmentLib)
        old_dire2 = '{}/{}'.format(path2fragment, old_pair2_fragmentLib)
        #print(old_dire1, old_dire2)
        top = 'fragment.pdb'
        xtc = 'fragment.xtc'
                
    elif previous_level == 0 and dict_to_fragment_folder is None:
        previous_level = 'MDfragments'             
        path2fragment = path0
        old_dire1 = '{}/{}/{}'.format(path2fragment, previous_level, old_pair1)
        old_dire2 = '{}/{}/{}'.format(path2fragment, previous_level, old_pair2)
        top = 'pair0.pdb'
        xtc = 'pair.xtc'
    else:
        path2fragment = path 
        old_dire1 = '{}/{}/{}'.format(path2fragment, previous_level, old_pair1)
        old_dire2 = '{}/{}/{}'.format(path2fragment, previous_level, old_pair2)
        top = 'pair0.pdb'
        xtc = 'pair.xtc'
        
    # create folder/path to store assembled pairs
    dire = "{}/{}/{}".format(path, level, old_pair1)
    pathlib.Path(dire).mkdir(parents=True, exist_ok=True)
 
    # promotion of unpaired fragment to next higher hierarchy level
    if promotion and m_i == (len(fragment_l)-1):
        # print('promotion in level, pair', level, old_pair1)
        if not os.path.exists('{}/{}'.format(dire, top)):
            # print('already copied promoted fragment ', 
            #       level, previous_level, old_pair1)

            shutil.copyfile('{}/{}'.format(old_dire1, top),
                                '{}/pair0.pdb'.format(dire))
            shutil.copyfile('{}/{}'.format(old_dire1, xtc),
                                '{}/pair.xtc'.format(dire))
    else:
        # load universe -> load conformations of fragment 1 and 2
        u1 = mda.Universe('{}/{}'.format(old_dire1, top),
                          '{}/{}'.format(old_dire1, xtc), refresh_offsets=True)
        u2 = mda.Universe('{}/{}'.format(old_dire2, top),
                          '{}/{}'.format(old_dire2, xtc), refresh_offsets=True)
 
        o, pair_overlap0, pair_capping_groups = _resolve_pair_overlap(
                pair_l, old_pair1, old_pair2, domain_id, domain_overlap,
                overlap, overlaps_d, capping_groups)

        # get indices for assembly
        index_aln_l, index_clash_l, index_merge_l = get_residue_indices_for_assembly(
                                                overlap0=pair_overlap0, current_overlap=o,
                                                capping_groups=pair_capping_groups, last_level=last_level,
                                                verbose=verbose, strip_cap_nterm=strip_cap_nterm,
                                                strip_cap_cterm=strip_cap_cterm)

        ## check if for u2 residue 2 that is aligned is a proline
        ## only makes a difference if residue overlap is < 2
        proline_2nd_posi = _is_proline_at_alignment_end(u2, index_aln_l)

        # create dictionary with desired residues
        # defined as "mobile" and "ref" you want to superimpose
        select = translate_concept(u1, u2, proline_2nd_posi, *index_aln_l)
        if verbose:
            print('level pair to grow, previous level, old_pair1, old_pair2, overlap',
                  level, previous_level, old_pair1, old_pair2, o)
            print('fragment 1 ',
                  u1.select_atoms('{}'.format(select['mobile'])).residues.resnames)
            print('fragment 2 ',
                  u2.select_atoms('{}'.format(select['reference'])).residues.resnames)
 
        # once assembly finishes (last_level), the domain and the grown IDR are one
        # single, covalently continuous molecule -- erase DOMAIN_SEGID's internal
        # bookkeeping tag (needed by find_clashes at every earlier level, see its
        # own docstring) from the output instead of leaving it looking like two
        # separate molecules/chains
        relabel_domain_segid = DOMAIN_SEGID if (last_level and domain_id is not None) else None
        # unlike relabel_domain_segid, this is not domain-specific: any run's truly
        # final merge can have a leftover cap (see strip_terminal_caps's docstring
        # on hierarchical_chain_growth)
        strip_caps_now = strip_terminal_caps and last_level

        # assemble the fragments into pairs
        # (or pairs into pairs of pairs)
        if  draw_indices:
            rs = fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
                              rmsd_cut_off, clash_distance,  kmax=k_max,
                              relabel_domain_segid=relabel_domain_segid,
                              strip_terminal_caps=strip_caps_now)
            ###############
            #r_l.append(rs)
            ###############
            return rs
        else:
            rs = r_l[m_i]
            fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
                              rmsd_cut_off, clash_distance,  kmax=k_max, ri_l=rs,
                              draw_indices=draw_indices, relabel_domain_segid=relabel_domain_segid,
                              strip_terminal_caps=strip_caps_now)
            return None
 

 

# reweighted_hierarchical_chain_growth/reweighted_fragment_assembly used to live in
# this module; they moved to chain_growth.rhcg_fct for organization, and are
# re-exported here unchanged so existing `from chain_growth.hcg_fct import ...`
# call sites (e.g. run_rhcg.py) keep working
from chain_growth.rhcg_fct import reweighted_fragment_assembly, reweighted_hierarchical_chain_growth
