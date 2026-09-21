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
from MDAnalysis.analysis import align
import pathlib, shutil, os
import MDAnalysis.analysis.distances as distances
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from chain_growth.hcg_list import flatten

def translate_concept(u1, u2, proline_2nd_posi, align_begin1, align_end1, 
                      align_begin2, align_end2):
    """ prepare a dictionary with atoms to align before the assembly step
    
    Parameter
    ---------
    u1 : universe
    u2 : universe
    proline_2nd_posi : boolean
        Whether or not in  u2 2nd residue to align is a proline 
    align_begin1 : integer
        residue index for u1, marks residue to begin the alignment for u1
    align_end1 : integer
        residue index for u1, marks residue to end the alignment for u1
    align_begin2 : integer
        residue index for u2, marks residue to begin the alignment for u2
    align_end2 : integer
        residue index for u2, marks residue to end the alignment for u2
        
    Returns
    -------
    select : dictionary 
        dictionary with atoms that are aligned prior to the assembly step
    """
    ## whether or not a amide proton is there
    if proline_2nd_posi:
        atoms_sel_2nd_res = "name  N"
    else:
        atoms_sel_2nd_res= "name  N or name H"
    
    select={}
    ## atom selection of fragment1 to be aligned
    sel1 = "(resid {} and (name C or name O)) or (resid {} and ({}))".format(
                            u1.atoms.residues[align_begin1].resid, 
                            u1.atoms.residues[align_end1].resid, atoms_sel_2nd_res)
    ## atom selection of fragment2 to be aligned
    sel2 = "(resid {}  and (name C or name O)) or (resid {} and ({}))".format(
                           u2.atoms.residues[align_begin2].resid,
                           u2.atoms.residues[align_end2].resid, atoms_sel_2nd_res)

    ## dict is argument for mda alignto function: arg = "select="
    ## mobile is superimposed on ref with mda.alignto in the assembly step
    select={'mobile': sel1, 'reference': sel2}
    return select


def find_clashes(u1,u2, index1b, index2e, 
                 index1e=-1, clash_radius=2.0):
    """ finds clashes: atoms with a distance below clash_radius
    
    Parameters
    ----------
    u1 : universe
        fragment 1 to pair
    u2 : universe
        fragment 2 to pair
    index1b : integer
        residue index for residues to *exclude* from clash detection
        marks residue to begin exclusion for u1
    index2e : integer
        residue index for residues to *exclude* from clash detection
        marks residue to end exclusion for u2
    index1e : integer
        residue index for residues to *exclude* from clash detection
        marks residue to end exclusion for u1. The default is -1
    clash_distance : float
        max. allowed distance between atoms, everything below is counted as clash
        The  default is 2.0

    Returns
    -------
    clsum : integer
        count of clashes between aligned fragments
        
    NOTE: MDAnalysis is inclusive! resid 1:2 -> selects residues 1+2
    """
    l1 = u1.select_atoms("protein and not (type H) and not (resid {} and backbone) and not resid {}:{}".format(
                                                    u1.atoms.residues[index1b].resid,
                                                    u1.atoms.residues[index1b+1].resid,
                                                    u1.atoms.residues[index1e].resid))

    # atom selection of u2 to scan for clashes
    l2 = u2.select_atoms("protein and not (type H) and not resid 1:{} and not (resid {} and backbone)".format(
                                                    u2.atoms.residues[index2e-1].resid,
                                                    u2.atoms.residues[index2e].resid))

    # -> use mda.distances to generate a matrix with distances
    distmat = distances.distance_array(l1.positions,l2.positions)
    cont = np.less(distmat,clash_radius) # see where distmat < cutoff
    cont = np.where(cont,1,0) # True --> 1, False --> 0
    clsum = np.sum(cont)
    
    return clsum


def merge_universe(u1,u2, merge_begin1, merge_end1, 
                   merge_begin2, merge_end2):
    """ assemble aligned universes and renumber residues in a consecutive manner
    
    Parameters
    ----------
    u1 : universe
        fragment 1 to pair
    u2 : universe
        fragment 2 to pair
    merge_begin1 : integer
        residue index to begin merge for u1
    merge_begin1 : integer
        residue index to end merge for u1
    merge_begin2 : integer
            residue index to begin merge for u2
    merge_end2 : integer
            residue index to end merge for u2
            
    Returns
    ------
    u : universe
        combined universes u1 and u2 with renumbered residues
    """
    
    # atom selection of u1 to be merged
    sel1="resid {}:{}".format(u1.atoms.residues[merge_begin1].resid , u1.atoms.residues[merge_end1].resid)
    sela = u1.select_atoms(sel1)
    # atom selection of u2 to be merged
    sel2="resid {}:{}".format(u2.atoms.residues[merge_begin2].resid , u2.atoms.residues[merge_end2].resid)
    selb = u2.select_atoms(sel2)
    u = mda.core.universe.Merge(sela, selb)
    u_atm = u.select_atoms('all')
    # renumber residue IDs after assembly
    u_atm.residues.resids = np.arange(1,len(u_atm.residues.resids)+1) 

    return u


def get_residue_indices_for_assembly(overlap0, current_overlap, capping_groups,
                                     last_level, verbose, strip_cap_nterm=None,
                                     strip_cap_cterm=None):
    """ assign residue indices for alignment, clash detection, assembly
    -> indices depend on the overlap between subsequent fragments

    Parameters
    ----------
    current_overlap : integer
        overlap between the fragments to assemble at current step
    overlap0 : initial overlap, optional
        overlap initially chosen for the fragments without taking into account exceptions. The default is 2
    capping_groups : boolean, optional
        MD fragment are sampled with or without end-capping groups. The default is True
    last_level : boolean, optional
        last level of hierarchical chain growth. The default is False
    strip_cap_nterm : boolean, optional
        whether to strip the capping-group residue at the N-terminal-most exposed end
        at last_level. Defaults to `capping_groups` when None. Set to False when that
        end is a folded domain's real terminus rather than a synthetic cap.
    strip_cap_cterm : boolean, optional
        whether to strip the capping-group residue at the C-terminal-most exposed end
        at last_level. Defaults to `capping_groups` when None. Set to False when that
        end is a folded domain's real terminus rather than a synthetic cap.

    Returns
    -------
    index_aln_l : list
       residue indices for alignment, used in translate_concept
    index_clash_l : list
       residue indices for clash detection, used in find_clashes
    index_merge_l : list
        residue indices for fragment assembly, used in merge_universe
    """
    if strip_cap_nterm is None:
        strip_cap_nterm = capping_groups
    if strip_cap_cterm is None:
        strip_cap_cterm = capping_groups

    ## e: additional factor for end-capping groups
    ## having 1 overlapping residue + align peptide bonds does work only with headgroup!
    e = 1
    if overlap0  == 0:
        e = 0
        align_begin1= -2
        align_end1= -1
        align_begin2= 0
    else:
        if overlap0 > 0 and capping_groups == False:
            e = 0
        # align peptide bond between two last/ two first residues
        align_begin1= -(overlap0 + e)
        align_end1= align_begin1+1
        align_begin2= 0 + e
        
    if current_overlap != overlap0:
        align_begin2= np.abs(current_overlap - overlap0) + e
    align_end2= align_begin2+1        
    
        
    # indices to exclude residues from clash calculation
    index1_clashB = align_begin1
    index2_clashE = align_end2
        
    # indicies for assembly of aligned pairs    
    merge_begin1 = 0 # always first residue, maybe change this to make it more flexible??
    merge_end1 = align_begin1
    merge_begin2 = align_end2
    merge_end2 = -1 #always last residue, maybe change this to make it more flexible??
    
    # exclude capping groups
    if last_level:
        if strip_cap_nterm:
            merge_begin1 = 1
        if strip_cap_cterm:
            merge_end2 = -2
    
    index_aln_l = [align_begin1, align_end1, align_begin2, align_end2]
    index_clash_l = [index1_clashB, index2_clashE]
    index_merge_l = [merge_begin1, merge_end1, merge_begin2, merge_end2]
    
    if verbose:
        print('overlap between fragment 1 & 2:' , current_overlap,
               'align begin / end in fragment 1 & 2, respectively: ', index_aln_l,
               'clash search begin / end in fragment 1 & 2, respectively: ', index_clash_l,
               'merge begin / end in fragment 1 & 2, respectively: ', index_merge_l)
       
    return index_aln_l, index_clash_l, index_merge_l
        
def fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
         rmsd_cut_off, clash_distance, kmax, ri_l=None, draw_indices=True):
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

    Returns
    -------
    if draw_indices:
        rs = list of lists
        successful indices drawn for fragment 1 and 2 during fragment assembly
    else:
        None
    """
    
    k = 0
    assembly_atempt = 0
    
    if ri_l is None:
        # array to store random frame indices of successfully assembled fragments / pairs
        rs = np.zeros((kmax, 2))
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
        
        # rmsd suportimposition of the subsequent fragmenmt         
        old,new = align.alignto(u1, u2, select=select, weights="mass",
                                tol_mass=5., match_atoms=False)
        assembly_atempt += 1
        if new < rmsd_cut_off:
            # calculate clashes
            clashes = find_clashes(u1 , u2 , index1b=index_clash_l[0] , index2e=index_clash_l[1],
                                  clash_radius=clash_distance)
            
            if clashes < 1:
                ## assemble the subsequent fragments 
                u = merge_universe(u1, u2, *index_merge_l)
                
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
             num_threads=None, progress=True):
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
        full-length chain at the last assembly level. Defaults to `capping_groups` when None. Set
        to False when the N-terminus is a folded domain's real terminus (attached via
        `domain_id`) rather than a synthetic cap.
    strip_cap_cterm : boolean, optional
        whether to strip the capping-group residue at the C-terminal-most exposed end of the
        full-length chain at the last assembly level. Defaults to `capping_groups` when None. Set
        to False when the C-terminus is a folded domain's real terminus (attached via
        `domain_id`) rather than a synthetic cap.
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

    Returns
    -------
    None.
    """

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
             "strip_cap_nterm": strip_cap_nterm, "strip_cap_cterm": strip_cap_cterm }
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

    overlap = overlaps_d[0]

    # if MD fragments are sampled with end-capping_groups, 
    # they are removed in the last assembly step 

    proline_2nd_posi = False
    # if promotion of MD fragment in first level DO NOT define old_pair2
    # index ERROR because in this case, pair_l is no list
    if promotion and isinstance(pair_l, list) == False:
        # subfolder fragment 1, 2
        old_pair1  = pair_l
        old_pair2  = pair_l
    else:
        # subfolder fragment 1, 2
        old_pair1 = flatten(pair_l[0])[0]
        old_pair2 = flatten(pair_l[1])[0]
        
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
 
        # the domain's own junction (where it first joins its neighboring fragment,
        # i.e. that side of pair_l is still a raw, unmerged fragment) is not a
        # "typical" variation of the general fragment overlap: it has its own
        # overlap length (domain_overlap) and its overlap residues are the
        # domain's real residues, not a synthetic cap. It is looked up via
        # domain_overlap, not overlaps_d, since overlaps_d reserves key 0 for the
        # run's general default overlap.
        domain_first_junction = domain_id is not None and (
                (old_pair1 == domain_id and isinstance(pair_l[0], list) == False) or
                (old_pair2 == domain_id and isinstance(pair_l[1], list) == False))

        # residue ovearlap MD fragment
        if domain_first_junction:
            o = domain_overlap
        elif len(flatten(pair_l[1])) == 1:
            o = overlaps_d[old_pair2]
        # overlap grown pairs, always == overlaps[0]
        # -> overlap that differs "corrected" for when growing pairs with MD fragment
        else:
            o = overlap

        pair_overlap0 = o if domain_first_junction else overlap
        pair_capping_groups = False if domain_first_junction else capping_groups

        # get indices for assembly
        index_aln_l, index_clash_l, index_merge_l = get_residue_indices_for_assembly(
                                                overlap0=pair_overlap0, current_overlap=o,
                                                capping_groups=pair_capping_groups, last_level=last_level,
                                                verbose=verbose, strip_cap_nterm=strip_cap_nterm,
                                                strip_cap_cterm=strip_cap_cterm)
 
        ## check if for u2 residue 2 that is aligned is a proline
        ## only makes a difference if residue overlap is < 2
        res2_u2 = u2.select_atoms('resid {}'.format(u2.atoms.residues[index_aln_l[-1]].resid))
        res2_u2_name = res2_u2.residues.resnames
        if res2_u2_name == 'PRO':
            proline_2nd_posi = True
 
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
 
        # assemble the fragments into pairs
        # (or pairs into pairs of pairs)
        if  draw_indices:
            rs = fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
                              rmsd_cut_off, clash_distance,  kmax=k_max)
            ###############
            #r_l.append(rs)
            ###############
            return rs
        else:
            rs = r_l[m_i]
            fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
                              rmsd_cut_off, clash_distance,  kmax=k_max, ri_l=rs,
                              draw_indices=draw_indices)
            return None
 

 
def reweighted_fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
         rmsd_cut_off, clash_distance, kmax, w_l, ri_l=None, draw_indices=True,
         chain_weights_prev_l=None):
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
    assembly_atempt = 0
    # array to store product of weights cW1 * cW2 of assembled fragments / pairs from current level
    assembled_chain_weights = np.zeros(kmax)
    
    if ri_l is None:
        # array to store random frame indices of successfully assembled fragments / pairs        
        rs = np.zeros((kmax, 2))

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
        
        # rmsd suportimposition of the subsequent fragmenmt         
        old,new = align.alignto(u1, u2, select=select, weights="mass",
                                tol_mass=5., match_atoms=False)
        assembly_atempt += 1
        if new < rmsd_cut_off:
            # calculate clashes
            clashes = find_clashes(u1 , u2 , index1b=index_clash_l[0] , index2e=index_clash_l[1],
                                  clash_radius=clash_distance)
            
            if clashes < 1:
                ## assemble the subsequent fragments 
                u = merge_universe(u1, u2, *index_merge_l)
                
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
             progress=True):
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
            # if promotion of MD fragment in first level DO NOT define old_pair2
            # index ERROR because in this case, pair_l is no list
            if promotion and isinstance(pair_l, list) == False:
                # subfolder old fragment 1, 2
                old_pair1  = pair_l
                old_pair2  = pair_l
            else:
                # subfolder old fragment 1, 2
                old_pair1 = flatten(pair_l[0])[0]
                old_pair2 = flatten(pair_l[1])[0]

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
            # the domain's own junction (where it first joins its neighboring fragment,
            # i.e. that side of pair_l is still a raw, unmerged fragment) is not a
            # "typical" variation of the general fragment overlap: it has its own
            # overlap length (domain_overlap) and its overlap residues are the
            # domain's real residues, not a synthetic cap. It is looked up via
            # domain_overlap, not overlaps_d, since overlaps_d reserves key 0 for the
            # run's general default overlap.
            domain_first_junction = domain_id is not None and (
                    (old_pair1 == domain_id and isinstance(pair_l[0], list) == False) or
                    (old_pair2 == domain_id and isinstance(pair_l[1], list) == False))

            # residue ovearlap MD fragment
            if domain_first_junction:
                o = domain_overlap
            elif len(flatten(pair_l[1])) == 1:
                o = overlaps_d[old_pair2]
            # overlap grown pairs, always == overlaps[0]
            # -> overlap that differs "corrected" for when growing pairs with MD fragment
            else:
                o = overlap

            pair_overlap0 = o if domain_first_junction else overlap
            pair_capping_groups = False if domain_first_junction else capping_groups

            # get indices for assembly
            index_aln_l, index_clash_l, index_merge_l = get_residue_indices_for_assembly(
                                                    overlap0=pair_overlap0, current_overlap=o,
                                                    capping_groups=pair_capping_groups, last_level=last_level,
                                                    verbose=verbose, strip_cap_nterm=strip_cap_nterm,
                                                    strip_cap_cterm=strip_cap_cterm)

            ## check if for u2 residue 2 that is aligned is ja proline
            res2_u2 = u2.select_atoms('resid {}'.format(u2.atoms.residues[index_aln_l[-1]].resid))
            res2_u2_name = res2_u2.residues.resnames
            if res2_u2_name == 'PRO':
                proline_2nd_posi = True

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
                                  chain_weights_prev_l=chain_weights_prev_l)
                r_l.append(rs)
                assembled_chain_weights.append(acw_mi)
            else:
                rs = r_l[m_i]
                acw_mi = reweighted_fragment_assembly(u1, u2, dire, select, index_clash_l, index_merge_l,
                                  rmsd_cut_off, clash_distance,  kmax=k_max, w_l=w_l, ri_l=rs,
                                  draw_indices=draw_indices, chain_weights_prev_l=chain_weights_prev_l)
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

