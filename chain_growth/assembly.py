#!/usr/bin/env python3
"""
assembly
--------
pure, per-pair geometry and selection helpers used by both plain and reweighted
hierarchical chain growth: choosing which residues/atoms to align, clash-check, and
merge for a single fragment-pair join, and performing that join. Nothing in this
module runs a multi-level growth loop or touches multiprocessing -- see hcg_fct.py
and rhcg_fct.py for that.
"""
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
import MDAnalysis.analysis.distances as distances
from chain_growth.hcg_list import flatten

# MDAnalysis's own `type`-based hydrogen guessing is unreliable and inconsistent
# across the naming conventions actually seen in this project's own data: it
# under-excludes (finds none at all) for at least one real GROMACS-written PDB, and
# over-excludes (misclassifies genuine heavy atoms as hydrogen) for this project's
# own AMBER-style MD fragment library. Matching the atom *name* directly instead
# covers both conventions actually observed here: a leading "H" (standard/GROMACS,
# e.g. "H", "HA", "HB1", "H01") and AMBER's digit-prefixed equivalent-hydrogen naming
# for chemically-equivalent hydrogens (e.g. "1HD1", "2HD1", "3HD1").
HYDROGEN_NAME_SELECTION = 'name H* or name [123]H*'

# reserved segid tag (PDB's segid column is limited to 4 characters) marking a
# prepared domain fragment's own atoms (see chain_growth.fragment_list's
# prepare_domain_fragment). Unlike resid, segid is left untouched by this module's
# merges and residue renumbering, so it reliably identifies the domain's atoms at
# every level of a run, however deep the domain ends up embedded within a growing
# merged chain -- used by find_clashes's optional domain-surface-only optimization
# (see chain_growth.fragment_list.compute_domain_surface_mask)
DOMAIN_SEGID = 'DOM'

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
    # "segid {} and prop tempfactor > 0.5" matches only atoms explicitly marked
    # buried by compute_domain_surface_mask + prepare_domain_fragment's
    # surface_mask (see their docstrings); harmless no-op otherwise, since every
    # atom defaults to tempfactor=0.0 (kept) unless a domain fragment was prepared
    # with that option. MDAnalysis's selection language requires "prop" for numeric
    # (as opposed to exact-match) property comparisons.
    l1 = u1.select_atoms(
        "protein and not ({}) and not (segid {} and prop tempfactor > 0.5) "
        "and not (resid {} and backbone) and not resid {}:{}".format(
                                                    HYDROGEN_NAME_SELECTION, DOMAIN_SEGID,
                                                    u1.atoms.residues[index1b].resid,
                                                    u1.atoms.residues[index1b+1].resid,
                                                    u1.atoms.residues[index1e].resid))

    # atom selection of u2 to scan for clashes
    l2 = u2.select_atoms(
        "protein and not ({}) and not (segid {} and prop tempfactor > 0.5) "
        "and not resid 1:{} and not (resid {} and backbone)".format(
                                                    HYDROGEN_NAME_SELECTION, DOMAIN_SEGID,
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

def _resolve_old_pairs(pair_l, promotion):
    """ get the fragment ids of the two (sub-)fragments a pair_l entry joins.

    Shared by `_loop_func` and `reweighted_hierarchical_chain_growth`: when a
    fragment was promoted unpaired from the previous level, `pair_l` is a bare
    fragment id rather than a `[left, right]` pair, and both "sides" are that same
    id.

    Parameters
    ----------
    pair_l : fragment id, or [left, right] pair of (possibly nested) fragment ids
    promotion : boolean
        whether this level has a promoted (unpaired) fragment, see `pair_fragments`

    Returns
    -------
    old_pair1, old_pair2 : fragment ids
        the (leaf) fragment id on each side, as used to look up MDfragments folders
    """
    if promotion and isinstance(pair_l, list) == False:
        return pair_l, pair_l
    else:
        return flatten(pair_l[0])[0], flatten(pair_l[1])[0]


def _resolve_pair_overlap(pair_l, old_pair1, old_pair2, domain_id, domain_overlap,
                          overlap, overlaps_d, capping_groups):
    """ resolve the overlap and capping-groups convention for joining a pair_l entry.

    Shared by `_loop_func` and `reweighted_hierarchical_chain_growth`. The domain's
    own junction (where it first joins its neighboring fragment, i.e. that side of
    pair_l is still a raw, unmerged fragment) is not a "typical" variation of the
    general fragment overlap: it has its own overlap length (domain_overlap) and its
    overlap residues are the domain's real residues, not a synthetic cap. It is
    looked up via domain_overlap, not overlaps_d, since overlaps_d reserves key 0
    for the run's general default overlap.

    Returns
    -------
    o : integer
        overlap between the fragments being joined right now
    pair_overlap0 : integer
        "initial"/reference overlap convention (see get_residue_indices_for_assembly)
        to use for this join
    pair_capping_groups : boolean
        capping_groups convention to use for this join
    """
    domain_first_junction = domain_id is not None and (
            (old_pair1 == domain_id and isinstance(pair_l[0], list) == False) or
            (old_pair2 == domain_id and isinstance(pair_l[1], list) == False))

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
    return o, pair_overlap0, pair_capping_groups


def _is_proline_at_alignment_end(u2, index_aln_l):
    """ whether the residue of u2 that ends up 2nd in the alignment selection is a
    proline (lacks the amide hydrogen translate_concept otherwise aligns on).

    Shared by `_loop_func` and `reweighted_hierarchical_chain_growth`.
    """
    res2_u2 = u2.select_atoms('resid {}'.format(u2.atoms.residues[index_aln_l[-1]].resid))
    return bool(res2_u2.residues.resnames == 'PRO')


def _unify_domain_segid(u, domain_segid):
    """ relabel every atom's segid (and chainID, if present) to match the
    non-domain atoms' own value, erasing the domain/non-domain distinction from a
    fully-assembled chain's output.

    `prepare_domain_fragment` tags the domain's own atoms with `domain_segid` (see
    `DOMAIN_SEGID`) so `find_clashes` can reliably keep excluding buried domain
    atoms at every level, however deep the domain ends up embedded within a
    growing merged chain. Once assembly is finished, though, the domain and the
    grown IDR are one single, covalently continuous molecule -- carrying that
    internal bookkeeping tag (and whatever separate chainID the domain's original
    source PDB happened to use) into the finished output would misrepresent it as
    two separate molecules/chains to anything reading segid or chainID. This is
    meant to be called only on that final, no-more-clash-checks-coming merge (see
    its callers' `relabel_domain_segid` parameter), never on an intermediate one.

    Parameters
    ----------
    u : universe
        a freshly merge_universe'd universe (modified in place)
    domain_segid : string
        the segid marking the domain's own atoms, i.e. `DOMAIN_SEGID`
    """
    domain_mask = u.atoms.segids == domain_segid
    if not domain_mask.any() or domain_mask.all():
        # nothing to unify: either no domain atoms present (this merge never
        # touched the domain), or every atom is domain-tagged (shouldn't happen for
        # a real domain-attachment run, since the whole point is joining the domain
        # to an IDR) -- either way, silently do nothing rather than guess
        return
    non_domain = u.atoms[~domain_mask]
    u.atoms.segments.segids = non_domain.segids[0]
    if hasattr(u.atoms, 'chainIDs'):
        u.atoms.chainIDs = non_domain.chainIDs[0]


def _attempt_merge(u1, u2, select, index_clash_l, index_merge_l, rmsd_cut_off, clash_distance,
                   relabel_domain_segid=None):
    """ attempt one rejection-sampling trial: align u1/u2's currently-loaded frames,
    and merge them if both the RMSD and clash-count criteria pass.

    Shared by `fragment_assembly` and `reweighted_fragment_assembly` -- the one
    difference between them is how the next frame to try is drawn (uniformly vs.
    weighted), not this per-trial accept/reject/merge logic.

    Parameters
    ----------
    relabel_domain_segid : string, optional
        when given (as `DOMAIN_SEGID`), a successful merge gets its domain/non-domain
        segid and chainID distinction erased via `_unify_domain_segid` before being
        returned -- callers pass this only for the truly final merge of a
        domain-attachment run (see `_unify_domain_segid`'s own docstring for why).
        The default is None (no relabeling, unchanged pre-existing behavior).

    Returns
    -------
    u : universe, or None
        the merged universe if this trial was accepted, None if rejected (RMSD too
        high, or a clash was found)
    """
    old, new = align.alignto(u1, u2, select=select, weights="mass",
                             tol_mass=5., match_atoms=False)
    if new < rmsd_cut_off:
        clashes = find_clashes(u1, u2, index1b=index_clash_l[0], index2e=index_clash_l[1],
                              clash_radius=clash_distance)
        if clashes < 1:
            u = merge_universe(u1, u2, *index_merge_l)
            if relabel_domain_segid is not None:
                _unify_domain_segid(u, relabel_domain_segid)
            return u
    return None
