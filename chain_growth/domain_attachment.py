#!/usr/bin/env python3
"""
domain_attachment
------------------
preparing and attaching a rigid folded domain to an IDR grown by hierarchical chain
growth: turning a domain PDB into an MD-fragment folder, splicing it into a
fragment-id list, and (when growing from the generic pre-sampled dimer library)
building the domain-junction fragment purely from that library.

See `chain_growth.fragment_list` for the ordinary (domain-free) fragment-list
preparation this extends, and the README's domain-attachment sections for the
worked examples.
"""
import MDAnalysis as mda
import numpy as np
import os
import shutil


def prepare_domain_fragment(domain_pdb, out_dir, surface_mask=None):
    """ prepare a rigid folded domain PDB as a single-frame MD-fragment folder, matching
    the "pair0.pdb" + "pair.xtc" convention every other MD fragment folder uses, so it
    can be loaded like any other fragment during hierarchical chain growth.

    Parameters
    ----------
    domain_pdb : string
        path to the folded domain's PDB structure (a single, rigid conformation)
    out_dir : string
        folder to write "pair0.pdb" and "pair.xtc" into, e.g. "MDfragments/<domain_id>"
    surface_mask : numpy.ndarray of bool, optional
        the result of `compute_domain_surface_mask(domain_pdb)`, one entry per atom
        (True = exposed/keep, False = buried/safe to exclude from clash-checks). If
        given, this gets baked into the written fragment's own B-factor column (1.0
        for exposed, 0.0 for buried) -- a column `find_clashes` always reads for the
        domain's own atoms (identified via `DOMAIN_SEGID`) to skip buried ones,
        regardless of how many levels deep the domain ends up embedded within a
        growing merged chain. B-factor survives every merge and residue-renumbering
        step in this package untouched, unlike resid. The default is None (every
        domain atom is checked, i.e. today's existing behavior).

    Returns
    -------
    None.

    Raises
    ------
    ValueError
        if `domain_pdb` has no hydrogen atoms, or has hydrogens but none literally
        named "H" (the backbone amide hydrogen). `hierarchical_chain_growth`'s
        alignment at the domain junction hardcodes that exact atom name, matching the
        MD-simulated fragment libraries the domain gets joined to (which always use
        it) -- a bare crystal structure, cryo-EM model, or structure prediction
        typically has no hydrogens at all, and generic protonation tools sometimes
        name them differently (e.g. sequentially numbered "H01", "H02", ...). Either
        way, re-protonate with a tool that follows standard PDB/AMBER naming (e.g.
        tleap) -- or rename the offending atoms -- and re-run.
    """
    # imported lazily to avoid a module-level dependency cycle with hcg_fct/assembly
    # (see build_domain_junction_fragment for the same pattern)
    from chain_growth.assembly import DOMAIN_SEGID, HYDROGEN_NAME_SELECTION

    u = mda.Universe(domain_pdb)
    # name-based (not type-based): MDAnalysis's `type` is a *guessed* classification
    # and is unreliable for at least some real-world PDBs (e.g. observed returning 0
    # "type H" atoms for a GROMACS/AMBER-written structure that plainly has H, HA,
    # HB1, ... in its atom names) -- matching the PDB's own atom-name field directly
    # (HYDROGEN_NAME_SELECTION) is what the rest of this check (and the code it
    # protects) relies on instead
    if len(u.select_atoms(HYDROGEN_NAME_SELECTION)) == 0:
        raise ValueError(
            "domain_pdb '{}' has no hydrogen atoms. hierarchical_chain_growth's "
            "alignment at the domain junction requires the domain structure to carry "
            "explicit hydrogens, matching the MD-simulated fragment libraries it's "
            "joined to -- a bare crystal structure, cryo-EM model, or structure "
            "prediction typically has none. Add hydrogens first (e.g. with tleap, "
            "pdb2pqr, or PyMOL/Reduce) and re-run.".format(domain_pdb))
    if len(u.select_atoms('name H')) == 0:
        raise ValueError(
            "domain_pdb '{}' has hydrogen atoms, but none literally named \"H\" (the "
            "backbone amide hydrogen). hierarchical_chain_growth's alignment at the "
            "domain junction hardcodes that exact atom name, matching the MD-simulated "
            "fragment libraries the domain gets joined to. Re-protonate with a tool "
            "that follows standard PDB/AMBER naming (e.g. tleap), or rename the "
            "offending atoms, and re-run.".format(domain_pdb))
    # tag with DOMAIN_SEGID unconditionally (harmless if never used): lets
    # hierarchical_chain_growth's optional domain_surface_mask optimization (see
    # compute_domain_surface_mask) reliably find the domain's own atoms at any
    # level, without needing this fragment re-prepared later just to enable it
    if not hasattr(u.atoms, 'segids'):
        u.add_TopologyAttr('segid')
    u.atoms.segments.segids = DOMAIN_SEGID
    if surface_mask is not None:
        if len(surface_mask) != len(u.atoms):
            raise ValueError(
                "surface_mask has {} entries but domain_pdb '{}' has {} atoms -- "
                "must be the result of compute_domain_surface_mask(domain_pdb), for "
                "this same file".format(len(surface_mask), domain_pdb, len(u.atoms)))
        if not hasattr(u.atoms, 'tempfactors'):
            u.add_TopologyAttr('tempfactors')
        # 1.0 = buried (explicitly excluded by find_clashes), 0.0 = exposed. This
        # way round, not the reverse: every atom (domain or not) defaults to
        # tempfactor=0.0 when never explicitly set, so "buried" must be the
        # non-default value -- otherwise every domain prepared without surface_mask
        # would look entirely buried and never get clash-checked at all
        u.atoms.tempfactors = (~np.asarray(surface_mask, dtype=bool)).astype(float)
    os.makedirs(out_dir, exist_ok=True)
    u.atoms.write('{}/pair0.pdb'.format(out_dir))
    u.atoms.write('{}/pair.xtc'.format(out_dir), frames='all')


def compute_domain_surface_mask(domain_pdb, sasa_threshold=1.0, probe_radius=1.4):
    """ classify a prepared domain's own (heavy) atoms as solvent-exposed or buried,
    for hierarchical_chain_growth's `domain_surface_mask` parameter: a
    performance-only optimization that excludes buried atoms from clash-checking
    against the domain, since they're geometrically unreachable by an external,
    non-penetrating IDR chain without first clashing with a more exposed atom.

    Since the domain is rigid (a single, fixed conformation for the whole run), this
    only needs to be computed once and reused for every clash-check at every level,
    however deep the domain ends up embedded within a growing merged chain -- its
    own atoms stay identifiable throughout via `DOMAIN_SEGID` (see
    `prepare_domain_fragment`).

    Uses true solvent-accessible surface area (SASA), via the separate `freesasa`
    package (`pip install freesasa`). A cheaper, dependency-free approximation
    (counting nearby atoms as a burial proxy) was tried and rejected during
    development: calibrated to be safe (no missed clashes), it excluded too few
    atoms to be worth the added complexity; calibrated to exclude a useful
    fraction, it risked classifying genuinely exposed atoms (tens of square
    Angstrom of real solvent-accessible area) as buried -- a real risk of silently
    missing a genuine clash, not just a performance tradeoff.

    Parameters
    ----------
    domain_pdb : string
        path to the folded domain's PDB structure (the same one passed to
        `prepare_domain_fragment`)
    sasa_threshold : float, optional
        an atom is classified as buried (safe to exclude) if its own per-atom SASA,
        in square Angstrom, is below this value. The default is 1.0 -- deliberately
        small/conservative, since this only needs to identify atoms with
        essentially zero solvent exposure, not merely "less exposed than average"
    probe_radius : float, optional
        solvent probe radius, in Angstrom, passed to freesasa. The default is 1.4
        (water)

    Returns
    -------
    exposed_mask : numpy.ndarray of bool
        one entry per atom in `mda.Universe(domain_pdb).atoms`, in that same order;
        True means "keep this atom in clash-checks" (exposed), False means "safe to
        exclude" (buried). Hydrogens are always True here (SASA is computed on heavy
        atoms only) since they're already excluded from clash-checking by
        `chain_growth.assembly.HYDROGEN_NAME_SELECTION` regardless

    Raises
    ------
    ImportError
        if the freesasa package isn't installed
    """
    # imported lazily to avoid a module-level dependency cycle with hcg_fct/assembly
    # (see build_domain_junction_fragment for the same pattern)
    from chain_growth.assembly import HYDROGEN_NAME_SELECTION
    try:
        import freesasa
    except ImportError as e:
        raise ImportError(
            "compute_domain_surface_mask requires the freesasa package: "
            "pip install freesasa") from e

    u = mda.Universe(domain_pdb)
    heavy = u.select_atoms('not ({})'.format(HYDROGEN_NAME_SELECTION))

    structure = freesasa.Structure()
    for atom in heavy:
        structure.addAtom(atom.name, atom.resname, str(atom.resid), 'A',
                           *[float(x) for x in atom.position])
    params = freesasa.Parameters({'probe-radius': probe_radius})
    result = freesasa.calc(structure, params)
    sasa = np.array([result.atomArea(i) for i in range(len(heavy))])
    buried_heavy = sasa < sasa_threshold

    exposed_mask = np.ones(len(u.atoms), dtype=bool)
    exposed_mask[heavy.indices] = ~buried_heavy
    return exposed_mask


def add_domain_to_fragment_list(n_fragments, domain_id, terminus):
    """ build the ordered list of fragment ids for hierarchical chain growth with a
    rigid folded domain attached at one terminus.

    This only orders the fragment ids so the domain is placed at the correct end
    without renumbering any of the ordinary MD fragments' folders. The domain's
    overlap with its neighboring fragment is passed separately, as the
    `domain_overlap` parameter of `hierarchical_chain_growth` /
    `reweighted_hierarchical_chain_growth` -- it is used only for the domain's own
    junction and does not touch `overlaps_d`, which reserves key `0` for the run's
    general default overlap.

    Parameters
    ----------
    n_fragments : integer
        number of ordinary IDR MD fragments (== len(fragment_list) returned by
        `generate_fragment_list`); ordinary fragment ids are assumed to be `0, ...,
        n_fragments - 1`, matching their MDfragments folder names
    domain_id : hashable
        id of the folded domain's MDfragments folder; must not collide with `0, ...,
        n_fragments - 1`
    terminus : string
        which of the domain's own termini the IDR attaches to. 'N' attaches the IDR to
        the domain's N-terminus (the domain's own N-terminal residues form the
        junction, so the domain is placed *last* in the returned list, after the
        IDR); 'C' attaches the IDR to the domain's C-terminus (the domain's own
        C-terminal residues form the junction, so the domain is placed *first*,
        before the IDR)

    Returns
    -------
    fragment_ids : list
        ordered list of fragment ids, including `domain_id`, to pass as `fragment_ids`
        to `chain_growth.hcg_list.make_hcl_l`
    """
    fragment_ids = list(range(n_fragments))
    if terminus == 'N':
        return fragment_ids + [domain_id]
    elif terminus == 'C':
        return [domain_id] + fragment_ids
    else:
        raise ValueError("terminus must be 'N' or 'C'")


# three-letter residue order used to index the pre-sampled, generic dimer fragment
# library (400 folders "0".."399", covering all ordered pairs of the 20 standard amino
# acids; see README's "Web application of HCG" section): folder index = 20*i + j, where
# i, j are this list's positions of the dimer's first and second residue, respectively
DIMER_LIBRARY_RESIDUE_ORDER = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'THR', 'SER', 'CYS',
                               'GLN', 'ASN', 'GLU', 'ASP', 'LYS', 'TRP', 'ARG', 'TYR',
                               'PHE', 'HIS', 'PRO', 'MET']


def dimer_library_fragment_dir(dimer_library, resname_a, resname_b):
    """ folder of the dimer fragment "resname_a - resname_b" in the pre-sampled,
    generic dimer fragment library.

    Parameters
    ----------
    dimer_library : string
        path to the unpacked dimer library's "dimerLibrary" folder (400 folders
        "0".."399", each with "fragment.pdb"/"fragment.xtc")
    resname_a : string
        three-letter code of the dimer's first residue
    resname_b : string
        three-letter code of the dimer's second residue

    Returns
    -------
    path : string
        path to the folder containing that dimer's "fragment.pdb"/"fragment.xtc"
    """
    index = (len(DIMER_LIBRARY_RESIDUE_ORDER) * DIMER_LIBRARY_RESIDUE_ORDER.index(resname_a)
             + DIMER_LIBRARY_RESIDUE_ORDER.index(resname_b))
    return os.path.join(dimer_library, str(index))


def build_domain_junction_fragment(dimer_library, domain_pdb, idr_boundary_residue, terminus,
                                    path0, junction_id, kmax, rmsd_cut_off=0.6, clash_distance=2.0,
                                    capping_groups=True, verbose=False):
    """ build a domain-junction MD-fragment folder purely from the pre-sampled generic
    dimer fragment library, with no new simulation.

    `hierarchical_chain_growth`'s domain junction always assembles without a capping
    group (the domain's real terminus has no synthetic cap), and its alignment math
    only supports a 1-residue overlap when a capping group is present -- see
    `chain_growth.assembly.get_residue_indices_for_assembly`'s own source comment ("having
    1 overlapping residue ... does work only with headgroup"). So attaching an IDR
    directly to the domain needs `domain_overlap=2`, which in turn needs a
    domain-adjacent fragment with at least 3 real residues (2 shared with the domain,
    at least 1 newly contributed) -- more than any single entry of the dimer library's
    2-residue dimers has on its own.

    This builds that 3-residue fragment without any new simulation. Every one of the
    library's 400 residue-pair combinations already exists in it, so the two ordinary
    dimers "[domain's own last 2 residues]" and "[domain's own last residue, the IDR's
    boundary residue]" (or their mirror image, for N-terminal attachment) can simply be
    looked up there, and merged with the completely ordinary (non-domain)
    overlap=1/capping_groups=True mechanism -- the same as any other internal HCG join.
    The result is a real, multi-frame (kmax-frame) sampled ensemble, not a single rigid
    conformation, ready to use as `junction_id` in a `hierarchical_chain_growth` call
    where `domain_id` is that domain's own prepared fragment (see
    `prepare_domain_fragment`) with `domain_overlap=2`.

    Parameters
    ----------
    dimer_library : string
        path to the unpacked dimer library's "dimerLibrary" folder
    domain_pdb : string
        path to the folded domain's PDB structure. Only its residue sequence is read
        here -- its coordinates are irrelevant to this function, since the domain's
        real, rigid structure is attached separately (via `domain_id`) in the main
        `hierarchical_chain_growth` call
    idr_boundary_residue : string
        three-letter code of the IDR's own residue that will neighbor the domain:
        `fragment_l[0][0]` for `terminus='C'`, `fragment_l[-1][-1]` for `terminus='N'`
        (`fragment_l` as returned by `generate_fragment_list` for the ordinary,
        dimer-library-grown part of the IDR)
    terminus : string
        which of the domain's own termini the IDR attaches to, exactly as in
        `add_domain_to_fragment_list`: 'N' or 'C'
    path0 : string
        local project directory whose "MDfragments/<junction_id>" folder this writes
        to (also used, under temporary ids, for the two intermediate library dimers
        this merges together)
    junction_id : hashable
        id to give the resulting junction fragment's MDfragments folder
    kmax : integer
        number of frames to sample for the junction fragment's ensemble
    rmsd_cut_off : float, optional
        cut-off for the RMSD of the two library dimers' alignment. The default is 0.6
    clash_distance : float, optional
        max. allowed distance between atoms. The default is 2.0
    capping_groups : boolean, optional
        whether the dimer library's own fragments carry end-capping groups. The
        default is True (matching the generic dimer library)
    verbose : boolean, optional
        see `hierarchical_chain_growth`. The default is False

    Returns
    -------
    None.
    """
    # imported here (not at module level) to avoid a needless hard dependency between
    # fragment-list preparation and the assembly engine for every other use of this
    # module, and to avoid a circular import (chain_growth.fragment_list imports this
    # module's public names for backward compatibility)
    from chain_growth.fragment_list import get_sequence
    from chain_growth.hcg_fct import hierarchical_chain_growth
    from chain_growth.hcg_list import make_hcl_l

    domain_sequence = get_sequence(domain_pdb, NA=False).get_sequence_list()

    if terminus == 'C':
        # domain's own last 2 residues are consumed at the junction; the IDR's
        # boundary residue continues from the second of those two, matching the
        # general overlap=1 convention used throughout the dimer-library-grown IDR
        res_a, res_b, res_c = domain_sequence[-2], domain_sequence[-1], idr_boundary_residue
        # the domain-facing end is this fragment's own leading end -- strip its
        # synthetic cap there; the free end (facing the ordinary IDR) keeps its
        # normal cap, needed as the alignment anchor for that later, ordinary join
        junction_strip_nterm, junction_strip_cterm = True, False
    elif terminus == 'N':
        res_a, res_b, res_c = idr_boundary_residue, domain_sequence[0], domain_sequence[1]
        # mirror image: the domain-facing end is this fragment's own trailing end
        junction_strip_nterm, junction_strip_cterm = False, True
    else:
        raise ValueError("terminus must be 'N' or 'C'")

    mdfragments_dir = os.path.join(path0, 'MDfragments')
    input_ids = ['{}_input0'.format(junction_id), '{}_input1'.format(junction_id)]
    for input_id, (r1, r2) in zip(input_ids, [(res_a, res_b), (res_b, res_c)]):
        dst = os.path.join(mdfragments_dir, str(input_id))
        os.makedirs(dst, exist_ok=True)
        src = dimer_library_fragment_dir(dimer_library, r1, r2)
        for fname, linkname in (('fragment.pdb', 'pair0.pdb'), ('fragment.xtc', 'pair.xtc')):
            link = os.path.join(dst, linkname)
            if not os.path.exists(link):
                os.symlink(os.path.join(src, fname), link)

    # ordinary (non-domain) merge of the two library dimers -- identical to any other
    # internal HCG join, no domain-specific handling involved. overlaps_d[0] is read as
    # the run's general default overlap; overlaps_d[input_ids[1]] as the specific
    # overlap for this one (raw, first-ever) fragment pairing -- see hierarchical_chain_growth.
    # strip_cap_nterm/strip_cap_cterm must be passed explicitly here: this build is
    # always a single-level hierarchical_chain_growth run, so last_level=True and
    # they'd otherwise both default to capping_groups (True), stripping the cap off
    # BOTH ends -- silently discarding the free end's cap along with it, which is
    # needed as the alignment anchor for this fragment's later, ordinary join with the
    # rest of the dimer-library-grown IDR (see module docstring). strip_terminal_caps
    # must be disabled for the same reason: from this internal call's own point of
    # view last_level=True too, but this junction fragment isn't the whole pipeline's
    # real final output -- its name-based safety net would otherwise strip that same
    # needed cap right back off regardless of strip_cap_nterm/strip_cap_cterm.
    hcg_l, promo_l = make_hcl_l(2, fragment_ids=input_ids)
    build_path = os.path.join(path0, '_junction_build_{}'.format(junction_id))
    hierarchical_chain_growth(
        hcg_l, promo_l, overlaps_d={0: 1, input_ids[1]: 1}, path0=path0, path=build_path,
        kmax=kmax, capping_groups=capping_groups, rmsd_cut_off=rmsd_cut_off,
        clash_distance=clash_distance, verbose=verbose, num_threads=1,
        strip_cap_nterm=junction_strip_nterm, strip_cap_cterm=junction_strip_cterm,
        strip_terminal_caps=False)

    junction_dst = os.path.join(mdfragments_dir, str(junction_id))
    os.makedirs(junction_dst, exist_ok=True)
    built_dir = os.path.join(build_path, '1', str(input_ids[0]))
    shutil.copyfile(os.path.join(built_dir, 'pair0.pdb'), os.path.join(junction_dst, 'pair0.pdb'))
    shutil.copyfile(os.path.join(built_dir, 'pair.xtc'), os.path.join(junction_dst, 'pair.xtc'))
