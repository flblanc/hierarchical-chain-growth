#!/usr/bin/env python3
"""
fragmentList
-----------
prepare a list of fragments with desired length + overlap from input sequence
== IDP/ IDR to grow using (r)hcg
"""
import MDAnalysis as mda
import os

class get_sequence:
    def __init__(self, input_sequence, NA):
        """
        Parameters
        ----------
        input_sequence : string
            Either path to sequence file (PDB or fasta) or sequence as string in one letter
            amino acid code / nucleic acid.

        """
        ## dictionairy to translate one letter to three letter amino acid code
        self.three_letter ={'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN', 'D':'ASP', 
                            'N':'ASN', 'H':'HIS', 'W':'TRP', 'F':'PHE', 'Y':'TYR', 'R':'ARG', 'K':'LYS', 
                            'S':'SER', 'T':'THR', 'M':'MET', 'A':'ALA',    'G':'GLY', 'P':'PRO', 'C':'CYS'}
        self.input_sequence = input_sequence
        self.NA = NA # boolean
        

    def _from_pdb(self):
        """
        Returns
        -------
        sequence_list : list
            created list with nucleic acid or amino acid sequence from PDB file
        """

        ## make universe with PDB file
        u = mda.Universe(self.input_sequence)
        ## get nucleic acid sequence - as an array
        sequence_list = u.residues.resnames
        sequence_list = sequence_list.tolist()

        return sequence_list

    def _from_fasta(self):
        """
        Returns
        -------
        sequence_list : list
            created list with nucleic acid  or amino acid sequence from fasta file
        """

        ## open fasta file and read lines
        NA = self.NA
        with open(self.input_sequence) as data:
            read = data.readlines()    
            if NA:
                sequence_list = [sl.upper() for i, l in enumerate(read[1:])
                                             for sl in l if sl!='\n']        
            else:
                sequence_list = [self.three_letter[sl.upper()] for i, l in enumerate(read[1:])
                                             for sl in l if sl!='\n']        
        return sequence_list

    def _from_string(self):
        """         
        Returns
        -------
        sequence_list : list
            created list with nucleic acid  or amino acid sequence from from string
        """
        NA = self.NA
        if NA:
            return [si.upper() for si in self.input_sequence]
        else:
            return [self.three_letter[si.upper()] for si in self.input_sequence]

    def get_sequence_list(self):
        input_basename = os.path.basename(self.input_sequence).split('.')
        try:
            if len(input_basename) == 2:
                if input_basename[-1].upper() == 'FASTA':
                    sequence_list = self._from_fasta()
                elif input_basename[-1].upper() == 'PDB':
                    sequence_list = self._from_pdb()

            elif len(input_basename) == 1:
                sequence_list = self._from_string()
            return sequence_list
        except (NameError , TypeError):
            print('Invalid sequence input. Allowed: ".fasta" ,".pdb" or string')


def generate_fragment_list(input_sequence, fragment_length, overlap, NA=False, n_to_c_term = True):
    """ generates list of fragments of desired length, with desired overlap between subsequent fragments

    Parameters
    ----------
    input_sequence : string
        path to PDB or fasta file with the sequence of the polymer
        or amino acid sequence in one letter code / nucleic acid sequence as string
        NOTE: functions are NOT case sensitive
    fragment_length : integer
        desired fragment length 
    overlap : integer
        desired fragment overlap 
        
    Returns
    -------
    fragment_list : list
            list with fragments of desired length and overlap
    overlaps : dictionary
        dict of overlaps between subsequent fragments
        key: number fragment, value: overlap
    """
    
    ## get sequence from PDB or fasta file (amino acid sequence in three letter code or nucleic acid sequence)
    sequence_list = get_sequence(input_sequence, NA).get_sequence_list()
    
    ## generate list of fragments with desired length
    fragment_list = []
    overlaps = {}
    ## counnter fragment 
    count = 0
    ## loop through range of the sequence length with as step size==fragment_length-overlap -> generate desired overlap
    steps = fragment_length-overlap
    for i in range(0, len(sequence_list) , steps): 
        ## if generated fragment has desired length as defined with fragment_length
        ## append fragment to fragment_list and "typical" overlap to overlaps
        if len(sequence_list[i:i+fragment_length]) == fragment_length:
            fragment_list.append(sequence_list[i:i+fragment_length])
            overlaps[count]=overlap
            count += 1
        ## if reached end of the sequence and last generated fragment has not desired length as defined with fragment_length
        ## go back in sequence such that the length of the new fragment == fragment_length and append to fragment_list
        ## re-evaluate new overlap with previous fragment and apped to overlaps
        else: 
            stepsBackInSeq = fragment_length - len(sequence_list[i:])
            newOverlap = overlap + stepsBackInSeq
            fragment_list.append(sequence_list[i-stepsBackInSeq:])
            overlaps[count]=newOverlap
            count += 1
    if overlaps[count-1] >= fragment_length:
        fragment_list.pop(-1)
        del overlaps[count-1]
    if n_to_c_term == False:
        fragment_list.reverse()

    return fragment_list, overlaps


def prepare_domain_fragment(domain_pdb, out_dir):
    """ prepare a rigid folded domain PDB as a single-frame MD-fragment folder, matching
    the "pair0.pdb" + "pair.xtc" convention every other MD fragment folder uses, so it
    can be loaded like any other fragment during hierarchical chain growth.

    Parameters
    ----------
    domain_pdb : string
        path to the folded domain's PDB structure (a single, rigid conformation)
    out_dir : string
        folder to write "pair0.pdb" and "pair.xtc" into, e.g. "MDfragments/<domain_id>"

    Returns
    -------
    None.
    """
    os.makedirs(out_dir, exist_ok=True)
    u = mda.Universe(domain_pdb)
    u.atoms.write('{}/pair0.pdb'.format(out_dir))
    u.atoms.write('{}/pair.xtc'.format(out_dir), frames='all')


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
        'N' to attach the domain at the N-terminus (domain placed first), 'C' to
        attach it at the C-terminus (domain placed last)

    Returns
    -------
    fragment_ids : list
        ordered list of fragment ids, including `domain_id`, to pass as `fragment_ids`
        to `chain_growth.hcg_list.make_hcl_l`
    """
    fragment_ids = list(range(n_fragments))
    if terminus == 'N':
        return [domain_id] + fragment_ids
    elif terminus == 'C':
        return fragment_ids + [domain_id]
    else:
        raise ValueError("terminus must be 'N' or 'C'")