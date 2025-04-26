import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.universe import Merge
from io import StringIO
import tempfile
import os
import re
import warnings
import requests
import gzip

class PDBArray(np.ndarray):
    def __new__(cls, pdb_string):
        ''' Create a PDBArray from a PDB string, file path or pdb_code. '''
        # Check if pdb_string is a file path
        if isinstance(pdb_string, str) and os.path.isfile(pdb_string): pdb_string = open(pdb_string, 'r').read()
        if isinstance(pdb_string, str) and len(pdb_string) == 4:  
            if not os.path.exists(f'{pdb_string}.pdb.gz'):  
                response = requests.get(f"https://files.rcsb.org/download/{pdb_string}.pdb.gz")
                open(f'{pdb_string}.pdb.gz', 'wb').write(response.content)
            pdb_string = gzip.decompress(open(f'{pdb_string}.pdb.gz', 'rb').read()).decode('utf-8')

        if "ase" in str(type(pdb_string)).lower():
            from ase.io import read, write
            import io
            temp_io = io.StringIO()
            write(temp_io, pdb_string, format='proteindatabank')
            temp_io.seek(0)
            pdb_string = temp_io.read()
            open('.tmp.pdb', 'w').write(pdb_string)
            universe = mda.Universe('.tmp.pdb', format="PDB")
            #os.remove('.tmp.pdb')

        if isinstance(pdb_string, str): universe = mda.Universe(StringIO(pdb_string), format="PDB")
        if isinstance(pdb_string, mda.Universe): universe = pdb_string
        coords = universe.atoms.positions
        obj = coords.view(cls)
        obj.universe = universe
        return obj

    def numpy(self):
        return self.universe.atoms.positions 
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.universe = getattr(obj, 'universe', None)

    def center(self):
        """Center the coordinates around the origin"""
        return self - self.mean(axis=0)          # remove translation 

    def align(self, to):
        """Align the coordinates to `to`"""
        if self.shape == to.shape:
            self = self.center()
            to = to.center()
            U, _, Vt = np.linalg.svd(self.T @ to)
            self = self @ U @ Vt 
            return self, to 
        else: 
            # shapes differ. 
            # align sequence, grab carbon-alpha find best rotation, apply rotation to all.
            #return self.universe.atoms.align(to.universe.atoms), to 
            aa3to1 = { 'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I', 'LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S', 'THR':'T','VAL':'V','TRP':'W','TYR':'Y' }
            p1, p2 = self, to 
            p1 = p1.center()
            p2 = p2.center()
            s1 = ''.join([aa3to1[r] for r in p1.universe.residues.resnames if r in aa3to1])
            s2 = ''.join([aa3to1[r] for r in p2.universe.residues.resnames if r in aa3to1])
            from Bio.Align import PairwiseAligner
            aligner = PairwiseAligner()
            alignments = aligner.align(s1, s2)
            alignment = alignments[0]  
            print(f"Alignment score: {alignment.score}")

            s1_map = {}
            s1_pos = 0
            for i in range(len(alignment[0])):
                if alignment[0][i] != '-':  
                    s1_map[i] = s1_pos
                    s1_pos += 1
            s2_map = {}
            s2_pos = 0
            for i in range(len(alignment[1])):
                if alignment[1][i] != '-': 
                    s2_map[i] = s2_pos
                    s2_pos += 1

            matching_positions = []
            for i in range(len(alignment[0])):
                if alignment[0][i] != '-' and alignment[1][i] != '-':
                    matching_positions.append((s1_map[i], s2_map[i]))

            # Get CA atoms for matching residues
            idxs1 = []
            idxs2 = []
            for s1_idx, s2_idx in matching_positions:
                res_p1 = p1.universe.residues[s1_idx]
                res_p2 = p2.universe.residues[s2_idx]
                
                ca_p1 = p1.universe.select_atoms(f"resid {res_p1.resid} and name CA")
                ca_p2 = p2.universe.select_atoms(f"resid {res_p2.resid} and name CA")
                
                if len(ca_p1) > 0 and len(ca_p2) > 0:
                    idxs1.append(s1_idx)
                    idxs2.append(s2_idx)

            p1_ca = p1[idxs1]
            p2_ca = p2[idxs2]

            U, _, Vt = np.linalg.svd(p1_ca.T @ p2_ca)
            R = U @ Vt 
            p1 = p1 @ R
            return p1, p2

    def __str__(self):
        """Return PDB string with current coordinates"""
        self.universe.atoms.positions = self.copy()
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w+') as tmp:
            with warnings.catch_warnings():  # hide warnings 
                warnings.simplefilter("ignore")
                self.universe.atoms.write(tmp.name)
            
            tmp.flush()
            with open(tmp.name, 'r') as f: pdb_content = f.read()
            
            # Remove REMARK lines
            pdb_content = re.sub(r'REMARK.*\n', '', pdb_content)
            pdb_content = re.sub(r'\n+', '\n', pdb_content)
            
            return pdb_content.strip()
    
    def __repr__(self):
        return f"PDBArray(shape={self.shape})"

    def concatenate(self, other):
        """Concatenate a list of PDBArrays"""
        return np.concatenate([self, other])

    def protein(self):
        """Return a PDBArray of the protein"""
        return array(Merge(self.universe.select_atoms("protein")))

    def rna(self):
        """Return a PDBArray of the RNA"""
        return array(Merge(self.universe.select_atoms("nucleic and name P and resname DA DT DG DC")))

    def dna(self):
        """Return a PDBArray of the nucleic acids"""
        return array(Merge(self.universe.select_atoms("nucleic and name P and resname A U G C")))

    def water(self):
        """Return a PDBArray of the water molecules"""
        return array(Merge(self.universe.select_atoms("resname HOH SOL WAT TIP3 TIP4")))

    def energy(self, calc):
        """Return the energy of the PDBArray"""
        # read atoms from pdb_string, add calc and compute enregy 
        import io 
        from ase.io import read 
        atoms = read(io.StringIO(addHs(str(self))), format='proteindatabank')
        atoms.calc = calc
        atoms.pbc = False
        a = atoms.get_potential_energy()
        energy = float(a)
        return energy 

def array(pdb_string):
    """Factory function to create a PDBArray from a PDB string"""
    return PDBArray(pdb_string)

def rmsd(a, b): return np.sqrt(np.mean(np.sum((a - b)**2, axis=1)))

def addHs(pdb_string):
    """Add hydrogens to a PDB structure.
    
    Uses pdbfixer if available, otherwise falls back to rdkit.
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        import io
        fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))
        fixer.addMissingHydrogens(7.0)  # pH 7.0
        output = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, output)
        return output.getvalue()
    except ImportError:
        from rdkit import Chem
        import io
        mol = Chem.MolFromPDBBlock(pdb_string, removeHs=False, sanitize=False)
        mol = Chem.AddHs(mol, addCoords=True)
        out = io.StringIO()
        out.write(Chem.MolToPDBBlock(mol))
        return out.getvalue()
