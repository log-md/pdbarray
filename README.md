# PDBArray 
Use a python `pdb_string` as a numpy array. 

```
import pdbarray as pa
import numpy as np 

pdb_string = """
ATOM      1  N   ASP A   1      11.411  86.258  12.853  1.00 24.81           N  
ATOM      2  CA  ASP A   1      10.477  85.405  13.585  1.00 25.57           C  
ATOM      3  C   ASP A   1       9.886  84.337  12.667  1.00 22.25           C  
ATOM      4  O   ASP A   1       9.786  84.519  11.454  1.00 23.00           O  
"""

arr = pa.array(pdb_string)
arr = arr - arr.mean(axis=0)          # move to (0,0,0)
arr = arr @ np.linalg.svd(arr)[2].T   # "remove rotation"
print(arr)
```
output
```
HEADER    
TITLE     MDANALYSIS FRAME 0: Created by PDBWriter
CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ASP A   1      -1.411   0.606  -0.055  1.00 24.81      A    N  
ATOM      2  CA  ASP A   1      -0.718  -0.671   0.107  1.00 25.57      A    C  
ATOM      3  C   ASP A   1       0.782  -0.505  -0.126  1.00 22.25      A    C  
ATOM      4  O   ASP A   1       1.347   0.570   0.073  1.00 23.00      A    O  
END
```

## Plan

[ ] support `.cif`
[ ] support `.xyz`
[ ] export `__str__(format='pdb,xyz,cif')`
[ ] support pytorch (to get backprop/gpu)
[ ] support trajectories [num_frames, num_atoms, 3]