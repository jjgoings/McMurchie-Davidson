This folder contains a set of function to generate eigenvalues
and eigenvectors to use in Huckel Molecular Orbital theory qualitative calculations.
There are three main functions.   

The function makemolskeleton(coordinates) takes the 3D atomic coordinates that are passed as a string.
These coordinates can be taken from gview after building AND cleaning up the molecular 
structure (this step is important to get reasonable bond lengths). Since gview coordinate string also contains
spin multiplicity and charge, this function removes them from the string, so the user does not need to do that.
It removes all the hydrogen atoms, since only the carbon skeleton is needed for huckel calculations.
It uses these coordinates to define edges and vertices from calculated bond lengths.  

The function adjacencymatrix(edges,vertices) takes the edges and vertices build by the make skeleton function
and creates an adjacency matrix of the carbon skeleton. It returns an array at the end

The functionn huckel(coordinates) will make function calls to the two functions described above (so the user does not 
need to call these functions individually but will need to import them in their own file.)
and calculate the 
eigenvalues and eigenvectors of the adjacency matrix, sort the eigenvalues in ascending order so the user can use 
readily
use them for Huckel calculation, and will get printed messages in a saved file keeping track of the existence and number 
of degenerate states.

An example on how to run the function is shown under the if __name__="__main__":
block.

RUNNING IN YOUR OWN FILE
If the user is running the huckel(coordinates) function in their own file, they will need to import 
the other two functions, since the huckel function calls them:
from huckelmot.py import makemolskeleton
from huckelmot.py import adjacencymatrix  

If a user wishes to get approximate energies, 
then they need to use alpha and beta (Huckel parameters) values of their choice based on the system
they are exploring. And perform the calculation: alpha - (eigenvalues * beta) (as given by Huckel MO theory) for each 
occupied state
However, one can get the same qualitative understanding from the eigenvalues alone.

CAUTION WITH REGARDS TO GRAPH
A word of caution: The connectivity graph we obtain is NOT a molecular structure. It does not know anything about the
physics of molecules and how bonds are formed, and how regions of electron density repel, etc. 
It is only a guide to the eye (satisfying at times, when it gets it right)
of the carbon skeleton we should be getting. At times, the graph will not conform to molecular structure 
(i.e vertices may cross) and it does not have to, but will always give the correct adjacency relation.

First atom is labeled zero. For instance in benzene 
one obtains carbon numbers 0,1,2,3,4,5 which is six in total.