"""This file contains functions to parse a string of
3dim coordinates of atoms, make adjacency matrix akin to carbon backbone of
molecules and diagonalize said matrix for application
in Huckel MO theory"""
import sys

import numpy as np
import networkx as nx
from pylab import show, title, plot, ylabel, xlabel


def makemolskeleton(molecule):
    """This function will make the arrays necessary
    to construct the adjacency matrix and perform Huckel calc later
    it takes
    :param: the entire molecular specifications i.e. spin multiplicity,
    charge and cartesian coordinates of atoms
    it returns an array of coordinates composed only of the carbon skeleton"""

    # Reading in molecular information and organizing
    coordinate_str = molecule.split(
        "\n"
    )  # makes a list of strings from the input molecular specification
    # print(change)
    coordinate_str.pop(0)
    coordinate_str.pop(
        0
    )  # two pop statements to remove empty value and to remove charge and multiplicity entries

    # Here, we go through the list of string coordinate and regrouped the strings so
    # that they are only separated by single space
    my_coord = []
    for items in coordinate_str:
        coords = " ".join(items.split()) # statement which makes each string separated by singspace
        my_coord.append(coords)

    # This line is to remove the strings with the Hydrogen coordinates: it makes a list
    coord_list_nohydrogen = [line for line in my_coord if not line.startswith("H")]
    coord_list_nohydrogen.pop(
        -1
    )  # Here we remove an empty value that is created at the end of list
    # from the previous procedure

    # We now make a zero-filled array to contain the coordinates of the carbon atoms,
    # we will use it to calculate distance and define edges
    array1 = np.zeros(
        (len(coord_list_nohydrogen), 3)
    )  # making it an n by 3 array where n is the number of carbon atoms in the molecule


    # Here we are stripping the letter C and then converting the remaining number strings
    # into floats to be put in array1
    count = (
        0  # initialize this value to keep track of the rows we are filling on each loop
    )
    for z in coord_list_nohydrogen:
        nocarbon = z.strip(
            "C"
        )  # remove the C string from all the strings in the coord_list_nohydrogen list
        lists = nocarbon.split(
            " "
        )  # make a list out the new string, where each value will be separated everytime we find
        # a space in each string
        lists.pop(0)  # remove the inital empty space in the list

        for k in range(3):
            array1[count, k] = lists[k]  # this line fills in the column values

        count += 1

    # Calculating distances between carbon atoms and defining edges
    edges = []
    for i,l in enumerate(array1):
        for k,b in enumerate(array1):
            bond = np.linalg.norm(l - b)
            if 0 < bond < 1.57:
                edges.append([i, k])

    # Making the vertices
    vertices = []
    for i,k in enumerate(array1):
        vertices.append(i + 1)

    return vertices, edges


def adjacencymatrix(vertices, edges):
    """This function creates an adjacency matrix to be used for Huckel calculation
    and also makes a graph using networkx
    :param: vertices which the list of vertices in the molecule as obtain from function
    makemolskeleton
    :param: edges which are lists of lists of atoms connected
    to one another: obtain from the makemolskeleton function
    """
    number_vertices = len(vertices)
    adjm = []  # create an empty list, which will become a list of lists
    while len(adjm) < number_vertices:
        temp = (
            []
        )  # makes an empty list which will be used to turn adjm into a
        # list of list of the appropriate size
        for i in range(
            number_vertices
        ):  # add nzeros to the temp list and add temp list to adjm,
            # redo until adjm is as long as n vertices
            temp.append(0)
        adjm.append(temp)

    for (
        edg
    ) in (
        edges
    ):  # creating the matrix entries in adjm as taken from the edge relationship
        i = edg[0]
        j = edg[1]
        adjm[i][j] = 1
        adjm[j][i] = 1

    # Making the network graph
    _g = nx.Graph()
    _g.add_edges_from(edges)
    nx.draw_networkx(_g, node_color="gray", edge_color="blue")
    # pos = nx.spectral_layout(G, dim =3)
    show()
    # since the above procedure makes a list of lists, we turn in into an array using np.asarray
    adjm = np.asarray(adjm)
    return adjm


def huckel(coordinates):
    """This function will perform a Huckel calculation on conjugated molecule once
    the 3D coordinates of each atom are provided
    it will make function calls to makemolskeleton(molecule) and also adjacency(vertex,edge)
    :param: coordinates 3D coordinates of atoms as obtained and
    "cleaned up" from a 3D molecular drawing software such as gview or Avogadro
    """

    myfile= open("huckel_results.txt", "w",encoding="utf-8")
    # file to save results later

    # Calling makemolskeleton(molecule)
    vertices, edges = makemolskeleton(coordinates)

    # calling function to make adjacency matrix
    ajdacency_matrix = adjacencymatrix(vertices, edges)

    # Calculating eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(ajdacency_matrix)

    # sorting eigenvalues in order of ascending energies
    sorted_eigenvals = np.sort(eigenvals)

    # counting instances of degeneracies
    degen_states = []
    for i,l in enumerate(sorted_eigenvals):
        for k,j in enumerate(sorted_eigenvals):
            if i == k:
                continue
            if abs(l - j) <= 10 ** (
                -12
            ):  # defining a treshold to determine that values are equal to one another
                degen_states.append(i)

    # making plot
    xvals = [
        i for i,k in enumerate(vertices)
    ]  # the xvals are simply the number of p orbital basis functions
    plot(xvals, sorted_eigenvals, "b.")
    xlabel("State number")
    ylabel("Eigenvalues")
    title("Eigenvalues as a function of state")
    show()

    np.set_printoptions(
        threshold=sys.maxsize
    )  # to prevent truncation of eigenvals and eigenvecs in the file

    # writing data to file
    print(
        "The eigenvalues in ascending order are: \n ",
        sorted_eigenvals,
        "\n",
        file=myfile,
    )
    print(
        "There are ", len(np.unique(degen_states)), " degenerate states\n", file=myfile
    )
    print("The degenerate states are ", np.unique(degen_states), "\n", file=myfile)
    print(
        "You can also calculate the pi energy using alpha and beta values of your choosing, "
        "using:\n ALPHA + eigenval "
        "* BETA",
        file=myfile,
    )
    print("\n The eigenvalues as obtained are:\n", eigenvals, sep=",", file=myfile)
    print("\n The eigenvectors as obtained are \n", eigenvecs, sep=",", file=myfile)
    myfile.close()


if __name__ == "__main__":

    BENZENE = """
     0 1
     C                  1.23987086    1.05585606    0.00007236
     C                  2.64127079    1.05578594    0.00050137
     C                  3.34203172    2.26939863   -0.00006381
     C                  2.64139271    3.48308144   -0.00105877
     C                  1.23999278    3.48315156   -0.00148574
     C                  0.53923185    2.26953886   -0.00092082
     H                  0.70482434    0.12923583    0.00050287
     H                  3.17622422    0.12911218    0.00125995
     H                  4.41203167    2.26934510    0.00026331
     H                  3.17643923    4.40970166   -0.00149182
     H                  0.70503935    4.40982531   -0.00224291
     H                 -0.53076809    2.26959240   -0.00124699
     """

    ETHENE = """
     0 1
     C                  1.47939142    1.01642035    0.00000000
     C                  2.83159535    1.10648465    0.00000000
     H                  0.88399079    1.90546370    0.00000000
     H                  1.00715760    0.05626660    0.00000000
     H                  3.30382916    2.06663840    0.00000000
     H                  3.42699598    0.21744129    0.00000000
     """

    ALLYL = """
     0 2
     C                  1.24673495    0.95784805    0.00000000
     C                  2.49943537    1.58607063    0.00000000
     C                  3.66984229    0.81531153    0.00000000
     H                  2.56226828    2.65422419    0.00000000
     H                  0.35310328    1.54633972    0.00000000
     H                  1.18390204   -0.11030551    0.00000000
     H                  4.62630687    1.29497341    0.00000000
     H                  3.60700938   -0.25284203    0.00000000
    """
    BUCKY = """
    0 1
     C                  0.67419752    0.35814852    0.09282822
     C                  2.21419194    0.35820181    0.09273596
     C                  2.89177497    1.53184413    0.09268063
     C                  2.12174170    2.86550842    0.09270462
     C                  0.76653362    2.86546518    0.09277105
     C                 -0.00343152    1.53177002    0.09283202
     C                  0.19831991   -0.80574593   -0.79626917
     C                  1.44421401   -1.52503078   -1.34581830
     C                  2.69007763   -0.80567248   -0.79637479
     C                  3.78642585   -0.65623507   -1.57884098
     C                  4.13764596    1.70166856   -0.79647167
     C                  2.89169346    3.85959203   -0.79643017
     C                  2.21403681    4.73434991   -1.57883402
     C                  0.67403431    4.73429901   -1.57878283
     C                 -0.00351962    3.85950863   -0.79633069
     C                 -1.24940476    3.14016438   -1.34579612
     C                 -1.24933261    1.70153990   -0.79626487
     C                 -1.66810564    0.67731349   -1.57866858
     C                 -0.89807804   -0.65634811   -1.57867440
     C                  1.44418907   -2.00861024   -2.61180777
     C                  0.19825272   -1.83884360   -3.50089951
     C                 -0.89812475   -1.20587152   -3.01730062
     C                 -1.66818306   -0.21183979   -3.90639836
     C                 -2.14406524    0.95203905   -3.01728749
     C                 -2.14413282    2.21802796   -3.50086531
     C                 -1.66824348    3.38192349   -2.61177498
     C                 -0.89829172    4.37598877   -3.50092018
     C                  0.19809458    5.00900861   -3.01739277
     C                  4.13759344    3.14029143   -1.34597154
     C                  2.68978043    3.97598423   -5.72210730
     C                  3.78617937    3.82658601   -4.93970888
     C                  4.55622069    2.49292123   -4.93970441
     C                  4.13744948    1.46869681   -5.72210815
     C                  2.89152351    1.63845728   -6.61119682
     C                  0.67390892    2.81203555   -6.61112772
     C                  0.19802217    3.97591453   -5.72201223
     C                  1.44391362    4.69526459   -5.17256603
     C                  1.44394536    5.17883760   -3.90657398
     C                  2.68985284    5.00908305   -3.01750848
     C                  3.78622630    4.37610380   -3.50109960
     C                  5.03217594    2.21819646   -3.50109660
     C                  5.03224989    0.95220089   -3.01752621
     C                  4.55636974   -0.21168735   -3.90661991
     C                  4.13753317    0.03007650   -5.17260005
     C                  2.89162236   -0.68927963   -5.72205922
     C                  2.12155880    0.30475687   -6.61116248
     C                  0.76635631    0.30471923   -6.61111306
     C                 -0.00366603    1.63837862   -6.61108735
     C                 -0.89832480    3.82646479   -4.93954573
     C                 -1.66830312    2.49277035   -4.93950867
     C                 -1.24951753    1.46855728   -5.72192816
     C                 -1.24946574    0.02993864   -5.17240575
     C                 -0.00358991   -0.68935007   -5.72194782
     C                  0.67406748   -1.56409852   -4.93953834
     C                  2.21407285   -1.56406338   -4.93958861
     C                  2.69002564   -1.83879369   -3.50096194
     C                  3.78640750   -1.20576572   -3.01744323
     C                  4.55630537    3.38205714   -2.61198307
     C                  4.55641798    0.67745480   -1.57888593
     C                  2.21389939    2.81207851   -6.61119368
    """

    PYRENE = """
    0 1
     C                 -1.95643267   -3.86036597   -0.00032586
     C                 -0.55374525   -3.85193872   -0.00033579
     C                  0.13789306   -2.63310578   -0.00033061
     C                 -0.57383622   -1.43013445   -0.00031350
     C                 -1.97158097   -1.43299323   -0.00030536
     C                 -2.66590075   -2.65030064   -0.00031102
     C                  1.54155874   -2.61516422   -0.00034391
     C                  0.11448763   -0.22024686   -0.00030797
     C                  1.51223239   -0.21738808   -0.00032039
     C                  2.21402174   -1.43315571   -0.00033626
     C                  2.20655217    0.99991933   -0.00031495
     H                  3.27653311    1.00630659   -0.00032403
     C                  1.49708411    2.20998465   -0.00030014
     C                  0.09439668    2.20155741   -0.00029067
     C                 -0.59724164    0.98272447   -0.00029479
     C                 -2.00090731    0.96478292   -0.00028667
     C                 -2.67337033   -0.21722560   -0.00028925
     H                 -3.74336507   -0.22057802   -0.00027951
     H                 -2.54474627    1.88627041   -0.00027818
     H                  2.08539769   -3.53665171   -0.00036009
     H                 -2.48553850   -4.79039125   -0.00032998
     H                 -0.01252213   -4.77496501   -0.00034924
     H                 -3.73588169   -2.65668792   -0.00030362
     H                  3.28401649   -1.42980328   -0.00034257
     H                  2.02618995    3.14000993   -0.00029878
     H                 -0.44682643    3.12458371   -0.00028085
     """

    NAPHTALENE = """
     0 1
     C                 -6.94267673    0.98574402    0.00032017
     C                 -5.54130305    0.99433052    0.00012815
     C                 -4.84805231    2.21224897    0.00023105
     C                 -5.55617525    3.42158093    0.00052623
     C                 -6.95754893    3.41299443    0.00072010
     C                 -7.65079967    2.19507598    0.00061867
     H                 -2.90601105    1.29748368   -0.00018803
     H                 -7.47198906    0.05583626    0.00024031
     H                 -5.00063547    0.07097873   -0.00009685
     C                 -3.44667863    2.22083547    0.00003658
     C                 -4.86292450    4.63949938    0.00062502
     H                 -7.49821650    4.33634622    0.00094583
     H                 -8.72077958    2.18852000    0.00076718
     C                 -3.46155082    4.64808588    0.00043049
     C                 -2.75342788    3.43875392    0.00013693
     H                 -5.40359208    5.56285116    0.00085084
     H                 -2.93223849    5.57799364    0.00050793
     H                 -1.68344798    3.44530990   -0.00001060
     """

    BUTADIENE = """
     0 1
     C                 -2.23062487    2.25364056    0.00000000
     C                 -0.90145695    2.51798819    0.00000000
     C                  0.11390184    1.36012527    0.00000000
     C                  1.44306976    1.62447291    0.00000000
     H                 -2.93610143    3.05812973    0.00000000
     H                 -2.57459465    1.24043535    0.00000000
     H                 -0.55748716    3.53119340    0.00000000
     H                 -0.23006795    0.34692007    0.00000000
     H                  1.78703954    2.63767811    0.00000000
     H                  2.14854632    0.81998373    0.00000000
     """
    huckel(PYRENE)