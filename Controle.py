# Matricules : 20190448 et 20228889
import csv
import time

t1 = time.time()
# Lecture du fichier et mise en place de la matrice d'adjacence
with open('ex2.csv', 'r') as file:

    reader = csv.reader(file)

    head = next(reader)
    head2 = head[0].split(";")
    nbSommet = int(head2[0]) # Nombre de sommets dans le grpahe
    k = int(head2[1]) # Nombre d'arêtes à enlever

    # Initialisation de la matrice d'adjacence
    matriceAdjacence = [[0 for j in range(nbSommet)] for i in range(nbSommet)]

    i = 0
    for row in reader:
        rowSplit = row[0].split(";")
        j = 0
        for sommet in rowSplit:
            if j == 0: # On évite le premier terme
                j += 1
                continue
            sommet2 = sommet.split("(")
            sommetLien = int(sommet2[0])
            flot = int(sommet2[1][:-1])
            j += 1
            matriceAdjacence[i][sommetLien] = flot # Modification de la matrice d'adjacence
        i += 1

# Classe Graph qui contient la partie qui calcule le flot maximal avec l'algorithme de
# Ford-Fulkerson.
# Source : https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
class Graph:

    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.ROW = len(graph)
        # self.COL = len(gr[0])

    '''Returns true if there is a path from source 's' to sink 't' in
    residual graph. Also fills parent[] to store the path '''

    def BFS(self, s, t, parent):

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            # Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    # If we find a connection to the sink node,
                    # then there is no point in BFS anymore
                    # We just have to set its parent and can return true
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True

        # We didn't reach sink in BFS starting
        # from source, so return false
        return False

    # Returns the maximum flow from s to t in the given graph
    def FordFulkerson(self, source, sink):

        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow

# Fonction copieMatrice qui permet le clonage de matrice. Fonction très utile lors du
# retour-en-arrière.
def copieMatrice(mat):
    copie = []
    for ligne in mat:
        newLigne = []
        for elem in ligne:
            newLigne.append(elem)
        copie.append(newLigne)
    return copie


# Fonction récursive qui contient le gros de notre algorithme de retour en arrière. Le
# premier paramètre est la matrice d'adjacence de notre graphe. Le deuxième est le nombre
# d'arêtes qu'il reste à enlever. Finalement, le troisième et le quatrième sont les indices
# associés à la ligne et à la colonne de la matrice à partir d'où on peut modifier la matrice
# d'adjacence. Ces derniers paramètres sont utiles pour éviter de visiter des parties de
# l'arbre de solutions partielles si ces dernières sont équivalentes à des parties déjà
# visitées.
def areteRecursive(matriceAdja, k, ligne, colonne):
    global minFlot
    global arreteEnleveMin
    global arreteEnleve

    if minFlot == 0: # Dans ce cas, on peut arrêter l'algorithme car on a trouvé une solution
        return       # optimale

    if k == 0: # Dans ce cas, on arrête l'algorithme car il ne reste plus d'arête à enlever
        return

    matriceAdjaCopie = copieMatrice(matriceAdja)
    for i in range(ligne, len(matriceAdjaCopie)): # On commence l'itération à partir de la
                                                  # ligne en paramètre jusqu'à la dernière
                                                  # ligne
        borneInf = 0
        if (i == ligne): # On veut seulement prendre en compte la colonne de départ (en
                         # paramètre) si on est sur la première ligne modifiable de la
                         # matrice
            borneInf = colonne
        for j in range(borneInf, len(matriceAdjaCopie)):
            if (matriceAdja[i][j] != 0): # S'il y a un lien entre le sommet i et le sommet j
                areteTemp = "(" + str(i) + ", " + str(j) + ")" # Arête temporaire
                # On ajoute l'arête aux arêtes enlevées et on l'enlève de la matrice
                arreteEnleve.append(areteTemp)
                matriceAdjaCopie[i][j] = 0
                # Création du nouveau graphe
                copieMat2 = copieMatrice(matriceAdjaCopie)
                g = Graph(copieMat2)
                flotTemp = g.FordFulkerson(source, puit) # On calcule sont flot maximal
                if(flotTemp < minFlot): # Si on trouve une nouveau flot maximal minimal
                    # On le remplace et on remplace la liste des arêtes enlevées minimales
                    minFlot = flotTemp
                    arreteEnleveMin = arreteEnleve.copy()
                # On rapelle la fonction sur ce graphe avec une arête en moins à enlever et
                # en gardant tout élément de matrice avant (i,j) non modifiable
                areteRecursive(matriceAdjaCopie, k - 1, i, j)
                # On s'apprête à aller sur une différente branche de l'arbre donc on retire
                # l'arête de la liste des arêtes enlevées et on remet la matrice dans son
                # était initial (au début de cet appel de fonction)
                arreteEnleve.remove(areteTemp)
                matriceAdjaCopie = copieMatrice(matriceAdja)

# Algorithme de retour-en-arrière

# Initialisation des éléments nécessaires au retour-en-arrière :
# Création du graphe selon le fichier lu
copieMat = copieMatrice(matriceAdjacence)
g = Graph(copieMat)

source = 0;
puit = nbSommet - 1
minFlot = g.FordFulkerson(source, puit) # Initialisation du flot maximal minimal
arreteEnleveMin = [] # Contient les arêtes enlevées pour avoir le flot maximal minimal
arreteEnleve = [] # Contient les arêtes enlevées par rapport à la position de la solution
                  # partielle explorée
areteRecursive(matriceAdjacence, k, 0, 0) # Appel de la fonction de retour-en-arrière

t2 = time.time()
print("%.8f" % (t2 - t1))
print(minFlot)
print(arreteEnleveMin)