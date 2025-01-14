# controle_trafic
Résolution algorithmique d'un problème de contrôle de trafic illicite - Introduction à l'algorithmique (IFT2125).

Utilisation d'un algorithme de retour-en-arrière pour trouver quelles arêtes devrions-nous enlever à un réseau de flot afin que ce dernier ait un flot maximal de 0 et cela tout en minisant le nombre d'arêtes enlevées. Le fichier "ex2.csv" est un exemple d'input.

### Discussion

Expliquons rapidement notre implémentation. On commence avec une lecture du csv donné et on construit une matrice d'adjacence qui lui correspond. Pour calculer le flot maximal, on utilise l'algorithme de Ford-Fulkerson et le code provient de la source : https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/

Ensuite, on appelle une fonction qui construit notre arbre de solutions partielles. Cette fonction a accès au flot maximal minimal, aux arêtes enlevées associées au flot maximal minimal, au nombre d'arêtes qu'il reste à enlever, aux arêtes enlevées associées à la solution partielle présente, à la matrice d'adjacence de la solution partielle présente et finalement à l'endroit à partir duquel on peut modifier la matrice d'adjacence. Avec toutes ces informations, on fait des appels récursifs sur cette fonction jusqu'à avoir visité l'arbre au complet ou jusqu'à obtenir un flot maximal minimal de 0. Après ça, le tour est joué, on a en mémoire le flot maximal minimal et les arêtes enlevées associées. Finalement, on écrit le fichier csv qui contient les arêtes retirées.

Soit n le nombre de sommets en entrée, soit a le nombre d'arêtes et soit k le nombre d'arêtes à enlever, regardons la complexité théorique. L'initialisation de la matrice d'adjacence se fait en $\Theta(n^2)$. L'algorithme de Ford-Fulkerson pour le flot maximal est dans $O(a \cdot$ flotMax) selon notre source. La fonction copieMatrice est dans $\Theta(n^2)$ si la matrice en entrée est n par n. La grosse partie se fait dans la partie du retour-en-arrière. Notons le nombre de noeuds dans l'arbre de solution $N = 1 + a + a(a-1) + a(a-1)(a-2) + ... + a(a-1)(a-2)(a-3)...(a-k+1)$. De plus, chaque noeud fait comme travail un $\Theta(n^2)$ pour copier la matrice et un $\Theta(a \cdot$ flotMax) pour trouver le flot maximal. Sans surprise, c'est ce qui détermine la complexité de l'algorithme : $\Theta(N \cdot n^2 + N \cdot a \cdot$ flotMax).

Regardons maintenant la complexité empirique. Le temps mesuré pour un fichier de 11 sommets, 22 arêtes et k=1 est d'environ 0,002 seconde. Pour le même fichier mais avec k = 3, le temps est d'environ 0,005 seconde. Si on refait la même chose avec k=22, on est à environ 0,004 seconde. Prenons un autre fichier de 18 sommets, 44 arêtes et k=2. On est à environ 0,2 secondes. Avec k=3, on est à environ 2,5 secondes. Mais avec k=4, on est à environ 0,03 secondes parce qu'on arrive à un flot maximal minimal de 0, donc on ne visite pas l'arbre au complet comme dans le cas de k=3.

Pour ce qui est de la profondeur k, il est important de comprendre que notre algorithme cesse d'explorer de nouvelles solutions partielles à partir du moment qu'il trouve un flot maximal minimal de 0. Ainsi, lorsque k dépasse le nombre d'arêtes nécessaires à enlever pour atteindre un flot de 0, l'algorithme arrête avant d'atteindre la profondeur k. Soit i le nombre d'arêtes reliant la source à un autre sommet et soit j le nombre d'arêtes reliant un sommet au puit, dès que k égale le minimum de i et j, le flot maximal devient 0 et l'algorithme arrête parce qu'on peut enlever les i ou les j arêtes directement pour avoir un flot maximal de 0. Comme i et j ne peuvent pas dépasser n-1, la profondeur k atteinte aussi. Par contre, cela n'exclut pas la possibilité qu'un nombre inférieur d'arêtes enlevées puisse amener le flot maximal à 0. Si le flot maximal minimal atteint est supérieur à 0, on visitera les k profondeurs de l'arbre de solution partielles.
