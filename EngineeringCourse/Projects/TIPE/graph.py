import networkx as nx
import matplotlib.pyplot as plt
import sqlite3 as sql
import winsound
import time
import threading
import keyboard
import pickle
import heapq

class Graphe:
    def __init__(self):
        self.graph = nx.Graph()

    def ajouter_sommet(self, sommet):
        """Ajoute un sommet au graphe."""
        self.graph.add_node(sommet)

    def ajouter_arrete(self, sommet1, sommet2, poids=None):
        """Ajoute une arête entre deux sommets. Un poids peut être ajouté."""
        self.graph.add_edge(sommet1, sommet2, weight=poids)

    def existe_arrete(self, sommet1, sommet2):
        """Vérifie si une arête existe entre deux sommets."""
        return self.graph.has_edge(sommet1, sommet2)

    def poids_arrete(self, sommet1, sommet2):
        """Recupère le poids d'une arrete"""
        return self.graph.get_edge_data(sommet1, sommet2)['weight']
    
    def incrementer_poids_arrete(self, sommet1, sommet2, pas = 1):
        if self.graph.has_edge(sommet1, sommet2):
            self.graph[sommet1][sommet2]['weight'] = self.graph[sommet1][sommet2]['weight'] + pas
    
    def afficher_graphe(self):
        """Affiche le graphe en utilisant matplotlib."""
        pos = nx.spring_layout(self.graph)  # Positionnement automatique des sommets
        weights = nx.get_edge_attributes(self.graph, 'weight')

        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold')
        if weights:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=weights)

        plt.title("Représentation du Graphe")
        plt.show()

def beep():
    """ Émet un bip en boucle jusqu'à ce que l'utilisateur appuie sur une touche. """
    while not keyboard.is_pressed("enter"):
        winsound.Beep(1000, 500)
        time.sleep(0.5) 

db = str

def make_Vgs_grph(d:db, test=False, show = True)->Graphe:
    cnect = sql.connect(d)
    cur = cnect.cursor()
    if test:
        txtest = "test"
    else:
        txtest = ""
    
    
    cur.execute("Select max(id) from ingred{}".format(txtest))
    n_ingred = cur.fetchall()[0][0]
    l_ingred = [i for i in range(1, n_ingred+1)]
    g = Graphe()
    for i in l_ingred:
        g.ajouter_sommet(i)
    
    print("Sommet ok")
    cur.execute("select * from jointab{}".format(txtest))
    print("Select all ok")
    data = cur.fetchall()
    n_data = len(data)
    ci = 0
    start = time.time()
    acttime = start
    print("z'est parti")
    print("n data:", n_data)
    for i in range(n_data):
        if (ci == 100):
            ci = 0
            print(i, time.time()-acttime)
            acttime = time.time()
        ci+=1
        for j in range(i+1, n_data):
            r1, i1 = data[i]
            r2, i2 = data[j]
            if r1 == r2:
                if g.existe_arrete(i1, i2):
                    g.incrementer_poids_arrete(i1, i2)
                else:
                    g.ajouter_arrete(i1, i2, 1)
    print("arrete ok")
    print("Final time:", time.time()-start)
    bip_thread = threading.Thread(target=beep, daemon=True)
    bip_thread.start()
    keyboard.wait("enter")
    if (show):
        print("graphe:")
        g.afficher_graphe()
    with open("graphe{}.bin".format(txtest), "wb") as f:
        pickle.dump(g, f)
    print("ok")
    cnect.close()

# Charge le graphe depuis le fichier .bin
def charger_graphe(filepath: str):
    with open(filepath, "rb") as f:
        graphe = pickle.load(f)
    return graphe

# Calcule le PageRank pondéré
def calculer_pagerank(graphe):
    return nx.pagerank(graphe.graph, weight='weight')

# Charge la table ingred pour avoir les noms
def get_ingredient_names(db_path):
    conn = sql.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM ingred")
    id_to_name = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return id_to_name

# Affiche du classement PageRank des ingrédients
def afficher_pagerank_top(graphe, db_path, top_n=20):
    pagerank_scores = calculer_pagerank(graphe)
    id_to_name = get_ingredient_names(db_path)

    sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {top_n} ingrédients selon PageRank :\n")
    for ing_id, score in sorted_pagerank[:top_n]:
        name = id_to_name.get(ing_id, "UNKNOWN")
        print(f"{name:<20} : {score:.6f}")

class KMSTMaxTopM:
    def __init__(self, graph: nx.Graph, k: int, m: int):
        self.graph = graph
        self.k = k
        self.m = m
        self.top_trees = []  
        self.tree_set = set()
        self.call_count = 0
        self.last_log = time.time()

    def add_solution(self, weight, tree):
        tree_frozen = frozenset(tree)
        if tree_frozen in self.tree_set:
            return
        self.tree_set.add(tree_frozen)
        heapq.heappush(self.top_trees, (weight, tree_frozen))
        if len(self.top_trees) > self.m:
            _, removed = heapq.heappop(self.top_trees)
            self.tree_set.remove(removed)

    def dfs(self, current, visited, edge_sum, depth):
        self.call_count += 1
        if self.call_count % 100000 == 0:
            now = time.time()
            best = max(self.top_trees, default=(0, None))[0]
            print(f"[{self.call_count} appels] Niveau {depth} | Sommet: {current} | Poids: {edge_sum} | BestMax: {best} | Δt: {now - self.last_log:.2f}s")
            self.last_log = now

        if len(visited) == self.k:
            self.add_solution(edge_sum, visited.copy())
            return

        for neighbor, data in sorted(self.graph[current].items(), key=lambda x: -x[1]['weight']):
            if neighbor in visited:
                continue
            max_edge = data['weight']
            projected = edge_sum + max_edge + (self.k - len(visited) - 1) * max_edge
            if self.top_trees and projected <= self.top_trees[0][0]:
                continue
            visited.add(neighbor)
            self.dfs(neighbor, visited, edge_sum + data['weight'], depth + 1)
            visited.remove(neighbor)

    def run(self):
        for node in self.graph.nodes:
            self.dfs(node, {node}, 0, 1)
        return sorted(self.top_trees, reverse=True)

class EdgeRMWCSTopM:
    def __init__(self, graph: nx.Graph, k: int, m: int, required_nodes: set):
        self.graph = graph
        self.k = k
        self.m = m
        self.required_nodes = required_nodes
        self.top_solutions = []
        self.solution_set = set()
        self.call_count = 0
        self.last_log = time.time()

    def add_solution(self, weight, nodes):
        frozen = frozenset(nodes)
        if frozen in self.solution_set:
            return
        self.solution_set.add(frozen)
        heapq.heappush(self.top_solutions, (weight, frozen))
        if len(self.top_solutions) > self.m:
            _, removed = heapq.heappop(self.top_solutions)
            self.solution_set.remove(removed)

    def dfs(self, current, visited, weight_sum, depth):
        self.call_count += 1
        if self.call_count % 100_000 == 0:
            now = time.time()
            best = max(self.top_solutions, default=(0, None))[0]
            print(f"[{self.call_count} calls] Depth: {depth} | Node: {current} | Weight: {weight_sum:.2f} | BestMax: {best:.2f} | Δt: {now - self.last_log:.2f}s")
            self.last_log = now

        if len(visited) == self.k:
            if self.required_nodes.issubset(visited):
                self.add_solution(weight_sum, visited.copy())
            return

        for neighbor in sorted(self.graph.neighbors(current), key=lambda n: -self.graph[current][n]['weight']):
            if neighbor in visited:
                continue

            edge_weight = self.graph[current][neighbor]['weight']
            projected_weight = weight_sum + edge_weight + (self.k - len(visited) - 1) * edge_weight

            if self.top_solutions and projected_weight <= self.top_solutions[0][0]:
                continue

            visited.add(neighbor)
            self.dfs(neighbor, visited, weight_sum + edge_weight, depth + 1)
            visited.remove(neighbor)

    def run(self):
        start_nodes = self.required_nodes if self.required_nodes else self.graph.nodes
        for start in start_nodes:
            self.dfs(start, {start}, 0, 1)
        return sorted(self.top_solutions, reverse=True)


def map_ingred (f, l):
    nl = []
    for el in l:
        nl.append(f(el))
    return nl





actions = ["fetchone", "autre ingred", "next", "quitter"]
"""

if __name__ == "__main__":
    actions = ["k-mst", "pagerank"]
    with open("graphe.bin", "rb") as f:
        g = pickle.load(f)
    print("Actions: ", actions)
    choice = input(">")
    db_path = "C:/Users/xavie/Documents/Etudes/TIPE/Salad/main.db"
    if (choice=="k-mst"):
        k = 10
        m = 15
        algo = KMSTMaxTopM(g.graph, k, m)
        print("Début de la recherche...")
        start = time.time()
        top_results = algo.run()
        end = time.time()
        print(f"\nTop {m} arbres de poids maximum pour k = {k} :")
        for i, (poids, sommets) in enumerate(top_results, 1):
            print(f"  #{i} - Poids: {poids}, Sommets: {map_ingred(base.get_ingred,sorted(sommets))}")
        print(f"\nTemps total: {end - start:.2f} sec")
    elif (choice=="pagerank"):
        afficher_pagerank_top(g, db_path)
    elif(choice=="redgemwcs"):
        assert nx.has_path(g.graph, 1, 2), "Sommets obligatoires non connectés"
        
        k = 15
        m = 10
        algo = EdgeRMWCSTopM(g.graph, k, m, set([5405, 14]))
        print("Début de la recherche...")
        start = time.time()
        top_results = algo.run()
        end = time.time()
        print(f"\nTop {m} sous graphes de poids maximum pour k = {k} :")
        for i, (poids, sommets) in enumerate(top_results, 1):
            print(f"  #{i} - Poids: {poids}, Sommets: {map_ingred(base.get_ingred,sorted(sommets))}")
        print(f"\nTemps total: {end - start:.2f} sec")

"""
file_created = True

if __name__ == "__main__":
    db_path = "C:/Users/xavie/Documents/Etudes/TIPE/Salad/main.db"
    if (file_created):
        graphe_path = "graphetest.bin"  # ou "graphetest.bin"
        g = charger_graphe(graphe_path)
        g.afficher_graphe()
        #afficher_pagerank_top(g, db_path, top_n=50)
    else:
        make_Vgs_grph(db_path, test= True, show=False)
