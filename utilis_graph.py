import os
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network

# Import des paramètres depuis la config
import config

def reach_n_liens_cible(param=0.3, step=0.05, decay=0.01, graphStyle="SBM"): 
    # On récupère les constantes depuis le fichier config
    n_n = config.N_NOEUDS_CIBLE
    n_l = config.N_LIENS_CIBLE
    n_g = config.N_GROUPS
    g_s = config.GROUP_SIZE
    f_ext = config.FREQ_LIEN_EXTERNE / n_n

    def generate(p):
        if graphStyle == "SBM":
            return nx.planted_partition_graph(n_g, g_s, p, f_ext, seed=42)
        elif graphStyle == "pos":
            return nx.soft_random_geometric_graph(n_n, radius=p, dim=2, seed=42)
        return None

    Graph = generate(param)
    if Graph is None:
        print("ERROR: mauvais GraphStyle")
        return None

    for i in range(1, 301):
        current_edges = Graph.number_of_edges()
        if current_edges == n_l:
            break
        
        # Ajustement du paramètre
        adjustment = step * decay * (1 - i/300)
        if current_edges < n_l:
            param = min(1.0, param + adjustment)
        else:
            param = max(0.0, param - adjustment)
            
        Graph = generate(param)

    if Graph.number_of_edges() == n_l:
        print(f"SUCCES après {i} steps : {graphStyle}, final param = {round(param, 6)}")
    else:
        print(f"ECHEC après {i} steps : {Graph.number_of_edges()} edges au lieu de {n_l}, , final param = {round(param, 12)}")
    
    return Graph, param

class GraphEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, np.ndarray)):
            return list(obj)
        return super().default(obj)

def save_graph_custom(G, style, param, base_path="outputs/graphs"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    n_n = G.number_of_nodes()
    n_e = G.number_of_edges()
    
    # Remplacement du point pour le nom de fichier
    p_str = str(round(param, 3)).replace('.', '')
    filename = f"G_{style}_{n_n}n_{n_e}e_p{p_str}.json"
    full_path = os.path.join(base_path, filename)
    
    data = nx.node_link_data(G)
    with open(full_path, 'w') as f:
        json.dump(data, f, cls=GraphEncoder)
    
    print(f"💾 Graphe sauvegardé : {full_path}")
    return full_path

def load_graph(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    # On reconstruit le graphe à partir du dictionnaire
    return nx.node_link_graph(data)

def displayGraphPyViz(GraphNetworkX) : 
    
    # 4. Config Pyvis
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    net.from_nx(GraphNetworkX)
    
    # 5. Options JSON (Le dictionnaire est maintenant compatible JS)
    options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -150, # Augmenté pour plus de répulsion entre groupes
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08,
          "avoidOverlap": 1
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": True,
          "iterations": 1000,
          "updateInterval": 25
        }
      },
      "edges": {
        "smooth": {"type": "continuous"},
        "color": {"inherit": "both"},
        "opacity": 0.2 # Baissé pour voir les clusters à travers les liens
      },
      "interaction": {
        "hover": True,
        "navigationButtons": True # Ajoute des flèches de zoom/déplacement
      }
    }
    
    net.set_options(json.dumps(options))
    
    # Export
    net.write_html("mon_graphe_interactif.html")
    print(f"Fichier généré pour {config.N_GROUPS} groupes.") 


def displayGraphPositionNetworkX(graph) : 

    pos = nx.get_node_attributes(graph, 'pos')
    
    # 3. Visualisation
    plt.figure(figsize=(8, 8))
    
    # On dessine les liens (transparents pour voir la structure)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color="steelblue")
    
    # On dessine les nœuds
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color="darkorange")
    
    # Optionnel : Dessiner un cercle pour illustrer le rayon de connexion (radius)
    # Cela aide à comprendre pourquoi certains nœuds sont liés et d'autres non.
    
    plt.title(f"Random Geometric Graph")
    plt.xlim(-0.05, 1.05) # Les positions sont entre 0 et 1
    plt.ylim(-0.05, 1.05)
    plt.gca().set_aspect('equal', adjustable='box') # Garde le ratio 1:1 (carré)
    plt.axis('on') # On garde les axes pour voir les coordonnées [0,1]
    plt.grid(alpha=0.2)
    plt.show()