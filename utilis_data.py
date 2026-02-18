import random
import math
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import config  # Import du fichier de constantes

def get_topology_features(G, u, v, precomputed, is_existing_edge=False):
    """Calcule les métriques topologiques pour une paire (u, v) avec optimisation SP."""
    # 1. Métriques de voisinage
    # On utilise try/except pour Adamic-Adar car il crash si aucun voisin commun
    try:
        aa = next(nx.adamic_adar_index(G, [(u, v)]))[2]
    except (ZeroDivisionError, StopIteration):
        aa = 0.0
        
    jc = next(nx.jaccard_coefficient(G, [(u, v)]))[2]
    pa = next(nx.preferential_attachment(G, [(u, v)]))[2]
    cn = len(list(nx.common_neighbors(G, u, v)))

    # 2. Métriques de Nœuds (extraites du pré-calcul)
    node_features = {
        'pr_u': precomputed['pr'].get(u, 0), 'pr_v': precomputed['pr'].get(v, 0),
        'lcc_u': precomputed['lcc'].get(u, 0), 'lcc_v': precomputed['lcc'].get(v, 0),
        'and_u': precomputed['and'].get(u, 0), 'and_v': precomputed['and'].get(v, 0),
        'dc_u': precomputed['dc'].get(u, 0), 'dc_v': precomputed['dc'].get(v, 0)
    }

    # 3. Shortest Path (SP) optimisé avec cutoff
    INF_DIST = 10
    if is_existing_edge:
        G.remove_edge(u, v)
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=INF_DIST)
        sp = lengths.get(v, INF_DIST)
        G.add_edge(u, v)
    else:
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=INF_DIST)
        sp = lengths.get(v, INF_DIST)

    topo_res = {'cn': cn, 'aa': aa, 'jc': jc, 'pa': pa, 'sp': sp}
    topo_res.update(node_features)
    return topo_res

def prepare_balanced_data(G, negative_ratio=1.0):
    """Prépare un DataFrame pour des modèles classiques (RandomForest, etc.)"""
    edges = list(G.edges())
    nodes = list(G.nodes())
    random.seed(42)

    print("📊 Pré-calcul des métriques globales...")
    precomputed = {
        'pr': nx.pagerank(G),
        'lcc': nx.clustering(G),
        'and': nx.average_neighbor_degree(G),
        'dc': nx.degree_centrality(G)
    }
    
    data = []
    
    # --- POSITIFS ---
    for u, v in edges:
        topo = get_topology_features(G, u, v, precomputed, is_existing_edge=True)
        row = {
            'u': u, 'v': v, 
            'dist': math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']),
            'same_block': 1 if G.nodes[u].get('block') == G.nodes[v].get('block') else 0,
            'target': 1
        }
        row.update(topo)
        data.append(row)
    
    # --- NÉGATIFS ---
    n_neg_target = int(len(edges) * negative_ratio)
    neg_count = 0
    while neg_count < n_neg_target:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and u != v:
            topo = get_topology_features(G, u, v, precomputed, is_existing_edge=False)
            row = {
                'u': u, 'v': v,
                'dist': math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']),
                'same_block': 1 if G.nodes[u].get('block') == G.nodes[v].get('block') else 0,
                'target': 0
            }
            row.update(topo)
            data.append(row)
            neg_count += 1
            
    return pd.DataFrame(data)

def build_hybrid_dataset_from_scratch(G, num_negative_samples=None):
    """Génère un objet PyTorch Geometric Data pour GNN."""
    nodes = sorted(list(G.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    print("🧠 Construction du dataset GNN...")
    
    # Features de noeuds
    pr, dc, lcc = nx.pagerank(G), nx.degree_centrality(G), nx.clustering(G)
    and_ = nx.average_neighbor_degree(G)
       
    x_list = []
    for n in nodes:
        feat = [G.nodes[n]['pos'][0], G.nodes[n]['pos'][1], pr[n], dc[n], lcc[n], and_[n]]
        # One-hot dynamique basé sur config
        block_oh = [0.0] * config.N_GROUPS
        block_oh[G.nodes[n]['block']] = 1.0
        x_list.append(feat + block_oh)
    
    x = torch.tensor(x_list, dtype=torch.float)
    edge_index = torch.tensor([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]).t().contiguous()

    # Échantillonnage pour les labels (MLP)
    pos_edges = list(G.edges())
    num_neg = num_negative_samples if num_negative_samples else len(pos_edges)
    
    neg_edges = []
    while len(neg_edges) < num_neg:
        u, v = random.sample(nodes, 2)
        if u != v and not G.has_edge(u, v):
            neg_edges.append((u, v))

    all_pairs = pos_edges + neg_edges
    labels = [1.0] * len(pos_edges) + [0.0] * len(neg_edges)

    # Features d'arrêtes (métriques de paires)
    attr_list = []
    for u, v in all_pairs:
        dist = math.dist(G.nodes[u]['pos'], G.nodes[v]['pos'])
        pa = G.degree(u) * G.degree(v)
        # Jaccard et Adamic simplifié pour la vitesse ici ou appel à get_topology_features
        jc = next(nx.jaccard_coefficient(G, [(u, v)]))[2]
        attr_list.append([dist, pa, jc]) # Ajoute les autres au besoin

    attr_np = np.array(attr_list, dtype=np.float32)
    scaler = StandardScaler()
    edge_attr = torch.from_numpy(scaler.fit_transform(attr_np))

    edge_label_index = torch.tensor([(node_to_idx[u], node_to_idx[v]) for u, v in all_pairs]).t().contiguous()
    edge_label = torch.tensor(labels, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                edge_label_index=edge_label_index, edge_label=edge_label)

def verify_dataset(pyg_data, feature_idx_to_plot=0, feature_name="Feature"):
    print("\n🔍 --- Vérification du Dataset PyG ---")
    
    # 1. Vérification des dimensions
    n_nodes = pyg_data.x.size(0)
    n_node_feat = pyg_data.x.size(1)
    n_pairs = pyg_data.edge_label.size(0)
    n_edge_feat = pyg_data.edge_attr.size(1)
    
    print(f"Nodes: {n_nodes} (Features: {n_node_feat})")
    print(f"Pairs: {n_pairs} (Features: {n_edge_feat})")

    # 2. Équilibre des classes
    labels = pyg_data.edge_label.cpu().numpy()
    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()
    print(f"Balance: {pos_count} (+) / {neg_count} (-)")

    # 3. Vérification de la propreté (NaNs/Infs)
    for i in range(n_edge_feat):
        col = pyg_data.edge_attr[:, i]
        n_nans = torch.isnan(col).sum().item()
        n_infs = torch.isinf(col).sum().item()
        status = "✅" if n_nans == 0 and n_infs == 0 else "❌"
        if status == "❌":
            print(f"{status} Col {i}: {n_nans} NaNs, {n_infs} Infs")

    # 4. Visualisation sécurisée
    if n_edge_feat > feature_idx_to_plot:
        plt.figure(figsize=(8, 4))
        val = pyg_data.edge_attr[:, feature_idx_to_plot].cpu().numpy()
        
        # On ne plot que si on a des variations (évite les erreurs de KDE)
        if np.var(val) > 1e-5:
            sns.kdeplot(x=val[labels == 1], label="Lien (1)", fill=True)
            sns.kdeplot(x=val[labels == 0], label="Non-Lien (0)", fill=True)
            plt.title(f"Distribution de : {feature_name} (index {feature_idx_to_plot})")
            plt.legend()
            plt.show()
        else:
            print(f"⚠️ Feature {feature_idx_to_plot} est constante, KDE impossible.")

########################################
# FONCTIONS DE DATA CLEAN POUR XGBOOST #
########################################
def get_topology_features(G, u, v, precomputed, is_existing_edge=False):
    """Calcule les métriques topologiques pour une paire (u, v)"""
    # 1. Métriques de paires (Voisinage)
    aa = next(nx.adamic_adar_index(G, [(u, v)]))[2]
    jc = next(nx.jaccard_coefficient(G, [(u, v)]))[2]
    pa = next(nx.preferential_attachment(G, [(u, v)]))[2]
    cn = len(list(nx.common_neighbors(G, u, v)))

    # 2. Métriques de Nœuds (extraites du dictionnaire pré-calculé)
    # On ajoute les versions pour u et pour v
    node_features = {
        'pr_u': precomputed['pr'].get(u, 0), 'pr_v': precomputed['pr'].get(v, 0),
        'lcc_u': precomputed['lcc'].get(u, 0), 'lcc_v': precomputed['lcc'].get(v, 0),
        'and_u': precomputed['and'].get(u, 0), 'and_v': precomputed['and'].get(v, 0),
        'dc_u': precomputed['dc'].get(u, 0), 'dc_v': precomputed['dc'].get(v, 0)
    }

    # 3. Shortest Path (SP)
    if is_existing_edge:
        G.remove_edge(u, v)
        try:
            sp = nx.shortest_path_length(G, source=u, target=v)
        except nx.NetworkXNoPath:
            sp = 0 
        G.add_edge(u, v)
    else:
        try:
            sp = nx.shortest_path_length(G, source=u, target=v)
        except nx.NetworkXNoPath:
            sp = 0

    # Fusion de toutes les métriques
    topo_res = {'cn': cn, 'aa': aa, 'jc': jc, 'pa': pa, 'sp': sp}
    topo_res.update(node_features)
    
    return topo_res

def prepare_balanced_data(G, negative_ratio=1.0):
    edges = list(G.edges())
    nodes = list(G.nodes())
    n_pos = len(edges)
    data = []
    random.seed(42)

    # --- ÉTAPE CRUCIALE : PRÉ-CALCUL ---
    # On calcule les métriques globales une seule fois ici
    print("Pré-calcul des métriques de nœuds...")
    precomputed = {
        'pr': nx.pagerank(G),                    # PageRank (PR)
        'lcc': nx.clustering(G),                # Local Clustering Coefficient (LCC)
        'and': nx.average_neighbor_degree(G),   # Average Neighbor Degree (AND)
        'dc': nx.degree_centrality(G)           # Degree Centrality (DC)
    }
    
    # --- 1. CLASSE POSITIVE ---
    for u, v in edges:
        pos_u, pos_v = G.nodes[u].get('pos'), G.nodes[v].get('pos')
        dist = math.dist(pos_u, pos_v)
        same_block = 1 if G.nodes[u].get('block') == G.nodes[v].get('block') else 0
        
        # Passage du dictionnaire precomputed
        topo = get_topology_features(G, u, v, precomputed, is_existing_edge=True)
        
        row = {
            'u': u, 'v': v, 'dist': dist, 'same_block': same_block,
            'block_u': G.nodes[u].get('block'), 'block_v': G.nodes[v].get('block'),
            'target': 1
        }
        row.update(topo)
        data.append(row)
    
    # --- 2. CLASSE NÉGATIVE ---
    n_neg_target = int(n_pos * negative_ratio)
    neg_count = 0
    while neg_count < n_neg_target:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and u != v:
            pos_u, pos_v = G.nodes[u].get('pos'), G.nodes[v].get('pos')
            dist = math.dist(pos_u, pos_v)
            same_block = 1 if G.nodes[u].get('block') == G.nodes[v].get('block') else 0
            
            topo = get_topology_features(G, u, v, precomputed, is_existing_edge=False)
            
            row = {
                'u': u, 'v': v, 'dist': dist, 'same_block': same_block,
                'block_u': G.nodes[u].get('block'), 'block_v': G.nodes[v].get('block'),
                'target': 0
            }
            row.update(topo)
            data.append(row)
            neg_count += 1
            
    return pd.DataFrame(data)
