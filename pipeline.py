import argparse

import networkx as nx
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from models.GNNmodels import *

from models.xgboost_model import * 

# Import des fichiers locaux
import config
from utilis_graph import load_graph ,reach_n_liens_cible, save_graph_custom, displayGraphPyViz, displayGraphPositionNetworkX
from utilis_data import build_hybrid_dataset_from_scratch, verify_dataset, prepare_balanced_data


def run_generate_graphs():
    G_SBM = reach_n_liens_cible(param=0.7517, step = 0.001, decay = 0.1, graphStyle = "SBM")

    def apply_visual_attributes(G, palette):
        """Applique les couleurs et styles aux nœuds pour PyVis."""
        for node in G.nodes():
            # Récupère le block (SBM), sinon 0 par défaut
            group_id = G.nodes[node].get('block', 0)
            G.nodes[node]['color'] = palette[group_id % len(palette)]
            G.nodes[node]['title'] = f"Nœud {node} | Groupe {group_id}"
            G.nodes[node]['size'] = 15
            G.nodes[node]['label'] = ""
        return G

    try:
        colormap = cm.get_cmap('tab10') 
    except:
        colormap = plt.get_cmap('tab10')
    palette = [mcolors.to_hex(colormap(i % 10)) for i in range(config.N_GROUPS)]

    print("--- Génération des graphes de base ---")
    
    # 2. Génération graphe SBM
    # On récupère le graphe ET le paramètre final (car reach_n_liens_cible renvoie un tuple)
    G_SBM, p_final_sbm = reach_n_liens_cible(param=0.7517, step=0.001, decay=0.1, graphStyle="SBM")
    G_SBM = apply_visual_attributes(G_SBM, palette)

    # 3. Génération graph Positionnel
    G_position, p_final_pos = reach_n_liens_cible(param=0.04915518, step=0.00000001, decay=0.1, graphStyle="pos")

    # 4. Phase de chimérisation
    print("\n--- Phase de Chimérisation (Approche Hybride) ---")

    ## APPROCHE 1 : on numérote les liens, puis prenons X% des liens du 1er graphe SBM, 100-X% du second Position
    
    # Calcul des quotas de liens
    n_pos_target = int(config.N_LIENS_CIBLE * (1 - config.RATIO_SBM))
    n_sbm_target = int(config.N_LIENS_CIBLE * config.RATIO_SBM)

    edges_pos = list(G_position.edges())
    edges_SBM = list(G_SBM.edges())

    sample_pos = random.sample(edges_pos, min(len(edges_pos), n_pos_target))
    sample_SBM = random.sample(edges_SBM, min(len(edges_SBM), n_sbm_target))

    # Construction du graphe hybride
    G_hybride = nx.Graph()
    G_hybride.add_nodes_from(range(config.N_NOEUDS_CIBLE))
    G_hybride.add_edges_from(sample_pos)
    G_hybride.add_edges_from(sample_SBM)

    # Transfert des métadonnées (positions et blocs)
    nx.set_node_attributes(G_hybride, nx.get_node_attributes(G_position, 'pos'), 'pos')
    nx.set_node_attributes(G_hybride, nx.get_node_attributes(G_SBM, 'block'), 'block')

    # Application du style visuel
    G_hybride = apply_visual_attributes(G_hybride, palette)

    print(f"Liens Pos: {len(sample_pos)} | Liens SBM: {len(sample_SBM)}")
    print(f"Total Hybride: {G_hybride.number_of_edges()} liens")

    print("\n--- Sauvegardes et Visualisation ---")
    
    # Utilisation de save_graph_custom pour avoir des noms de fichiers propres
    save_graph_custom(G_SBM, "SBM", p_final_sbm)
    save_graph_custom(G_position, "Pos", p_final_pos)
    save_graph_custom(G_hybride, "Hybride", config.RATIO_SBM)

    # Visualisations
    displayGraphPositionNetworkX(G_hybride)
    displayGraphPyViz(G_hybride)

def run_prepare_datasets_for_GNN():
        # 1. Chargement des graphes sauvegardés précédemment
        print("📂 Chargement des graphes...")
        try:
            # Si tes fichiers sont dans un sous-dossier, ajuste le chemin (ex: "outputs/graphs/...")
            g_sbm = load_graph("outputs/graphs/G_SBM_1000n_3500e_p0752.json")
            g_pos = load_graph("outputs/graphs/G_Pos_1000n_3507e_p0049.json")
            g_hybride = load_graph("outputs/graphs/G_Hybride_1000n_3494e_p04.json")
            
            print(f"✅ Graphes chargés avec succès.")
            print(f"📊 Liens dans le graphe hybride : {g_hybride.number_of_edges()}")
        except FileNotFoundError as e:
            print(f"❌ Erreur : Fichier introuvable.({e})")
            return

        # 2. Génération du dataset PyTorch Geometric (PyG)
        # C'est ici que le calcul intensif (Shortest Path, Jaccard, etc.) a lieu
        print("\n🛠️ Construction du dataset hybride (GNN)...")
        my_dataset = build_hybrid_dataset_from_scratch(g_hybride)

        # 3. Vérification et Diagnostic
        # On précise quelle feature on veut voir (par ex. l'index 0 pour la distance euclidienne)
        verify_dataset(
            my_dataset, 
            feature_idx_to_plot=0, 
            feature_name="Distance Euclidienne (normalisée)"
        )

        # 4. Sauvegarde optionnelle du dataset
        # Très utile pour ne pas avoir à tout recalculer avant l'entraînement
        dataset_path = "outputs/datasets/dataset_hybride_pyg.pt"
        torch.save(my_dataset, dataset_path)
        print(f"\n💾 Dataset PyG sauvegardé sous : {dataset_path}")

def run_xgboost_pipeline(scenario = "Total (Pos + SBM + Topo)"):
    print("📂 Chargement du graphe pour XGBoost...")
    # On charge le graphe hybride généré à l'étape 'generate'
    try:
        G = load_graph("outputs/graphs/G_Hybride_1000n_3494e_p04.json")
    except FileNotFoundError:
        print("❌ Graphe introuvable. Lancez d'abord 'generate'.")
        return

    # 1. Préparation des données tabulaires
    print("🏗️  Préparation des features (XGBoost)...")
    df_features = prepare_balanced_data(G, negative_ratio=0.1)
    
    
    # 2. Entraînement et évaluation
    print("🚀 Entraînement des différents scénarios...")
    stats_df, model, X_test = train_and_eval_xgboost(df_features, K=50)

    # 3. Affichage et Sauvegarde
    print("\n--- Résultats Comparatifs ---")
    print(stats_df.to_string(index=False))
    
    # Sauvegarde du meilleur modèle
    model.save_model(f"outputs/models/xgboost_{scenario}.json")
    print(f"\n💾 Modèle {scenario} sauvegardé dans outputs/models/")

    print("\n🕵️ Lancement de l'analyse SHAP...")
    # 1. Calcul et Plots SHAP
    shap_vals = analyze_with_shap(model, X_test)
    
    # 2. Rankings
    feature_names = X_test.columns.tolist()
    rank_dist = calculate_feature_rankings(shap_vals, feature_names)
    
    print("✅ Analyses terminées. Plots sauvegardés dans outputs/plots/")
    print("\nTop 5 des rangs SHAP (Aperçu) :")
    print(rank_dist.head(5))


def run_gnn_pipeline():
    """Charge le dataset PyG pré-calculé et entraîne le HybridLinkPredictor."""
    dataset_path = "outputs/datasets/dataset_hybride_pyg.pt"
    
    print(f"📂 Chargement du dataset PyG : {dataset_path}")
    try:
        # On charge le fichier .pt (objet torch_geometric.data.Data)
        my_dataset = torch.load(dataset_path)
        print("✅ Dataset chargé avec succès.")
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {dataset_path} est introuvable. Lance d'abord la préparation.")
        return

    # Diagnostic rapide avant de lancer
    print(f"📊 Nombre de nœuds : {my_dataset.x.size(0)}")
    print(f"📊 Features de nœuds : {my_dataset.x.size(1)}")
    print(f"📊 Features de liens : {my_dataset.edge_attr.size(1)}")
    
    # Lancement de l'entraînement
    # On passe l'objet data directement à la fonction pilote
    model, final_auc = run_gnn_training(my_dataset, epochs=1000, lr=0.005)
    
    print(f"\n🏆 Entraînement GNN terminé.")
    print(f"Final Best AUC : {final_auc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de génération et processing de graphes")
    
    # Création des sous-commandes
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # Sous-commande : genGraphs
    parser_genGraphs = subparsers.add_parser('genGraphs', help='Génère les graphes SBM, Pos et Hybride')
    
    # Sous-commande : genDatasetsGNN
    parser_genDatasetsGNN = subparsers.add_parser('genDatasetsGNN', help='Génère le dataset PyG à partir des fichiers JSON, pour entrainer le GNN')

    # Sous-commande : xgboost
    parser_xgboost = subparsers.add_parser('xgboost', help='Entraine et évalue un modèle xgboost à partir des graphes générés + Analyse SHAP')

    # Sous-commande : trainGNN
    parser_trainGNN = subparsers.add_parser('trainGNN', help='Entraine et évalue un modèle GNN hybride (GCN pour noeuds feat. + MLP pour paires feat.)')


    # Analyse des arguments
    args = parser.parse_args()

    if args.command == 'genGraphs':
        print("🚀 Lancement de la génération des graphes...")
        run_generate_graphs()
        
    elif args.command == 'genDatasetsGNN':
        print("📊 Lancement de la préparation du dataset GNN...")
        run_prepare_datasets_for_GNN()

    elif args.command == 'xgboost':
        print("Lancement de la génération du modèle xgboost ...")
        run_xgboost_pipeline()

    elif args.command == 'trainGNN':
        print("Lancement du training du GNN ...")
        run_gnn_pipeline()
        
    else:
        # Si aucune commande n'est fournie, on affiche l'aide
        parser.print_help()

if __name__ == "__main__":
    main()