# Configuration générale
N_NOEUDS_CIBLE = 1000
N_LIENS_CIBLE = 3500
N_GROUPS = 100

# Spécifiques à SBM
GROUP_SIZE = N_NOEUDS_CIBLE // N_GROUPS
FREQ_LIEN_EXTERNE = 0.2

#Pour phase de création du graphe mixte : 
RATIO_SBM = 0.4 #ratio de liens issus de l'approche SBM
