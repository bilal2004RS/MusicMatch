# MusicMatch - Code de Training du Modèle de Clustering
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration de style pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("🎵 MUSICMATCH - TRAINING DU MODÈLE DE CLUSTERING")
print("=" * 60)

# =========================================================
# 1. GÉNÉRATION DES DONNÉES SIMULÉES
# =========================================================

print("\n📊 Étape 1: Génération des données simulées")
print("-" * 60)

np.random.seed(42)

# Listes d'artistes par genre
artists_by_genre = {
    'Pop': ['Taylor Swift', 'Ed Sheeran', 'Ariana Grande', 'Justin Bieber', 'Dua Lipa', 'The Weeknd'],
    'Rock': ['Queen', 'Led Zeppelin', 'Metallica', 'AC/DC', 'Guns N Roses', 'Nirvana'],
    'Hip-Hop': ['Drake', 'Kendrick Lamar', 'Eminem', 'Travis Scott', 'J. Cole', 'Kanye West'],
    'Electronic': ['Daft Punk', 'Calvin Harris', 'Avicii', 'David Guetta', 'Deadmau5', 'Tiesto'],
    'Jazz': ['Miles Davis', 'John Coltrane', 'Ella Fitzgerald', 'Louis Armstrong', 'Billie Holiday'],
    'Classical': ['Mozart', 'Beethoven', 'Bach', 'Chopin', 'Vivaldi', 'Tchaikovsky']
}

# Générer 500 utilisateurs avec des préférences musicales
data = []
user_id = 1

for _ in range(500):
    # Choisir un genre dominant pour chaque utilisateur
    dominant_genre = np.random.choice(list(artists_by_genre.keys()))
    
    # 70% des artistes viennent du genre dominant
    num_artists = np.random.randint(3, 15)
    num_dominant = int(num_artists * 0.7)
    num_other = num_artists - num_dominant
    
    # Sélectionner les artistes
    selected_artists = np.random.choice(artists_by_genre[dominant_genre], 
                                       min(num_dominant, len(artists_by_genre[dominant_genre])), 
                                       replace=False).tolist()
    
    # Ajouter quelques artistes d'autres genres
    other_genres = [g for g in artists_by_genre.keys() if g != dominant_genre]
    for _ in range(num_other):
        genre = np.random.choice(other_genres)
        artist = np.random.choice(artists_by_genre[genre])
        selected_artists.append(artist)
    
    # Créer les entrées pour chaque artiste
    for artist in selected_artists:
        # Trouver le genre de l'artiste
        artist_genre = None
        for genre, artists in artists_by_genre.items():
            if artist in artists:
                artist_genre = genre
                break
        
        data.append({
            'UserID': f'user_{user_id}',
            'Artist': artist,
            'Genre': artist_genre,
            'PlayCount': np.random.randint(1, 100),
            'Country': np.random.choice(['USA', 'UK', 'France', 'Germany', 'Canada'])
        })
    
    user_id += 1

df = pd.DataFrame(data)

print(f"✓ Dataset généré: {len(df)} entrées pour {df['UserID'].nunique()} utilisateurs")
print(f"✓ Nombre d'artistes uniques: {df['Artist'].nunique()}")
print(f"✓ Genres: {df['Genre'].unique()}")

# Afficher les premières lignes
print("\n📋 Aperçu des données:")
print(df.head(10))

print("\n📈 Statistiques descriptives:")
print(df.describe())

# =========================================================
# 2. EXPLORATION ET VISUALISATION
# =========================================================

print("\n\n📊 Étape 2: Exploration et Visualisation des Données")
print("-" * 60)

# Vérifier les valeurs manquantes
print(f"Valeurs manquantes:\n{df.isnull().sum()}")

# Distribution des genres
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Genre distribution
genre_counts = df['Genre'].value_counts()
axes[0, 0].bar(genre_counts.index, genre_counts.values, color='skyblue')
axes[0, 0].set_title('Distribution des Genres Musicaux', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Genre')
axes[0, 0].set_ylabel('Nombre d\'écoutes')
axes[0, 0].tick_params(axis='x', rotation=45)

# Top 10 artistes
top_artists = df['Artist'].value_counts().head(10)
axes[0, 1].barh(top_artists.index, top_artists.values, color='lightcoral')
axes[0, 1].set_title('Top 10 Artistes les Plus Écoutés', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Nombre d\'utilisateurs')
axes[0, 1].invert_yaxis()

# Distribution des PlayCount
axes[1, 0].hist(df['PlayCount'], bins=30, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Distribution des Compteurs d\'Écoute', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Nombre d\'écoutes')
axes[1, 0].set_ylabel('Fréquence')

# Nombre d'artistes par utilisateur
artists_per_user = df.groupby('UserID')['Artist'].count()
axes[1, 1].hist(artists_per_user, bins=20, color='plum', edgecolor='black')
axes[1, 1].set_title('Distribution du Nombre d\'Artistes par Utilisateur', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Nombre d\'artistes')
axes[1, 1].set_ylabel('Nombre d\'utilisateurs')

plt.tight_layout()
plt.savefig('musicmatch_eda.png', dpi=300, bbox_inches='tight')
print("✓ Graphiques d'exploration sauvegardés: musicmatch_eda.png")

# =========================================================
# 3. PRÉTRAITEMENT DES DONNÉES
# =========================================================

print("\n\n🔧 Étape 3: Prétraitement des Données")
print("-" * 60)

# Suppression des doublons
df_clean = df.drop_duplicates(subset=['UserID', 'Artist'])
print(f"✓ Doublons supprimés: {len(df) - len(df_clean)} entrées")

# Création de la matrice utilisateur-artiste (User-Item Matrix)
user_artist_matrix = df_clean.pivot_table(
    index='UserID',
    columns='Artist',
    values='PlayCount',
    fill_value=0
)

print(f"✓ Matrice utilisateur-artiste créée: {user_artist_matrix.shape}")

# Feature Engineering: Ajouter des caractéristiques dérivées
user_features = pd.DataFrame()
user_features['UserID'] = user_artist_matrix.index

# Nombre d'artistes écoutés
user_features['num_artists'] = (user_artist_matrix > 0).sum(axis=1).values

# Total d'écoutes
user_features['total_plays'] = user_artist_matrix.sum(axis=1).values

# Moyenne d'écoutes par artiste
user_features['avg_plays'] = user_features['total_plays'] / user_features['num_artists']

# Diversité des genres (entropie)
genre_by_user = df_clean.groupby('UserID')['Genre'].apply(list).to_dict()
genre_diversity = []

for user_id in user_features['UserID']:
    genres = genre_by_user.get(user_id, [])
    genre_counts = pd.Series(genres).value_counts(normalize=True)
    entropy = -sum(genre_counts * np.log2(genre_counts + 1e-10))
    genre_diversity.append(entropy)

user_features['genre_diversity'] = genre_diversity

# Encoder les genres principaux
dominant_genre = df_clean.groupby('UserID')['Genre'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Pop')
user_features['dominant_genre'] = dominant_genre.values

print("\n📊 Features créées:")
print(user_features.head())

# Normalisation des features numériques
scaler = StandardScaler()
features_to_scale = ['num_artists', 'total_plays', 'avg_plays', 'genre_diversity']
user_features_scaled = user_features.copy()
user_features_scaled[features_to_scale] = scaler.fit_transform(user_features[features_to_scale])

# Combiner avec la matrice utilisateur-artiste
X = user_artist_matrix.values

# Normalisation de la matrice
X_scaled = StandardScaler().fit_transform(X)

print(f"✓ Données normalisées: {X_scaled.shape}")

# =========================================================
# 4. DÉTERMINATION DU NOMBRE OPTIMAL DE CLUSTERS
# =========================================================

print("\n\n🔍 Étape 4: Détermination du Nombre Optimal de Clusters")
print("-" * 60)

# Méthode du coude (Elbow Method)
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"K={k}: Inertie={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Nombre de Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertie', fontsize=12)
axes[0].set_title('Méthode du Coude (Elbow Method)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette scores
axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Nombre de Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Scores de Silhouette', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('musicmatch_optimal_k.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphiques sauvegardés: musicmatch_optimal_k.png")

# Choisir K optimal (basé sur le silhouette score)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n🎯 Nombre optimal de clusters: K = {optimal_k}")

# =========================================================
# 5. ENTRAÎNEMENT DU MODÈLE K-MEANS
# =========================================================

print("\n\n🤖 Étape 5: Entraînement du Modèle K-Means")
print("-" * 60)

# Entraîner le modèle final
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
clusters = kmeans_final.fit_predict(X_scaled)

print(f"✓ Modèle K-Means entraîné avec {optimal_k} clusters")
print(f"✓ Nombre d'itérations: {kmeans_final.n_iter_}")

# Ajouter les clusters aux données
user_features['Cluster'] = clusters

# Distribution des clusters
print("\n📊 Distribution des clusters:")
cluster_counts = pd.Series(clusters).value_counts().sort_index()
for cluster, count in cluster_counts.items():
    percentage = (count / len(clusters)) * 100
    print(f"  Cluster {cluster}: {count} utilisateurs ({percentage:.1f}%)")

# =========================================================
# 6. ÉVALUATION DU MODÈLE
# =========================================================

print("\n\n📈 Étape 6: Évaluation du Modèle")
print("-" * 60)

# Calcul des métriques d'évaluation
silhouette = silhouette_score(X_scaled, clusters)
davies_bouldin = davies_bouldin_score(X_scaled, clusters)
calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)

print(f"Silhouette Score: {silhouette:.4f} (plus proche de 1 = meilleur)")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (plus bas = meilleur)")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.2f} (plus haut = meilleur)")

# =========================================================
# 7. VISUALISATION DES CLUSTERS (PCA)
# =========================================================

print("\n\n🎨 Étape 7: Visualisation des Clusters")
print("-" * 60)

# Réduction de dimension avec PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"✓ PCA effectuée: variance expliquée = {sum(pca.explained_variance_ratio_):.2%}")

# Visualisation
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                     s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Cluster')

# Ajouter les centres des clusters
centers_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=300, 
           alpha=0.8, edgecolors='black', linewidth=2, marker='X', label='Centres')

plt.xlabel(f'Composante Principale 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
plt.ylabel(f'Composante Principale 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
plt.title('Visualisation des Clusters (PCA)', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('musicmatch_clusters_pca.png', dpi=300, bbox_inches='tight')
print("✓ Visualisation sauvegardée: musicmatch_clusters_pca.png")

# =========================================================
# 8. ANALYSE DES CLUSTERS
# =========================================================

print("\n\n🔬 Étape 8: Analyse des Caractéristiques des Clusters")
print("-" * 60)

# Analyser le genre dominant par cluster
cluster_analysis = user_features.groupby('Cluster').agg({
    'num_artists': 'mean',
    'total_plays': 'mean',
    'avg_plays': 'mean',
    'genre_diversity': 'mean',
    'dominant_genre': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
})

print("\n📊 Profil moyen par cluster:")
print(cluster_analysis.round(2))

# Nommer les clusters selon leur genre dominant
cluster_names = {}
for cluster in range(optimal_k):
    dominant = cluster_analysis.loc[cluster, 'dominant_genre']
    cluster_names[cluster] = f"Cluster {cluster}: {dominant} Fans"

print("\n🏷️ Noms des clusters:")
for cluster, name in cluster_names.items():
    print(f"  {name}")

# =========================================================
# 9. COMPARAISON AVEC DBSCAN (OPTIONNEL)
# =========================================================

print("\n\n🔄 Étape 9: Comparaison avec DBSCAN")
print("-" * 60)

dbscan = DBSCAN(eps=5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
n_noise = list(dbscan_clusters).count(-1)

print(f"✓ DBSCAN: {n_clusters_dbscan} clusters trouvés")
print(f"✓ Points de bruit: {n_noise}")

if n_clusters_dbscan > 1:
    silhouette_dbscan = silhouette_score(X_scaled[dbscan_clusters != -1], 
                                         dbscan_clusters[dbscan_clusters != -1])
    print(f"✓ Silhouette Score DBSCAN: {silhouette_dbscan:.4f}")

# =========================================================
# 10. SAUVEGARDE DU MODÈLE ET DES OBJETS
# =========================================================

print("\n\n💾 Étape 10: Sauvegarde du Modèle et des Objets")
print("-" * 60)

# Sauvegarder le modèle K-Means
joblib.dump(kmeans_final, 'musicmatch_kmeans_model.pkl')
print("✓ Modèle K-Means sauvegardé: musicmatch_kmeans_model.pkl")

# Sauvegarder le scaler
joblib.dump(scaler, 'musicmatch_scaler.pkl')
print("✓ Scaler sauvegardé: musicmatch_scaler.pkl")

# Sauvegarder le PCA
joblib.dump(pca, 'musicmatch_pca.pkl')
print("✓ PCA sauvegardé: musicmatch_pca.pkl")

# Sauvegarder les données avec clusters
user_features.to_csv('musicmatch_user_clusters.csv', index=False)
print("✓ Données avec clusters sauvegardées: musicmatch_user_clusters.csv")

# Sauvegarder la matrice utilisateur-artiste
user_artist_matrix.to_csv('musicmatch_user_artist_matrix.csv')
print("✓ Matrice utilisateur-artiste sauvegardée: musicmatch_user_artist_matrix.csv")

# Sauvegarder le mapping des noms de clusters
import json
with open('musicmatch_cluster_names.json', 'w') as f:
    json.dump(cluster_names, f, indent=2)
print("✓ Noms des clusters sauvegardés: musicmatch_cluster_names.json")

# =========================================================
# 11. EXEMPLE D'UTILISATION DU MODÈLE
# =========================================================

print("\n\n🎯 Étape 11: Exemple d'Utilisation du Modèle")
print("-" * 60)

# Charger le modèle
loaded_model = joblib.load('musicmatch_kmeans_model.pkl')
loaded_scaler = joblib.load('musicmatch_scaler.pkl')

# Simuler un nouvel utilisateur
print("\n📝 Exemple: Prédiction pour un nouvel utilisateur")
new_user_artists = ['Taylor Swift', 'Ed Sheeran', 'Ariana Grande']
print(f"Artistes du nouvel utilisateur: {new_user_artists}")

# Créer un vecteur pour le nouvel utilisateur
new_user_vector = np.zeros(len(user_artist_matrix.columns))
for artist in new_user_artists:
    if artist in user_artist_matrix.columns:
        idx = user_artist_matrix.columns.get_loc(artist)
        new_user_vector[idx] = np.random.randint(10, 50)

# Normaliser et prédire
new_user_scaled = StandardScaler().fit_transform([new_user_vector])
predicted_cluster = loaded_model.predict(new_user_scaled)[0]

print(f"\n✓ Cluster prédit: {predicted_cluster}")
print(f"✓ Profil: {cluster_names.get(predicted_cluster, 'N/A')}")

# =========================================================
# RAPPORT FINAL
# =========================================================

print("\n\n" + "=" * 60)
print("📊 RAPPORT FINAL - MUSICMATCH CLUSTERING")
print("=" * 60)
print(f"\n✅ Dataset: {len(df_clean)} écoutes, {df['UserID'].nunique()} utilisateurs")
print(f"✅ Nombre de clusters: {optimal_k}")
print(f"✅ Silhouette Score: {silhouette:.4f}")
print(f"✅ Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"✅ Calinski-Harabasz Score: {calinski_harabasz:.2f}")
print(f"\n✅ Fichiers générés:")
print(f"   - musicmatch_kmeans_model.pkl")
print(f"   - musicmatch_scaler.pkl")
print(f"   - musicmatch_pca.pkl")
print(f"   - musicmatch_user_clusters.csv")
print(f"   - musicmatch_user_artist_matrix.csv")
print(f"   - musicmatch_cluster_names.json")
print(f"   - musicmatch_eda.png")
print(f"   - musicmatch_optimal_k.png")
print(f"   - musicmatch_clusters_pca.png")
print("\n🎉 Training terminé avec succès!")
print("=" * 60)