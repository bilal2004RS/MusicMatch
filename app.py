# MusicMatch - Flask Backend Application
# =========================================

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import gradio as gr

def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()

# Initialisation de l'application Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'musicmatch_secret_key_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///musicmatch.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# =========================================================
# MODÈLES DE BASE DE DONNÉES
# =========================================================

class User(db.Model):
    """Modèle Utilisateur"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    cluster = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relations
    artists = db.relationship('UserArtist', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'cluster': self.cluster,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }


class UserArtist(db.Model):
    """Modèle Association Utilisateur-Artiste"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    artist_name = db.Column(db.String(200), nullable=False)
    genre = db.Column(db.String(50), nullable=False)
    play_count = db.Column(db.Integer, default=1)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'artist_name': self.artist_name,
            'genre': self.genre,
            'play_count': self.play_count,
            'added_at': self.added_at.strftime('%Y-%m-%d %H:%M:%S')
        }


# =========================================================
# CHARGEMENT DU MODÈLE ML
# =========================================================

class MLModel:
    """Classe pour gérer le modèle de Machine Learning"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.cluster_names = {}
        self.all_artists = []
        
    def load_model(self):
        """Charger le modèle et les objets associés"""
        try:
            if os.path.exists('musicmatch_kmeans_model.pkl'):
                self.model = joblib.load('musicmatch_kmeans_model.pkl')
                print("✓ Modèle K-Means chargé")
            
            if os.path.exists('musicmatch_scaler.pkl'):
                self.scaler = joblib.load('musicmatch_scaler.pkl')
                print("✓ Scaler chargé")
            
            if os.path.exists('musicmatch_pca.pkl'):
                self.pca = joblib.load('musicmatch_pca.pkl')
                print("✓ PCA chargé")
            
            if os.path.exists('musicmatch_cluster_names.json'):
                with open('musicmatch_cluster_names.json', 'r') as f:
                    self.cluster_names = json.load(f)
                print("✓ Noms des clusters chargés")
            
            # Charger la liste des artistes
            if os.path.exists('musicmatch_user_artist_matrix.csv'):
                df = pd.read_csv('musicmatch_user_artist_matrix.csv', index_col=0)
                self.all_artists = list(df.columns)
                print(f"✓ {len(self.all_artists)} artistes chargés")
            
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def predict_cluster(self, user_id):
        """Prédire le cluster d'un utilisateur"""
        if not self.model or not self.all_artists:
            return None
        
        # Récupérer les artistes de l'utilisateur
        user_artists = UserArtist.query.filter_by(user_id=user_id).all()
        
        if not user_artists:
            return None
        
        # Créer le vecteur utilisateur
        user_vector = np.zeros(len(self.all_artists))
        
        for ua in user_artists:
            if ua.artist_name in self.all_artists:
                idx = self.all_artists.index(ua.artist_name)
                user_vector[idx] = ua.play_count
        
        # Normaliser et prédire
        from sklearn.preprocessing import StandardScaler
        temp_scaler = StandardScaler()
        user_vector_scaled = temp_scaler.fit_transform([user_vector])
        
        cluster = self.model.predict(user_vector_scaled)[0]
        
        # Mettre à jour la BDD
        user = User.query.get(user_id)
        if user:
            user.cluster = int(cluster)
            db.session.commit()
        
        return int(cluster)
    
    def get_cluster_name(self, cluster_id):
        """Obtenir le nom d'un cluster"""
        return self.cluster_names.get(str(cluster_id), f"Cluster {cluster_id}")


# Instance du modèle ML
ml_model = MLModel()


# =========================================================
# ROUTES - PAGES HTML
# =========================================================

@app.route('/')
def index():
    """Page d'accueil"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Page d'inscription"""
    if request.method == 'POST':
        data = request.get_json()
        
        # Vérifier si l'utilisateur existe
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': 'Email déjà utilisé'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'success': False, 'message': 'Nom d\'utilisateur déjà pris'}), 400
        
        # Créer un nouveau compte
        user = User(username=data['username'], email=data['email'])
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        
        return jsonify({'success': True, 'message': 'Compte créé avec succès', 'user': user.to_dict()})
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Page de connexion"""
    if request.method == 'POST':
        data = request.get_json()
        
        user = User.query.filter_by(email=data['email']).first()
        
        if user and user.check_password(data['password']):
            session['user_id'] = user.id
            return jsonify({'success': True, 'message': 'Connexion réussie', 'user': user.to_dict()})
        
        return jsonify({'success': False, 'message': 'Email ou mot de passe incorrect'}), 401
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Déconnexion"""
    session.pop('user_id', None)
    return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    """Tableau de bord utilisateur"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('logout'))
    
    return render_template('dashboard.html', user=user)


@app.route('/analytics')
def analytics():
    """Page d'analytics"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('analytics.html')


# =========================================================
# API ENDPOINTS
# =========================================================

@app.route('/api/user/profile')
def get_user_profile():
    """Obtenir le profil utilisateur"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Non authentifié'}), 401
    
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Utilisateur introuvable'}), 404
    
    user_data = user.to_dict()
    user_data['artists'] = [ua.to_dict() for ua in user.artists]
    user_data['cluster_name'] = ml_model.get_cluster_name(user.cluster) if user.cluster is not None else None
    
    return jsonify({'success': True, 'user': user_data})


@app.route('/api/artists/add', methods=['POST'])
def add_artist():
    """Ajouter un artiste aux préférences"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Non authentifié'}), 401
    
    data = request.get_json()
    
    # Vérifier si l'artiste existe déjà
    existing = UserArtist.query.filter_by(
        user_id=session['user_id'],
        artist_name=data['artist_name']
    ).first()
    
    if existing:
        existing.play_count += data.get('play_count', 1)
        db.session.commit()
    else:
        artist = UserArtist(
            user_id=session['user_id'],
            artist_name=data['artist_name'],
            genre=data['genre'],
            play_count=data.get('play_count', 1)
        )
        db.session.add(artist)
        db.session.commit()
    
    # Recalculer le cluster
    cluster = ml_model.predict_cluster(session['user_id'])
    
    return jsonify({
        'success': True,
        'message': 'Artiste ajouté',
        'cluster': cluster,
        'cluster_name': ml_model.get_cluster_name(cluster) if cluster is not None else None
    })


@app.route('/api/artists/remove/<int:artist_id>', methods=['DELETE'])
def remove_artist(artist_id):
    """Supprimer un artiste"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Non authentifié'}), 401
    
    artist = UserArtist.query.filter_by(id=artist_id, user_id=session['user_id']).first()
    
    if not artist:
        return jsonify({'success': False, 'message': 'Artiste introuvable'}), 404
    
    db.session.delete(artist)
    db.session.commit()
    
    # Recalculer le cluster
    cluster = ml_model.predict_cluster(session['user_id'])
    
    return jsonify({'success': True, 'message': 'Artiste supprimé', 'cluster': cluster})


@app.route('/api/recommendations')
def get_recommendations():
    """Obtenir des recommandations personnalisées"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Non authentifié'}), 401
    
    user = User.query.get(session['user_id'])
    
    if not user or user.cluster is None:
        return jsonify({'success': True, 'recommendations': []})
    
    # Trouver les utilisateurs du même cluster
    similar_users = User.query.filter(
        User.cluster == user.cluster,
        User.id != user.id
    ).limit(10).all()
    
    # Récupérer les artistes de l'utilisateur
    user_artist_names = [ua.artist_name.lower() for ua in user.artists]
    
    # Compter les artistes populaires dans le cluster
    artist_counts = {}
    for similar_user in similar_users:
        for ua in similar_user.artists:
            if ua.artist_name.lower() not in user_artist_names:
                artist_counts[ua.artist_name] = artist_counts.get(ua.artist_name, 0) + 1
    
    # Trier et retourner les top recommandations
    recommendations = [
        {'artist_name': name, 'popularity': count}
        for name, count in sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    return jsonify({'success': True, 'recommendations': recommendations})


@app.route('/api/similar-users')
def get_similar_users():
    """Trouver des utilisateurs similaires"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Non authentifié'}), 401
    
    user = User.query.get(session['user_id'])
    
    if not user or user.cluster is None:
        return jsonify({'success': True, 'similar_users': []})
    
    # Trouver les utilisateurs du même cluster
    similar_users = User.query.filter(
        User.cluster == user.cluster,
        User.id != user.id
    ).limit(10).all()
    
    result = []
    for similar_user in similar_users:
        result.append({
            'id': similar_user.id,
            'username': similar_user.username,
            'num_artists': len(similar_user.artists),
            'cluster': similar_user.cluster
        })
    
    return jsonify({'success': True, 'similar_users': result})


@app.route('/api/statistics')
def get_statistics():
    """Statistiques globales de la plateforme"""
    total_users = User.query.count()
    total_artists = UserArtist.query.count()
    
    # Distribution des clusters
    cluster_distribution = {}
    for i in range(6):  # 6 clusters possibles
        count = User.query.filter_by(cluster=i).count()
        if count > 0:
            cluster_distribution[i] = {
                'count': count,
                'name': ml_model.get_cluster_name(i),
                'percentage': round((count / total_users * 100) if total_users > 0 else 0, 1)
            }
    
    # Genres les plus populaires
    genre_counts = db.session.query(
        UserArtist.genre,
        db.func.count(UserArtist.id)
    ).group_by(UserArtist.genre).all()
    
    return jsonify({
        'success': True,
        'statistics': {
            'total_users': total_users,
            'total_artists': total_artists,
            'cluster_distribution': cluster_distribution,
            'top_genres': [{'genre': g[0], 'count': g[1]} for g in sorted(genre_counts, key=lambda x: x[1], reverse=True)]
        }
    })


# =========================================================
# INITIALISATION
# =========================================================

def init_db():
    """Initialiser la base de données"""
    with app.app_context():
        db.create_all()
        print("✓ Base de données initialisée")


def create_sample_users():
    """Créer des utilisateurs de test"""
    with app.app_context():
        if User.query.count() == 0:
            # Créer quelques utilisateurs de test
            users_data = [
                {'username': 'alice', 'email': 'alice@musicmatch.com', 'password': 'password123'},
                {'username': 'bob', 'email': 'bob@musicmatch.com', 'password': 'password123'},
                {'username': 'charlie', 'email': 'charlie@musicmatch.com', 'password': 'password123'},
            ]
            
            for user_data in users_data:
                user = User(username=user_data['username'], email=user_data['email'])
                user.set_password(user_data['password'])
                db.session.add(user)
            
            db.session.commit()
            print(f"✓ {len(users_data)} utilisateurs de test créés")


# =========================================================
# DÉMARRAGE DE L'APPLICATION
# =========================================================

if __name__ == '__main__':
    # Initialiser la base de données
    init_db()
    
    # Charger le modèle ML
    print("\n🤖 Chargement du modèle ML...")
    if ml_model.load_model():
        print("✓ Modèle ML prêt\n")
    else:
        print("⚠️ Modèle ML non trouvé. Exécutez d'abord le script de training.\n")
    
    # Créer des utilisateurs de test (optionnel)
    # create_sample_users()
    
    # Démarrer l'application
    print("=" * 60)
    print("🎵 MusicMatch - Serveur Flask")
    print("=" * 60)
    print("Application disponible sur: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
