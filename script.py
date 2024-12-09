import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import json
import joblib

random_state = 42

sessions_tracks_path = "sessions/sessions_tracks.json"
user_sessions_path = "sessions/user_sessions.json"

with open(sessions_tracks_path, 'r') as file:
    sessions_tracks = json.load(file)
    
with open(user_sessions_path, 'r') as file:
    user_sessions = json.load(file)
    
track_info_df = pd.read_csv("tracks/track_id_info.csv")
track_info_dict = {}
for index, row in track_info_df.iterrows():
    track_info_dict[row['index']] = {
        "artistname": row['artistname'],
        "trackname": row['trackname'],
    }
    
def temporal_split(sessions, train_size=0.8, test_size=0.2):
    n_sessions = len(sessions)
    train_end = int(n_sessions * train_size)
    return { 
        "train": sessions[:train_end],
        "test": sessions[train_end:]        
    }
    
def extract_track_ids_from_session(user_sessions):
    tracks = set()
    for sessions in user_sessions.values():
        for session in sessions:
            if str(session) not in sessions_tracks: continue
            session_tracks = sessions_tracks[str(session)]
            for track in session_tracks: tracks.add(track)
        
    return list(tracks)
        

train_user_sessions = {}
for user, sessions in user_sessions.items():
    train_user_sessions[user] = temporal_split(sessions)["train"]

    
test_user_sessions = {}
for user, sessions in user_sessions.items():
    test_user_sessions[user] = temporal_split(sessions)["test"]

"""
Lendo corpus

corpus_df = pd.read_csv('lyrics/corpus.csv', delimiter=',')
corpus_dict = {}
for index, row in corpus_df.iterrows():
    corpus_dict[row['track_uri']] = row['lyric']
"""

"""
Treinando o modelo Word2Vec

tokenized_lyrics = [simple_preprocess(lyric) for lyric in corpus_dict.values()]
model = Word2Vec(sentences=tokenized_lyrics, vector_size=100)

model.save("models/music_lyrics_word2vec_TRAIN.model")
"""

model = Word2Vec.load("models/music_lyrics_word2vec_TRAIN.model")

vocab = list(model.wv.index_to_key)
# print(f"Vocabulary: {vocab}")
# print(f"Vocabulary size: {len(vocab)}")

"""
def get_average_vector(tokens, model):
  vectors = [model.wv[word] for word in tokens if word in model.wv]
  if vectors:
    return np.mean(vectors, axis=0)
  else:
    return np.zeros(model.vector_size)

tracks_vector_array = []
tracks_vector_dict = {}
for track_id, lyric in corpus_dict.items():
    tokens = simple_preprocess(lyric)
    vector = get_average_vector(tokens, model)
    tracks_vector_dict[track_id] = vector
    row = {'track_id': track_id}
    row.update({f'vector_{i}': value for i, value in enumerate(vector)})
    tracks_vector_array.append(row)

tracks_vector_df = pd.DataFrame(tracks_vector_array)
tracks_vector_df.to_csv(f'lyrics/track_word2vec.csv', index=False)
print(len(tracks_vector_dict))
"""

tracks_vector_df = pd.read_csv(f'tracks/track_word2vec.csv', delimiter=',', on_bad_lines='skip')
tracks_vector_dict = {}

for _, row in tracks_vector_df.iterrows():
    track_id = row['track_id']
    vector = row.filter(like='vector_').to_numpy()
    tracks_vector_dict[track_id] = vector
    

sessions_vector_array = []
sessions_vector_dict = {}
for session, tracks in sessions_tracks.items():
    session_tracks_vector = []
    for track in tracks:
        session_tracks_vector.append(tracks_vector_dict[track])
    
    if len(session_tracks_vector) > 0:
        session_tracks_vector_mean = np.mean(np.array(session_tracks_vector), axis=0)
        sessions_vector_dict[session] = session_tracks_vector_mean
        
        row = {'session': session}
        row.update({f'vector_{i}': value for i, value in enumerate(sessions_vector_dict[session])})
        sessions_vector_array.append(row)
    

sessions_vector_df = pd.DataFrame(sessions_vector_array)
sessions_vector_df.to_csv(f'sessions/session_word2vec.csv', index=False)
print(len(sessions_vector_dict))

sessions_vector_df = pd.read_csv(f'sessions/session_word2vec.csv', delimiter=',', on_bad_lines='skip')
sessions_vector_dict = {}

for _, row in sessions_vector_df.iterrows():
    session = row['session']
    vector = row.filter(like='vector_').to_numpy()
    sessions_vector_dict[session] = vector
    
train_sessions_path = 'sessions/train_sessions.json'

#Saving train sessions
train_sessions = np.concatenate((list(sessions_vector_dict.values())), axis=0).tolist()

with open(train_sessions_path, "w") as json_file:
    json.dump(train_sessions, json_file, indent=4)

with open(train_sessions_path, 'r') as file:
    train_sessions = json.load(file)
    
train_sessions_vector_dict = {session: vector for session, vector in sessions_vector_dict.items() if session in train_sessions}


# Treinando o KMeans
print("Treinando o KMeans")

sessions = list(train_sessions_vector_dict.keys())
vectors = np.array(list(train_sessions_vector_dict.values()))

n_clusters = 100
model = KMeans(n_clusters=n_clusters, random_state=random_state)
clusters = model.fit_predict(vectors)

results_df = pd.DataFrame({
    'session': sessions,
    'cluster': clusters
})

results_df.to_csv('sessions/kmeans_session_clusters.csv', index=False)

print("Exportando o KMeans")
joblib.dump(model, f'models/kmeans_model_{n_clusters}.pkl')

session_clusters_df = pd.read_csv('sessions/kmeans_session_clusters.csv')
cluster_sessions = {}
cluster_tracks_count = {}

cluster_tracks_vector = {}
for index, row in session_clusters_df.iterrows():
    session = int(row["session"])
    cluster = row["cluster"]
    if cluster not in cluster_sessions: cluster_sessions[cluster] = []
    if cluster not in cluster_tracks_count: cluster_tracks_count[cluster] = {}
    
    if cluster not in cluster_tracks_vector: cluster_tracks_vector[cluster] = {}
    cluster_sessions[cluster].append(session)
    tracks = sessions_tracks[str(session)]
    for index, track in enumerate(tracks):
        cluster_tracks_vector[cluster][track] = tracks_vector_dict[track]
        
        if track in tracks[:index]: continue
        if track not in cluster_tracks_count[cluster]: cluster_tracks_count[cluster][track] = 0
        cluster_tracks_count[cluster][track] += 1
        
cluster_tracks_sorted = {}
for cluster, tracks in cluster_tracks_count.items():
    sorted_tracks = sorted(tracks.items(), key=lambda x: x[1], reverse=True)
    cluster_tracks_sorted[cluster] = [track for track, _ in sorted_tracks]

def split_session_to_predict(session_tracks, threshold=0.9):
    n_tracks = len(session_tracks)
    tracks_end = int(n_tracks * threshold)
    return { 
        "x": session_tracks[:tracks_end],
        "Y": session_tracks[tracks_end:],
    }
    
def calculate_session_vector(session_tracks):
    session_tracks_vector = []
    for track in session_tracks:
        if track in tracks_vector_dict: 
            session_tracks_vector.append(tracks_vector_dict[track])
    
    if session_tracks_vector:
        session_vector = np.mean(np.array(session_tracks_vector), axis=0)
    else:
        session_vector = np.zeros(100)
    
    return session_vector

def predict_next_session_song(session_tracks, model):
    session_vector = calculate_session_vector(session_tracks)
    session_vector = np.array(session_vector, dtype=np.float64)
    
    if np.isnan(session_vector).any():
        raise ValueError("Session vector contains NaN values.")
    
    session_vector = session_vector.reshape(1, -1)
    result = model.predict(session_vector)
    
    return result

def get_most_frequent_track_in_cluster(cluster, tracks_to_remove):
    tracks = cluster_tracks_sorted[cluster]
    available_tracks = [track for track in tracks if track not in tracks_to_remove]
    if not available_tracks:
        raise ValueError("No available tracks in cluster after removing forbidden tracks")
    return available_tracks[0]

def find_most_similar_vector_in_dict(query_vector, dict, items_to_remove=[]):
    max_similarity = -1
    most_similar_key = None
    
    for key, vector in dict.items():
        if key in items_to_remove: continue
        vector = np.array(vector).reshape(1, -1)
        query_vector_reshaped = np.array(query_vector).reshape(1, -1)
        
        similarity = cosine_similarity(query_vector_reshaped, vector)[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_key = key
    
    return most_similar_key

def find_most_similar_track(track_to_find_similar, tracks, tracks_to_remove=[]):
    if not tracks:
        raise ValueError("Empty tracks list provided")
    
    available_tracks = [t for t in tracks if t not in tracks_to_remove]
    if not available_tracks:
        raise ValueError("No available tracks after removing forbidden tracks")
    
    max_similarity = -1
    most_similar_track = None
    
    vector1 = np.array(tracks_vector_dict[track_to_find_similar]).reshape(1, -1)
    
    for track in available_tracks:
        vector2 = np.array(tracks_vector_dict[track]).reshape(1, -1)
        similarity = cosine_similarity(vector1, vector2)[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_track = track
    
    return most_similar_track

def get_most_similar_track_to_last_track_in_the_most_similar_session_in_cluster(cluster, track, session, tracks_to_remove):
    if cluster not in cluster_sessions:
        raise ValueError(f"Invalid cluster: {cluster}")
    if track not in tracks_vector_dict:
        raise ValueError(f"Invalid track: {track}")
    if str(session) not in sessions_tracks:
        raise ValueError(f"Invalid session: {session}")
    
    session_vector = sessions_vector_dict[session]
    
    sessions_vector_cluster = {}
    for sess_id in cluster_sessions[cluster]:
        sess_tracks = sessions_tracks[str(int(sess_id))]
        available_tracks = [t for t in sess_tracks if t not in tracks_to_remove]
        if available_tracks:
            sessions_vector_cluster[sess_id] = sessions_vector_dict[sess_id]
    
    if not sessions_vector_cluster:
        raise ValueError("No valid sessions found in cluster after filtering")
    
    most_similar_session = find_most_similar_vector_in_dict(session_vector, sessions_vector_cluster)
    if most_similar_session is None:
        raise ValueError("Could not find similar session")
    
    tracks = sessions_tracks[str(int(most_similar_session))]
    most_similar_track = find_most_similar_track(track, tracks, tracks_to_remove)
    
    if most_similar_track is None:
        raise ValueError("Could not find similar track")
        
    return most_similar_track

def get_most_similar_track_to_session_in_cluster(cluster, session, tracks_to_remove):
    if cluster not in cluster_sessions:
        raise ValueError(f"Invalid cluster: {cluster}")
    if str(session) not in sessions_tracks:
        raise ValueError(f"Invalid session: {session}")
    
    session_vector = sessions_vector_dict[session]

    most_similar_track = find_most_similar_vector_in_dict(session_vector, cluster_tracks_vector[cluster], tracks_to_remove)
    
    if most_similar_track is None:
        raise ValueError("Could not find similar track")
        
    return most_similar_track
    

for threshold in (0.6, 0.9):
    print(threshold)
    kmeans_model = joblib.load(f'models/kmeans_model_{n_clusters}.pkl')
    def get_predictions():
        predictions = {}
        total_users = len(test_user_sessions.keys())
        i = 1
        for user, sessions in test_user_sessions.items():
            print(f'User {i} of {total_users} users')
            i += 1
            predictions[user] = {}
            
            total_sessions = len(sessions)
            s = 1
            for session in sessions:
                print(f'Session {s} of {total_sessions} session from user {i}')
                s += 1
                if str(session) not in sessions_tracks:
                    continue
                    
                session_tracks = sessions_tracks[str(session)]
                session_split = split_session_to_predict(session_tracks, threshold)
                
                base_session_similarity = session_split["x"].copy()
                base_session_frequency = session_split["x"].copy()
                
                base_tracks = session_split["x"].copy()
                if len(base_session_similarity) == 0: continue
                
                Y = session_split["Y"]
                y_similarity_similar_in_cluster = []
                y_frequency = []
                
                for _ in range(len(Y)):
                    try:
                        predicted_cluster = predict_next_session_song(base_tracks, kmeans_model)[0]
                                            
                        predicted_track_similarity_in_cluster = get_most_similar_track_to_session_in_cluster(
                            predicted_cluster,
                            session,
                            base_session_similarity
                        )
                        
                        predicted_track_frequency = get_most_frequent_track_in_cluster(
                            predicted_cluster,
                            base_session_frequency
                        )
                        
                        base_session_similarity.append(predicted_track_similarity_in_cluster)
                        base_session_frequency.append(predicted_track_frequency)
                        
                        y_similarity_similar_in_cluster.append(predicted_track_similarity_in_cluster)

                        y_frequency.append(predicted_track_frequency)
                        
                    except ValueError as e:
                        print(f"Error processing prediction: {e}")
                        break
                
                predictions[user][session] = {
                    "x": session_split["x"],
                    "Y": Y,
                    "y_similarity_similar_in_cluster": y_similarity_similar_in_cluster,
                    "y_frequency": y_frequency
                }

        return predictions

    predictions = get_predictions()
        
    import json
    with open(f'results/predictions_{threshold}.json', "w") as json_file:
        json.dump(predictions, json_file, indent=4)