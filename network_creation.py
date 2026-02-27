import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import community as community_louvain
from collections import  defaultdict
import random
import os


def load_snap_network(file_path):
    """
    Loading the SNAP facebook network from an edge list file

    """
    G =  nx.Graph()
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):  # Skipping comment lines
                continue
            nodes = line.strip().split()
            if len(nodes) >= 2:
                G.add_edge(int(nodes[0]), int(nodes[1]))
    
    print(f"Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def load_movielens_data(ratings_file, movies_file):
    """
    Loaing MovieLens dataset
    """
    # Loading ratings
    ratings = pd.read_csv(ratings_file)
    
    # Loading movies with genres
    movies = pd.read_csv(movies_file)
    
    # Extracting genre information
    all_genres = set()
    for genres in movies['genres'].str.split('|'):
        all_genres.update(genres)
    
    print(f"Loaded MovieLens data with {len(ratings)} ratings and {len(all_genres)} unique genres")
    return ratings, movies, list(all_genres)


def detect_communities(G):
    """
    Detecting communities in network using Louvain algorithm
    
    """
    print("Detecting communities...")
    
    try:
        G_copy = G.copy()
        needs_relabeling = not all(isinstance(n, int) for n in G_copy.nodes())
        
        if needs_relabeling:
            print("Converting node IDs to integers...")
            mapping = {node: i for i, node in enumerate(G_copy.nodes())}
            G_copy = nx.relabel_nodes(G_copy, mapping)
            reverse_mapping = {i: node for node, i in mapping.items()}
        
        # Finding communities
        print("Running Louvain community detection...")
        partition = community_louvain.best_partition(G_copy)
        
        # Map community IDs back to original node IDs if needed
        if needs_relabeling:
            partition = {reverse_mapping[node]: comm for node, comm in partition.items()}
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        
        print(f"Detected {len(communities)} communities")
        return partition, communities
        
    except Exception as e:
        print(f"Error during community detection: {e}")
        print("Falling back to simple partition (one community)...")
        
        # Creating a fallback partition where all nodes are in the same community
        partition = {node: 0 for node in G.nodes()}
        communities = {0: list(G.nodes())}
        
        return partition, communities

def create_homophilic_preferences(G, partition, genres, homophily_strength=0.7):
    """
    Assign genre preferences to nodes with community-based homophily

    """
    # Creating base preferences for each community
    community_preferences = {}
    for community_id in set(partition.values()):  
        # Each community gets a random distribution of genre preferences
        prefs = np.random.dirichlet(np.ones(len(genres)) * 0.5)  
        community_preferences[community_id] = prefs
    
    # Assigning preferences to each node
    node_preferences = {}
    for node in G.nodes():
        if node in partition:  
            community_id = partition[node]
            base_prefs = community_preferences[community_id]
            
            # Mixing community preferences with random noise
            if homophily_strength < 1:
                individual_noise = np.random.dirichlet(np.ones(len(genres)) * 0.5)
                mixed_prefs = homophily_strength * base_prefs + (1 - homophily_strength) * individual_noise
                # Normalizing
                mixed_prefs = mixed_prefs / mixed_prefs.sum()
            else:
                mixed_prefs = base_prefs
                
            node_preferences[node] = {genres[i]: mixed_prefs[i] for i in range(len(genres))}
        else:
            # If node doesn't have a community, assigning random preferences
            prefs = np.random.dirichlet(np.ones(len(genres)) * 0.5)
            node_preferences[node] = {genres[i]: prefs[i] for i in range(len(genres))}
    
    return node_preferences


def measure_preference_homophily(G, node_preferences):
    """
    Measure how similar connected nodes' preferences are compared to random pairs
    """
    print("Measuring preference homophily...")
    
    # Calculate similarity between connected nodes
    edge_similarities = []
    for u, v in G.edges():
        try:
            if u in node_preferences and v in node_preferences:
                genres = sorted(set(node_preferences[u].keys()) | set(node_preferences[v].keys()))
                u_prefs = np.array([node_preferences[u].get(g, 0) for g in genres])
                v_prefs = np.array([node_preferences[v].get(g, 0) for g in genres])
                
                # Calculating cosine similarity
                u_norm = np.linalg.norm(u_prefs)
                v_norm = np.linalg.norm(v_prefs)
                if u_norm > 0 and v_norm > 0:
                    similarity = np.dot(u_prefs, v_prefs) / (u_norm * v_norm)
                    edge_similarities.append(similarity)
        except Exception as e:
            print(f"Error calculating similarity for edge ({u}, {v}): {e}")
            continue
    
    # Calculating similarity between random node pairs
    random_similarities = []
    nodes = list(G.nodes())
    attempts = 0
    max_attempts = min(len(G.edges()) * 2, 2000)
    
    while len(random_similarities) < min(len(edge_similarities), 1000) and attempts < max_attempts:
        attempts += 1
        try:
            u, v = random.sample(nodes, 2)
            if u != v and not G.has_edge(u, v) and u in node_preferences and v in node_preferences:
                genres = sorted(set(node_preferences[u].keys()) | set(node_preferences[v].keys()))
                u_prefs = np.array([node_preferences[u].get(g, 0) for g in genres])
                v_prefs = np.array([node_preferences[v].get(g, 0) for g in genres])
                
                # Calculating cosine similarity
                u_norm = np.linalg.norm(u_prefs)
                v_norm = np.linalg.norm(v_prefs)
                if u_norm > 0 and v_norm > 0:
                    similarity = np.dot(u_prefs, v_prefs) / (u_norm * v_norm)
                    random_similarities.append(similarity)
        except Exception as e:
            print(f"Error calculating similarity for random pair: {e}")
            continue
    
    if edge_similarities and random_similarities:
        print(f"Mean similarity between connected nodes: {np.mean(edge_similarities):.4f}")
        print(f"Mean similarity between random node pairs: {np.mean(random_similarities):.4f}")
        
    else:
        print("Warning: Could not calculate enough similarity values for valid comparison")
    
    return edge_similarities, random_similarities


def generate_synthetic_ratings(node_preferences, movies_df, num_ratings_per_user=20):
    """
    Generate synthetic movie ratings based on user preferences
    """
    print("Generating synthetic ratings...")
    ratings_data = []
    
    try:
        # Creating genre dictionary for each movie
        movie_genres = {}
        for _, row in movies_df.iterrows():
            try:
                movie_id = row['movieId']
                if isinstance(row['genres'], str):
                    if '|' in row['genres']:
                        genre_list = row['genres'].split('|')
                    else:
                        genre_list = [g.strip() for g in row['genres'].split(',')]
                elif isinstance(row['genres'], list):
                    genre_list = row['genres']
                else:
                    genre_list = []
                
                genre_dict = {genre: 1 for genre in genre_list if genre} 
                movie_genres[movie_id] = genre_dict
            except Exception as e:
                print(f"Error processing movie {row.get('movieId', 'unknown')}: {e}")
                continue
        
        # Generating ratings for each user based on their preferences
        user_count = 0
        rating_count = 0
        
        for user_id, preferences in node_preferences.items():
            try:
                user_count += 1
                if user_count % 1000 == 0:
                    print(f"Processed {user_count} users...")
                
                # Skip users with no preferences
                if not preferences:
                    continue
                
                # Calculating preference scores for each movie
                movie_scores = {}
                for movie_id, movie_genre_dict in movie_genres.items():
                    if not movie_genre_dict:
                        continue
                        
                    score = 0
                    for genre, present in movie_genre_dict.items():
                        if genre in preferences:
                            score += preferences[genre]
                    movie_scores[movie_id] = score
                
                # Select top movies based on preference score
                scored_movies = [(mid, score) for mid, score in movie_scores.items() if score > 0]
                num_to_select = min(num_ratings_per_user, len(scored_movies))
                
                if num_to_select > 0:
                    top_movies = sorted(scored_movies, key=lambda x: x[1], reverse=True)[:num_to_select]
                    
                    # Generating ratings with some noise
                    for movie_id, score in top_movies:
                        base_score = min(score * 5, 5)  # Scale to 0-5 range, capping at 5
                        noise = np.random.normal(0, 0.5)  # Adding some randomness
                        rating = min(max(base_score + noise, 0.5), 5)  # Constraining to 0.5-5 range
                        
                        ratings_data.append({
                            'userId': user_id,
                            'movieId': movie_id,
                            'rating': rating,
                            'timestamp': int(pd.Timestamp.now().timestamp())
                        })
                        rating_count += 1
            except Exception as e:
                print(f"Error generating ratings for user {user_id}: {e}")
                continue
        
        ratings_df = pd.DataFrame(ratings_data)
        print(f"Generated {len(ratings_df)} synthetic ratings for {user_count} users")
        return ratings_df
        
    except Exception as e:
        print(f"Error in generate_synthetic_ratings: {e}")
        # Returning empty DataFrame as fallback
        return pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])



def create_movie_network_dataset(network_file, movielens_ratings_file, movielens_movies_file, 
                                output_dir, homophily_strength=0.7):
    """
    Main function to create a network with movie preferences
    """


    os.makedirs(output_dir, exist_ok=True)
    
    # Loading network
    G = load_snap_network(network_file)
    
    # Loading MovieLens data
    ratings_df, movies_df, genres = load_movielens_data(movielens_ratings_file, movielens_movies_file)
    
    # Detect communities
    partition, _ = detect_communities(G)
    
    # Creating homophilic preferences
    node_preferences = create_homophilic_preferences(G, partition, genres, homophily_strength)
    
    # Evaluate homophily
    edge_sims, random_sims = measure_preference_homophily(G, node_preferences)
    
    # Generate synthetic ratings
    synthetic_ratings = generate_synthetic_ratings(node_preferences, movies_df)
    
    # Save outputs
    nx.write_edgelist(G, f"{output_dir}/network.edgelist")
    pd.DataFrame([(node, genre, pref) for node, prefs in node_preferences.items() 
                  for genre, pref in prefs.items()],
                 columns=['userId', 'genre', 'preference']).to_csv(f"{output_dir}/user_preferences.csv", index=False)
    synthetic_ratings.to_csv(f"{output_dir}/synthetic_ratings.csv", index=False)
    
    # Create and save visualizations
    plt.figure(figsize=(10, 6))
    plt.hist(edge_sims, alpha=0.5, label='Connected Nodes')
    plt.hist(random_sims, alpha=0.5, label='Random Pairs')
    plt.xlabel('Preference Similarity (Cosine)')
    plt.ylabel('Frequency')
    plt.title('Homophily in Genre Preferences')
    plt.legend()
    plt.savefig(f"{output_dir}/preference_homophily.png")
    
    return G, node_preferences, synthetic_ratings


# Paths to data files
facebook_network_file = "facebook_combined.txt"
movielens_ratings_file = "ml-32m/ratings.csv" 
movielens_movies_file = "ml-32m/movies.csv"


create_movie_network_dataset(
    facebook_network_file,
    movielens_ratings_file,
    movielens_movies_file,
    "output/facebook_movie_network/high",
    homophily_strength=0.7
)







