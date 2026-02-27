import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import community as community_louvain
import random
import time
from matplotlib.lines import Line2D

def load_network_data(data_dir):
    """
    Load network data from the specified directory
    
    """
    print(f"Loading network data from {data_dir}...")
    
    # Loading network
    network_file = os.path.join(data_dir, "network.edgelist")
    G = nx.read_edgelist(network_file, nodetype=int)
    print(f"Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Loading user preferences
    prefs_file = os.path.join(data_dir, "user_preferences.csv")
    prefs_df = pd.read_csv(prefs_file)
    
    user_preferences = {}
    for _, row in prefs_df.iterrows():
        user_id = int(row['userId'])
        genre = row['genre']
        preference = row['preference']
        
        if user_id not in user_preferences:
            user_preferences[user_id] = {}
        
        user_preferences[user_id][genre] = preference
    
    return G, user_preferences


def select_seed_nodes(G, user_preferences, movie_genres, strategy='random', seed_count=100, communities=None):
    """
    Select seed nodes based on the specified strategy
    """

    if strategy == 'random':
        # Randomly select nodes
        return random.sample(list(G.nodes()), min(seed_count, G.number_of_nodes()))
        
    elif strategy == 'degree':
        # Select nodes with highest degree
        degree_centrality = nx.degree_centrality(G)
        sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:seed_count]]
        
    elif strategy == 'preference':
        # Select nodes with highest preference for the movie genres
        match_scores = {}
        for user_id, preferences in user_preferences.items():
            if user_id in G:
                # Sum preferences for movie genres
                match_score = sum(preferences.get(genre, 0) for genre in movie_genres)
                match_scores[user_id] = match_score
        
        sorted_users = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
        return [user for user, _ in sorted_users[:seed_count]]
    
 
    elif strategy == 'largest_communities':
        if communities is None:
            print(f"Warning: {strategy} strategy requires communities, falling back to degree strategy")
            degree_centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:seed_count]]
        return largest_communities_seeding(G, communities, seed_count)
        
    elif strategy == 'community_balanced':
        if communities is None:
            print(f"Warning: {strategy} strategy requires communities, falling back to degree strategy")
            degree_centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:seed_count]]
        return community_balanced_seeding(G, communities, seed_count)
        
    elif strategy == 'community_proportional':
        if communities is None:
            print(f"Warning: {strategy} strategy requires communities, falling back to degree strategy")
            degree_centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:seed_count]]
        return community_proportional_seeding(G, communities, seed_count)
        
    elif strategy == 'community_bridges':
        if communities is None:
            print(f"Warning: {strategy} strategy requires communities, falling back to degree strategy")
            degree_centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:seed_count]]
        return community_bridges_seeding(G, communities, seed_count)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    

def community_balanced_seeding(G, communities, seed_count):
    """
    Distribute seeds equally across all communities
    """
    # Count communities
    community_ids = set(communities.values())
    seeds_per_community = seed_count // len(community_ids)
    remainder = seed_count % len(community_ids)
    
    seed_nodes = []
    for comm_id in community_ids:
        # Get nodes in this community
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        
        # Get degree for each node in community
        comm_degrees = {n: G.degree(n) for n in comm_nodes}
        
        # Sort by degree
        sorted_nodes = sorted(comm_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Add extra seed to largest communities if we have remainder
        n_seeds = seeds_per_community
        if remainder > 0:
            n_seeds += 1
            remainder -= 1
            
        # Add top nodes as seeds
        seed_nodes.extend([n for n, _ in sorted_nodes[:n_seeds]])
    
    return seed_nodes

def community_proportional_seeding(G, communities, seed_count):
    """
    Distribute seeds proportionally to community size
    """
    # Count nodes in each community
    community_sizes = {}
    for comm_id in set(communities.values()):
        community_sizes[comm_id] = sum(1 for c in communities.values() if c == comm_id)
    
    # Calculate seeds per community proportionally
    total_nodes = G.number_of_nodes()
    seeds_allocated = 0
    seed_allocation = {}
    
    for comm_id, size in community_sizes.items():
        # Calculate proportion of total network
        proportion = size / total_nodes
        # Allocate seeds based on proportion
        n_seeds = max(1, int(proportion * seed_count))
        seed_allocation[comm_id] = n_seeds
        seeds_allocated += n_seeds
    
    # Adjust if allocated too many or too few seeds
    while seeds_allocated != seed_count:
        if seeds_allocated < seed_count:
            for comm_id in sorted(community_sizes, key=community_sizes.get, reverse=True):
                seed_allocation[comm_id] += 1
                seeds_allocated += 1
                if seeds_allocated == seed_count:
                    break
        else:
            # Remove seeds from smallest communities
            for comm_id in sorted(community_sizes, key=community_sizes.get):
                if seed_allocation[comm_id] > 1:  # Ensuring at least 1 seed per community
                    seed_allocation[comm_id] -= 1
                    seeds_allocated -= 1
                    if seeds_allocated == seed_count:
                        break
    
    # Select top degree nodes from each community based on allocation
    seed_nodes = []
    for comm_id, n_seeds in seed_allocation.items():
        # Get nodes in this community
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        
        # Get degree for each node in community
        comm_degrees = {n: G.degree(n) for n in comm_nodes}
        
        # Sort by degree
        sorted_nodes = sorted(comm_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Add top nodes as seeds
        seed_nodes.extend([n for n, _ in sorted_nodes[:n_seeds]])
    
    return seed_nodes

def largest_communities_seeding(G, communities, seed_count):
    """
    Target only the largest communities
    """
    # Count nodes in each community
    community_sizes = {}
    for comm_id in set(communities.values()):
        community_sizes[comm_id] = sum(1 for c in communities.values() if c == comm_id)
    
    # Get top 3 largest communities
    largest_comms = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:3]
    largest_comm_ids = [comm_id for comm_id, _ in largest_comms]
    
    # Select nodes from largest communities only
    nodes_in_largest = [n for n, c in communities.items() if c in largest_comm_ids]
    
    # Using degree centrality to select most influential nodes
    subgraph = G.subgraph(nodes_in_largest)
    degree_centrality = nx.degree_centrality(subgraph)
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    
    return [node for node, _ in sorted_nodes[:seed_count]]

def community_bridges_seeding(G, communities, seed_count):
    """
    Target nodes that bridge between communities (high betweenness)
    """
    # Calculae betweenness centrality 
    if G.number_of_nodes() > 1000:
        betweenness = nx.betweenness_centrality(G, k=500)  
    else:
        betweenness = nx.betweenness_centrality(G)
    
    # Identify bridge nodes 
    bridge_scores = {}
    
    for node in G.nodes():
        # Skipping isolated nodes
        if G.degree(node) == 0:
            continue
            
        # Get communities of neighbors
        neighbor_comms = set()
        for neighbor in G.neighbors(node):
            if neighbor in communities:
                neighbor_comms.add(communities[neighbor])
        
        # Score based on number of different communities connected
        # Multiply by betweenness to prioritize nodes with high centrality
        bridge_scores[node] = len(neighbor_comms) * betweenness.get(node, 0)
    
    # Select nodes with highest bridge scores
    sorted_bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_bridges[:seed_count]]

class MovieDiffusionModel:
    """Simple movie diffusion model"""
    
    def __init__(self, G, user_preferences):
        """Initialize the model"""
        self.G = G
        self.user_preferences = user_preferences
        self.reset()
    
    def reset(self):
        """Reset the simulation state"""
        self.adopted = set()  # Users who have adopted
        self.exposed = set()  # Users who have been exposed
        self.adoption_timestamps = {}  # Dictionary mapping node -> time of adoption
        self.time_step = 0
        self.adoption_history = []
        self.exposure_history = []
    
    def run_diffusion(self, seed_nodes, movie_genres, 
                     preference_weight=0.6, social_weight=0.4, 
                     threshold_mean=0.5, threshold_std=0.1,
                     max_steps=30):
        """
        Running the diffusion simulation
        
        """
        # Reset simulation state
        self.reset()
        
        # Initialize with seed nodes
        self.adopted = set(seed_nodes)
        self.exposed = set(seed_nodes).union(
            {neighbor for node in seed_nodes for neighbor in self.G.neighbors(node)}
        )
        
        # Record seed node adoption timestamps
        for node in seed_nodes:
            self.adoption_timestamps[node] = 0
        
        # Record initial state
        self.adoption_history.append(len(self.adopted))
        self.exposure_history.append(len(self.exposed))
        
        # Run simulation steps
        for _ in range(max_steps):
            new_adopted = set()
            # For all nodes that have been exposed but haven't adopted yet
            for user_id in self.exposed - self.adopted:
                # Calculate preference match
                match_score = 0.5  
                if user_id in self.user_preferences:
                    genre_scores = [self.user_preferences[user_id].get(genre, 0) for genre in movie_genres]
                    if genre_scores:
                        match_score = sum(genre_scores) / len(genre_scores)
                
                # Calculate social influence
                neighbors = set(self.G.neighbors(user_id))
                if neighbors:
                    adopted_neighbors = neighbors.intersection(self.adopted)
                    influence_score = len(adopted_neighbors) / len(neighbors)
                else:
                    influence_score = 0
                
                # Combine factors with weights
                combined_score = (preference_weight * match_score) + (social_weight * influence_score)
                
                # Generate random threshold for this user
                threshold = np.random.normal(threshold_mean, threshold_std)
                threshold = max(0, min(1, threshold))  # Clamp to [0, 1]
                
                # Adoption decision
                if combined_score >= threshold:
                    new_adopted.add(user_id)
            
            # Update adoption set
            self.adopted.update(new_adopted)
            
            # Record adoption timestamps for new adopters
            current_step = self.time_step + 1
            for node in new_adopted:
                self.adoption_timestamps[node] = current_step
            
            # Expose neighbors of newly adopted nodes
            new_exposed = set()
            for user_id in new_adopted:
                new_exposed.update(self.G.neighbors(user_id))
            
            self.exposed.update(new_exposed)
            
            # Record history
            self.adoption_history.append(len(self.adopted))
            self.exposure_history.append(len(self.exposed))
            
            # Stop if no new adoptions
            if not new_adopted:
                break
            
            self.time_step += 1
        
        # Calculating results
        total_nodes = self.G.number_of_nodes()
        results = {
            'adoption_history': self.adoption_history,
            'exposure_history': self.exposure_history,
            'final_adoption_rate': len(self.adopted) / total_nodes,
            'final_exposure_rate': len(self.exposed) / total_nodes,
            'steps': self.time_step + 1,
            'seed_count': len(seed_nodes),
            'seed_nodes': list(seed_nodes),
            'adopted_nodes': list(self.adopted),
            'exposed_nodes': list(self.exposed),
            'adoption_timestamps': self.adoption_timestamps
        }
        
        return results
    
def detect_communities(G):
    """
    Detect communities in the network using the Louvain method
    
    """
    print("Detecting communities...")
    try:
        communities = community_louvain.best_partition(G)
        print(f"Detected {len(set(communities.values()))} communities")
        return communities
    except Exception as e:
        print(f"Error detecting communities: {e}")
        communities_list = list(nx.community.louvain_communities(G))
        communities = {}
        for i, community in enumerate(communities_list):
            for node in community:
                communities[node] = i
        print(f"Detected {len(communities_list)} communities")
        return communities

def identify_diffusion_path_edges(G, seed_nodes, adopted_nodes, simulation_result):
    """
    Identify edges that form the diffusion path between seeds, adopters, and exposed nodes

    """
    # Get exposed nodes
    exposed_nodes = set(simulation_result['exposed_nodes']) if 'exposed_nodes' in simulation_result else set()
    
    # Initializing different types of edges
    adoption_edges = []  # Edges that led to adoption (strong influence)
    exposure_edges = []  # Edges that only led to exposure (weak influence)
    
    # If we have adoption timestamps, use them to reconstruct the path
    if 'adoption_timestamps' in simulation_result:
        timestamps = simulation_result['adoption_timestamps']
        

        for node in adopted_nodes - seed_nodes:
            if node not in timestamps:
                continue
            node_time = timestamps[node]
            earlier_adopters = [
                n for n in G.neighbors(node) 
                if n in timestamps and timestamps[n] < node_time
            ]
            
            if earlier_adopters:
                # Connect to the most recent adopter among neighbors
                influencer = max(earlier_adopters, key=lambda x: timestamps[x])
                adoption_edges.append((influencer, node))
        
        # For exposed but not adopted nodes, find potential influencers
        for node in exposed_nodes - adopted_nodes:
            potential_influencers = [
                n for n in G.neighbors(node)
                if n in adopted_nodes or n in seed_nodes
            ]
            
            if potential_influencers:
                timestamped_influencers = [n for n in potential_influencers if n in timestamps]
                if timestamped_influencers:
                    influencer = max(timestamped_influencers, key=lambda x: timestamps[x])
                    exposure_edges.append((influencer, node))
                else:
                    exposure_edges.append((potential_influencers[0], node))
    
    # If no timestamps or for additional coverage
    else:
        # For each seed node, connect to its exposed neighbors
        for seed in seed_nodes:
            for neighbor in G.neighbors(seed):
                if neighbor in adopted_nodes - seed_nodes:
                    adoption_edges.append((seed, neighbor))
                elif neighbor in exposed_nodes - adopted_nodes:
                    exposure_edges.append((seed, neighbor))
        
        # For each non-seed adopter, connect to its exposed neighbors
        for adopter in adopted_nodes - seed_nodes:
            for neighbor in G.neighbors(adopter):
                if neighbor in adopted_nodes and (neighbor, adopter) not in adoption_edges:
                    # Both nodes adopted, connect from earlier to later
                    if neighbor in seed_nodes or neighbor < adopter:
                        adoption_edges.append((neighbor, adopter))
                    else:
                        adoption_edges.append((adopter, neighbor))
                elif neighbor in exposed_nodes - adopted_nodes:
                    # Neighbor was exposed but didn't adopt
                    exposure_edges.append((adopter, neighbor))
    
    # Return both types of edges
    return adoption_edges, exposure_edges

def visualize_diffusion_path_with_communities(G, simulation_result, communities, pos=None, output_file="diffusion_path_community.png"):
    """
    Visualize diffusion path with colored nodes showing different statuses and communities

    """
    plt.figure(figsize=(12, 10))
    
    # Get node sets from simulation results
    seed_nodes = set(simulation_result['seed_nodes'])
    adopted_nodes = set(simulation_result['adopted_nodes'])
    exposed_nodes = set(simulation_result['exposed_nodes']) if 'exposed_nodes' in simulation_result else set()
    
    # Calculate derived sets
    influenced_not_adopted = exposed_nodes - adopted_nodes
    unaffected_nodes = set(G.nodes()) - exposed_nodes - adopted_nodes - seed_nodes
    
    # Sample nodes if network is large
    max_nodes_to_show = 1000
    if len(G) > max_nodes_to_show:
        # Keep all important nodes
        important_nodes = seed_nodes.union(adopted_nodes)
        important_nodes = important_nodes.union(influenced_not_adopted)
        
        # Sample from unaffected nodes
        num_remaining = max_nodes_to_show - len(important_nodes)
        if num_remaining > 0 and len(unaffected_nodes) > num_remaining:
            unaffected_sample = set(random.sample(list(unaffected_nodes), num_remaining))
        else:
            unaffected_sample = unaffected_nodes
        
        # Create subgraph
        nodes_to_show = important_nodes.union(unaffected_sample)
        G_sub = G.subgraph(nodes_to_show)
    else:
        G_sub = G
        unaffected_sample = unaffected_nodes
    
    # Use provided positions or calculate new ones
    if pos is None:
        pos = nx.spring_layout(G_sub, seed=42)
    else:
        # Filter positions to only include nodes in G_sub
        pos = {n: p for n, p in pos.items() if n in G_sub}
    
    # Identify diffusion path edges
    adoption_edges, exposure_edges = identify_diffusion_path_edges(G_sub, seed_nodes, adopted_nodes, simulation_result)
    
    # Get regular edges (edges not in the diffusion path)
    all_diffusion_edges = []
    for edge in adoption_edges:
        all_diffusion_edges.append(edge)
        all_diffusion_edges.append((edge[1], edge[0]))  # Add reverse edge
    
    for edge in exposure_edges:
        all_diffusion_edges.append(edge)
        all_diffusion_edges.append((edge[1], edge[0]))  # Add reverse edge 
    
    regular_edges = [(u, v) for u, v in G_sub.edges() if (u, v) not in all_diffusion_edges]
    
    # Drawing edges first (so the nodes appear on top)
    # Regular edges in light gray
    nx.draw_networkx_edges(G_sub, pos, 
                          edgelist=regular_edges,
                          edge_color='lightgray', alpha=0.1, width=0.5)
    
    # Exposure diffusion path edges in orange 
    if exposure_edges:
        nx.draw_networkx_edges(G_sub, pos, 
                              edgelist=exposure_edges,
                              edge_color='orange', alpha=0.5, width=1.5)
    
    # Adoption diffusion path edges in bright yellow
    if adoption_edges:
        nx.draw_networkx_edges(G_sub, pos, 
                              edgelist=adoption_edges,
                              edge_color='yellow', alpha=0.8, width=2.0)
    
    # Get community for each node
    node_communities = {n: communities.get(n, -1) for n in G_sub.nodes()}
    
    # Get unique community IDs
    unique_communities = sorted(set(node_communities.values()))
    
    # Creating a colormap for communities (excluding -1, which is for nodes without a community)
    cmap = plt.cm.get_cmap('tab20', len(unique_communities))
    community_colors = {comm_id: cmap(i) for i, comm_id in enumerate(unique_communities)}
    community_colors[-1] = (0.7, 0.7, 0.7, 1.0)  # Gray for nodes without community
    
    # Defining border colors and widths for different node types
    status_borders = {
        'seed': {'color': 'red', 'width': 2.0},
        'adopted': {'color': 'green', 'width': 1.5},
        'exposed': {'color': 'blue', 'width': 1.0},
        'unaffected': {'color': None, 'width': 0.0}
    }
    
    # Drawing nodes by adoption status, colored by community
    for status, node_set in [
        ('unaffected', unaffected_sample.intersection(G_sub.nodes())),
        ('exposed', influenced_not_adopted.intersection(G_sub.nodes())),
        ('adopted', (adopted_nodes - seed_nodes).intersection(G_sub.nodes())),
        ('seed', seed_nodes.intersection(G_sub.nodes()))
    ]:
        # Skip if no nodes of this stats
        if not node_set:
            continue
            
        # Group nodes by community
        nodes_by_community = {}
        for node in node_set:
            comm_id = node_communities[node]
            if comm_id not in nodes_by_community:
                nodes_by_community[comm_id] = []
            nodes_by_community[comm_id].append(node)
        
        # Draw nodes for each community
        for comm_id, nodes in nodes_by_community.items():
            # Base color is the community color
            base_color = community_colors[comm_id]
            
            # Node size and alpha depend on status
            if status == 'unaffected':
                node_size = 20
                node_alpha = 0.3
            elif status == 'exposed':
                node_size = 30
                node_alpha = 0.7
            elif status == 'adopted':
                node_size = 50
                node_alpha = 0.8
            else:  # seed
                node_size = 70
                node_alpha = 1.0
            
            # Get border properties
            border_color = status_borders[status]['color']
            border_width = status_borders[status]['width']
            
            # Draw nodes
            nx.draw_networkx_nodes(G_sub, pos, 
                                 nodelist=nodes, 
                                 node_color=[base_color],
                                 node_size=node_size, 
                                 alpha=node_alpha,
                                 edgecolors=border_color,
                                 linewidths=border_width)
    
    # Create legend for community colors and status
    comm_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=community_colors[comm_id], 
              markersize=10, label=f'Community {comm_id}')
        for comm_id in sorted(community_colors.keys()) if comm_id != -1
    ]
    
    # Enhanced legend with colored borders matching node types
    status_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.5, 0.5, 0.5, 1.0), 
              markersize=10, label='Seed Nodes', markeredgecolor='red', markeredgewidth=2.0),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.5, 0.5, 0.5, 1.0), 
              markersize=10, label='Adopted', markeredgecolor='green', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.5, 0.5, 0.5, 0.7), 
              markersize=10, label='Exposed Not Adopted', markeredgecolor='blue', markeredgewidth=1.0),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.5, 0.5, 0.5, 0.3), 
              markersize=10, label='Unaffected'),
        Line2D([0], [0], marker='_', color='yellow', linewidth=2, markersize=0, label='Adoption Path'),
        Line2D([0], [0], marker='_', color='orange', linewidth=2, markersize=0, label='Exposure Path')
    ]
    

    plt.legend(handles=status_legend_elements, loc='upper right', title='Status')
    
    # Separate legend for communities
    comm_legend = plt.legend(handles=comm_legend_elements, loc='lower left', title='Communities')
    plt.gca().add_artist(comm_legend)
    

    plt.title(f"Diffusion Result with {len(seed_nodes)} seed nodes ({len(adopted_nodes)} adopted)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved community diffusion path visualization to {output_file}")
    
    return pos  

def save_network_with_diffusion_data(G, simulation_result, output_file):
    """
    Save the network with diffusion data attached as node attributes
    
    """
    G_with_data = G.copy()
    
    # Extracting node sets from simulation results
    seed_nodes = set(simulation_result['seed_nodes'])
    adopted_nodes = set(simulation_result['adopted_nodes'])
    exposed_nodes = set(simulation_result['exposed_nodes']) if 'exposed_nodes' in simulation_result else set()
    

    for node in G_with_data.nodes():
        G_with_data.nodes[node]['is_seed'] = node in seed_nodes
        G_with_data.nodes[node]['adopted'] = node in adopted_nodes
        G_with_data.nodes[node]['exposed'] = node in exposed_nodes
        
        # Adding adoption timestamp if available
        if 'adoption_timestamps' in simulation_result and node in simulation_result['adoption_timestamps']:
            G_with_data.nodes[node]['adoption_timestamp'] = simulation_result['adoption_timestamps'][node]
        else:
            G_with_data.nodes[node]['adoption_timestamp'] = -1  # -1 indicates no adoption
    
    # Saving the graph 
    nx.write_graphml(G_with_data, output_file)
    print(f"Saved network with diffusion data to {output_file}")

def run_diffusion_analysis(data_dir, output_dir, movie_genres, seed_count=100, num_runs=5):
    """
    Run diffusion analysis using different seeding strategies

    """
    # Output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Loading network data
    G, user_preferences = load_network_data(data_dir)

    communities = detect_communities(G)

    nx.set_node_attributes(G, communities, 'community')
    
    
    actual_seed_count = seed_count
    
    # Initializing diffusion model
    diffusion_model = MovieDiffusionModel(G, user_preferences)
    
    # Strategies to test
    strategies = [
        'random', 
        'degree', 
        'preference',
        'community_balanced',
        'community_proportional',
        'largest_communities',
        'community_bridges'
    ]    
    # Run diffusion for each strategy
    all_results = {}
    
    for strategy in strategies:
        print(f"Testing {strategy} strategy...")
        strategy_results = []
        
        # Run multiple simulations
        for run in range(num_runs):
            # Select seed nodes
            seed_nodes = select_seed_nodes(G, user_preferences, movie_genres, 
                                          strategy=strategy, seed_count=actual_seed_count, communities=communities)
            
            # Run diffusion simulation
            result = diffusion_model.run_diffusion(seed_nodes, movie_genres)
            strategy_results.append(result)
            
            # Saving the network with diffusion data for this run
            network_file = os.path.join(output_dir, f"network_{strategy}_run{run}.graphml")
            save_network_with_diffusion_data(G, result, network_file)
        
        # Aggregating results
        adoption_histories = [r['adoption_history'] for r in strategy_results]
        max_length = max(len(h) for h in adoption_histories)
        
        # Padding
        padded_histories = []
        for hist in adoption_histories:
            padded = hist + [hist[-1]] * (max_length - len(hist))
            padded_histories.append(padded)
        
        mean_adoption = np.mean(padded_histories, axis=0)
        std_adoption = np.std(padded_histories, axis=0)
        
        # Calculating average final rate
        final_rates = [r['final_adoption_rate'] for r in strategy_results]
        mean_final_rate = np.mean(final_rates)
        
        all_results[strategy] = {
            'mean_adoption_history': mean_adoption.tolist(),
            'std_adoption_history': std_adoption.tolist(),
            'mean_final_rate': mean_final_rate,
            'individual_results': strategy_results
        }
    
    # Creating adoption curve visualization
    plt.figure(figsize=(10, 6))
    
    for strategy, results in all_results.items():
        mean_history = results['mean_adoption_history']
        std_history = results['std_adoption_history']
        
        x = range(len(mean_history))
        plt.plot(x, mean_history, label=f"{strategy} (Final: {results['mean_final_rate']:.3f})")
        plt.fill_between(x, 
                        [max(0, m - s) for m, s in zip(mean_history, std_history)],
                        [min(1, m + s) for m, s in zip(mean_history, std_history)],
                        alpha=0.2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Adoption Rate')
    plt.title('Movie Diffusion by Seeding Strategy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "movie_diffusion_by_strategy.png"), dpi=300)
    plt.close()
    
    # Creating bar chart of final adoption rates
    plt.figure(figsize=(8, 5))
    
    sorted_strategies = sorted(all_results.items(), key=lambda x: x[1]['mean_final_rate'], reverse=True)
    strategies = [s for s, _ in sorted_strategies]
    rates = [r['mean_final_rate'] for _, r in sorted_strategies]
    
    plt.bar(strategies, rates)
    plt.ylabel('Final Adoption Rate')
    plt.title('Final Adoption Rates by Strategy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_adoption_rates_by_strategy.png"), dpi=300)
    plt.close()
    
    # Saving results to CSV
    summary_data = []
    for strategy, results in all_results.items():
        summary_data.append({
            'Strategy': strategy,
            'Final_Adoption_Rate': results['mean_final_rate'],
            'Average_Steps': np.mean([r['steps'] for r in results['individual_results']])
        })
    
    pd.DataFrame(summary_data).to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    
    # Creating community-specific visualizations
    analyze_community_diffusion(G, communities, all_results, output_dir)
    
    print(f"Diffusion analysis completed. Results saved to {output_dir}/")
    return all_results

def analyze_community_diffusion(G, communities, diffusion_results, output_dir):
    """
    Analyze how diffusion patterns vary across communities for different strategies

    """

    output_file = os.path.join(output_dir, "community_diffusion_analysis.csv")
    
    # Get unique community IDs
    community_ids = sorted(set(communities.values()))
    
    # Counting nodes in each community
    community_sizes = {}
    for comm_id in community_ids:
        community_sizes[comm_id] = sum(1 for c in communities.values() if c == comm_id)
    
    # Getting top 5 largest communities
    top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
    top_comm_ids = [comm_id for comm_id, _ in top_communities]
    
    # For each strategy, analyzing adoption by community
    community_results = []
    
    for strategy, results in diffusion_results.items():
        if 'individual_results' not in results or not results['individual_results']:
            continue
            
        # Get the first simulation result for this strategy
        sim_result = results['individual_results'][0]
        
        # Get adoption sets
        seed_nodes = set(sim_result['seed_nodes'])
        adopted_nodes = set(sim_result['adopted_nodes'])
        
        # Calculate adoption by community
        for comm_id in community_ids:
            # Get nodes in this community
            comm_nodes = [n for n, c in communities.items() if c == comm_id]
            
            # Calculate metrics
            seeds_in_comm = sum(1 for n in comm_nodes if n in seed_nodes)
            adopters_in_comm = sum(1 for n in comm_nodes if n in adopted_nodes and n not in seed_nodes)
            total_adopted = seeds_in_comm + adopters_in_comm
            
            # Calculate rates
            comm_size = community_sizes[comm_id]
            seed_rate = seeds_in_comm / comm_size if comm_size > 0 else 0
            adoption_rate = adopters_in_comm / comm_size if comm_size > 0 else 0
            total_rate = total_adopted / comm_size if comm_size > 0 else 0
            
            # Adding to results
            community_results.append({
                'strategy': strategy,
                'community_id': comm_id,
                'community_size': comm_size,
                'seeds': seeds_in_comm,
                'adopters': adopters_in_comm,
                'total_adopted': total_adopted,
                'seed_rate': seed_rate,
                'adoption_rate': adoption_rate,
                'total_rate': total_rate
            })
    
    # Saving to CSV
    pd.DataFrame(community_results).to_csv(output_file, index=False)
    print(f"Saved community diffusion analysis to {output_file}")
    
    # Creating visualizations
    
    # 1. Community adoption rates by strategy 
    plt.figure(figsize=(12, 8))
    
    # Convert data to DataFrame for easier plotting
    df = pd.DataFrame(community_results)
    
    # Filter for top communities
    df_top = df[df['community_id'].isin(top_comm_ids)]
    
    # Creating grouped bar chart
    sns.barplot(x='community_id', y='total_rate', hue='strategy', data=df_top)
    plt.title('Adoption Rate by Community and Strategy (Top 5 Largest Communities)')
    plt.xlabel('Community ID')
    plt.ylabel('Adoption Rate')
    plt.legend(title='Strategy')
    plt.savefig(os.path.join(output_dir, "community_adoption_by_strategy.png"), dpi=300)
    plt.close()
    
    # 2. Time-series adoption plots for key strategies
    key_strategies = ['degree', 'preference', 'community_bridges', 'community_proportional', 'random']
    
    for strategy in key_strategies:
        if 'individual_results' not in diffusion_results[strategy] or not diffusion_results[strategy]['individual_results']:
            continue
        
        # Get first simulation result
        sim_result = diffusion_results[strategy]['individual_results'][0]
        
        # Using adoption timestamps to track community adoption over time
        if 'adoption_timestamps' in sim_result:
            timestamps = sim_result['adoption_timestamps']
            
            # Counting adoptions per community over time
            max_time = max(timestamps.values()) if timestamps else 0
            community_adoption = {comm_id: [0] * (max_time + 1) for comm_id in top_comm_ids}
            
            # Counting nodes in each community
            for node, time in timestamps.items():
                if node in communities and communities[node] in top_comm_ids:
                    comm_id = communities[node]
                    for t in range(time, max_time + 1):
                        community_adoption[comm_id][t] += 1
            
            # Convert to rates
            for comm_id in top_comm_ids:
                size = community_sizes[comm_id]
                if size > 0:  # Avoiding division by zero
                    community_adoption[comm_id] = [n / size for n in community_adoption[comm_id]]
            
            # Plot time series
            plt.figure(figsize=(10, 6))
            for comm_id in top_comm_ids:
                plt.plot(range(max_time + 1), community_adoption[comm_id], 
                         label=f"Community {comm_id} (size: {community_sizes[comm_id]})")
            
            plt.xlabel('Time Step')
            plt.ylabel('Adoption Rate')
            plt.title(f'Adoption Rate Over Time by Community ({strategy} strategy)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"adoption_time_by_community_{strategy}.png"), dpi=300)
            plt.close()
    
    # 3. Seed Rate vs Adoption Rate scatter plots
    for comm_id in top_comm_ids:
        plt.figure(figsize=(10, 6))
        
        # Filter data for this community
        comm_data = df[df['community_id'] == comm_id]
        
        # Creating scatter plot
        for _, row in comm_data.iterrows():
            plt.scatter(row['seed_rate'], row['total_rate'], s=100, label=row['strategy'])
            plt.annotate(row['strategy'], (row['seed_rate'], row['total_rate']), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Seed Rate (fraction of community)')
        plt.ylabel('Final Adoption Rate')
        plt.title(f'Seed Rate vs Final Adoption Rate for Community {comm_id}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"seed_vs_adoption_community_{comm_id}.png"), dpi=300)
        plt.close()
    
    # 4. Strategy effectiveness by community size
    plt.figure(figsize=(12, 8))
    
    # Group data by strategy and community
    size_data = []
    
    for strategy, results in diffusion_results.items():
        if 'individual_results' not in results or not results['individual_results']:
            continue
        
        sim_result = results['individual_results'][0]
        adopted_nodes = set(sim_result['adopted_nodes'])
        
        for comm_id, size in community_sizes.items():
            if size < 10:  # Skipping very small communities
                continue
            
            comm_nodes = [n for n, c in communities.items() if c == comm_id]
            adopters = sum(1 for n in comm_nodes if n in adopted_nodes)
            rate = adopters / size if size > 0 else 0
            
            size_data.append({
                'strategy': strategy,
                'community_size': size,
                'adoption_rate': rate,
                'community_id': comm_id
            })
    
    # Convert to DataFrame
    size_df = pd.DataFrame(size_data)
    
    # Creating scatter plot
    for strategy in diffusion_results.keys():
        strategy_data = size_df[size_df['strategy'] == strategy]
        plt.scatter(strategy_data['community_size'], strategy_data['adoption_rate'], 
                   s=50, alpha=0.7, label=strategy)
    
    plt.xscale('log')  # Log scale for community size
    plt.xlabel('Community Size (log scale)')
    plt.ylabel('Adoption Rate')
    plt.title('Strategy Effectiveness by Community Size')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_effectiveness_by_community_size.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run movie diffusion simulation")
    parser.add_argument("--data_dir", required=True, help="Directory containing network data")
    parser.add_argument("--output_dir", default="diffusion_results", help="Directory to save results")
    parser.add_argument("--genres", nargs='+', default=["Action", "Comedy", "Drama"], 
                     help="List of movie genres")
    parser.add_argument("--seed_count", type=int, default=100, help="Number of seed nodes")
    parser.add_argument("--runs", type=int, default=5, help="Number of simulation runs per strategy")
    
    args = parser.parse_args()
    
    run_diffusion_analysis(
        args.data_dir,
        args.output_dir,
        args.genres,
        seed_count=args.seed_count,
        num_runs=args.runs
    )