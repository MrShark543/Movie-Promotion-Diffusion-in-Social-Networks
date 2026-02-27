import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import community as community_louvain  
import json

def analyze_network_structure(network_file, output_dir):
    """
    Analyze the structure of a network without considering diffusion results

    """
    print(f"Loading network from {network_file}...")
    G = nx.read_edgelist(network_file, nodetype=int)
    print(f"Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    

    os.makedirs(output_dir, exist_ok=True)
    
    # Calculating basic network metrics
    print("Calculating network metrics...")
    
    # Basic statistics
    avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    density = nx.density(G)
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    max_degree = max(degrees)
    
    # Clustering coefficient
    avg_clustering = nx.average_clustering(G)
    
    # Path length 
    if G.number_of_nodes() > 1000:
        # Sampling 1000 random pairs
        import random
        sample_nodes = random.sample(list(G.nodes()), min(1000, G.number_of_nodes()))
        path_lengths = []
        for i, u in enumerate(sample_nodes):
            for v in sample_nodes[i+1:]:
                try:
                    path_length = nx.shortest_path_length(G, source=u, target=v)
                    path_lengths.append(path_length)
                except nx.NetworkXNoPath:
                    pass
        avg_path_length = np.mean(path_lengths) if path_lengths else float('nan')
    else:
        avg_path_length = nx.average_shortest_path_length(G)
    
    # Centrality metrics (for top 100 nodes in order to avoid performance issues)
    print("Calculating centrality metrics...")
    degree_centrality = nx.degree_centrality(G)
    
    # Sort by centrality and get top 100 nodes
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:100]
    top_node_ids = [n for n, _ in top_nodes]
    
    # Calculate betweenness only for top nodes
    betweenness = nx.betweenness_centrality(G, k=100)  
    

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G)
    except:
        print("  Warning: Eigenvector centrality calculation failed")
        eigenvector = {n: 0 for n in G.nodes()}
    
    # Create summary DataFrame for the top nodes
    top_nodes_df = pd.DataFrame({
        'node_id': top_node_ids,
        'degree': [G.degree(n) for n in top_node_ids],
        'degree_centrality': [degree_centrality[n] for n in top_node_ids],
        'betweenness_centrality': [betweenness.get(n, 0) for n in top_node_ids],
        'eigenvector_centrality': [eigenvector.get(n, 0) for n in top_node_ids],
        'clustering_coefficient': [nx.clustering(G, n) for n in top_node_ids]
    })
    
    # Saving top nodes data
    top_nodes_df.to_csv(os.path.join(output_dir, 'top_nodes.csv'), index=False)
    
    # Community detection
    print("Detecting communities...")
    communities = None
    try:
        partition = community_louvain.best_partition(G)
        communities = partition  
        
        # Counting nodes in each community
        community_sizes = pd.Series(partition).value_counts().sort_values(ascending=False)
        
        # Add community attribute to nodes
        nx.set_node_attributes(G, partition, 'community')
        
        # Calculate statistics by community
        communities_df = pd.DataFrame({
            'community': list(range(len(community_sizes))),
            'size': [community_sizes.get(i, 0) for i in range(len(community_sizes))],
            'fraction': [community_sizes.get(i, 0) / G.number_of_nodes() 
                       for i in range(len(community_sizes))]
        })
        
        # Calculate internal/external edge ratio for each community
        community_metrics = []
        
        for comm_id in range(len(community_sizes)):
            # Get nodes in this community
            comm_nodes = [n for n, c in partition.items() if c == comm_id]
            
            if not comm_nodes: 
                continue
                
            # Create subgraph of just this community
            subgraph = G.subgraph(comm_nodes)
            
            # Counting internal edges
            internal_edges = subgraph.number_of_edges()
            
            # Counting external edges
            external_edges = sum(G.degree(n) for n in comm_nodes) - 2 * internal_edges
            
            # Calculating metrics
            total_edges = internal_edges + (external_edges / 2)  
            internal_ratio = internal_edges / total_edges if total_edges > 0 else 0
            
            community_metrics.append({
                'community': comm_id,
                'nodes': len(comm_nodes),
                'internal_edges': internal_edges,
                'external_edges': external_edges / 2,  # Dividing by 2 to avoid double counting
                'internal_ratio': internal_ratio,
                'density': nx.density(subgraph)
            })
        
        # Create DataFrame and save
        community_metrics_df = pd.DataFrame(community_metrics)
        community_metrics_df.to_csv(os.path.join(output_dir, 'community_metrics.csv'), index=False)
        
        # Create summary of top communities
        top_communities = community_metrics_df.nlargest(10, 'nodes')
        
        # Visualize community sizes
        plt.figure(figsize=(10, 6))
        plt.bar(range(min(20, len(community_sizes))), 
               community_sizes.values[:20])
        plt.xlabel('Community ID')
        plt.ylabel('Number of Nodes')
        plt.title('Size of Top 20 Communities')
        plt.savefig(os.path.join(output_dir, 'community_sizes.png'), dpi=300)
        plt.close()
        
        # Visualize internal/external edge ratio
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='nodes', 
            y='internal_ratio', 
            size='nodes',
            sizes=(20, 500),
            data=community_metrics_df.nlargest(20, 'nodes')
        )
        plt.xlabel('Community Size (Nodes)')
        plt.ylabel('Internal Edge Ratio')
        plt.title('Community Cohesion vs Size')
        plt.savefig(os.path.join(output_dir, 'community_cohesion.png'), dpi=300)
        plt.close()
        
        # Analyze community structure 
        analyze_community_structure(G, partition, output_dir)
    
    except Exception as e:
        print(f"  Error in community detection: {e}")
    
    # Visualize degree distribution
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50, alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Degree Distribution')
    plt.savefig(os.path.join(output_dir, 'degree_distribution.png'), dpi=300)
    plt.close()
    
    # Degree distribution on log-log scale 
    degree_counts = pd.Series(degrees).value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    plt.loglog(degree_counts.index, degree_counts.values, 'o-')
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Count (log scale)')
    plt.title('Log-Log Degree Distribution')
    plt.savefig(os.path.join(output_dir, 'loglog_degree_distribution.png'), dpi=300)
    plt.close()
    
    # Saving overall network metrics
    network_metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'average_degree': avg_degree,
        'max_degree': max_degree,
        'density': density,
        'average_clustering': avg_clustering,
        'average_path_length': avg_path_length,
        'number_of_communities': len(community_sizes) if 'community_sizes' in locals() else 'N/A',
        'largest_community_size': community_sizes.iloc[0] if 'community_sizes' in locals() else 'N/A',
        'largest_community_fraction': community_sizes.iloc[0] / G.number_of_nodes() if 'community_sizes' in locals() else 'N/A',
    }
    
    pd.DataFrame([network_metrics]).to_csv(os.path.join(output_dir, 'network_metrics.csv'), index=False)
    
    # Create text summary
    with open(os.path.join(output_dir, 'network_summary.txt'), 'w') as f:
        f.write("# Network Structure Summary\n\n")
        f.write(f"Nodes: {G.number_of_nodes():,}\n")
        f.write(f"Edges: {G.number_of_edges():,}\n")
        f.write(f"Average Degree: {avg_degree:.2f}\n")
        f.write(f"Density: {density:.6f}\n")
        f.write(f"Average Clustering Coefficient: {avg_clustering:.4f}\n")
        f.write(f"Average Path Length: {avg_path_length:.2f}\n\n")
        
        if 'community_sizes' in locals():
            f.write(f"Number of Communities: {len(community_sizes)}\n")
            f.write(f"Largest Community Size: {community_sizes.iloc[0]:,} nodes ")
            f.write(f"({100*community_sizes.iloc[0]/G.number_of_nodes():.1f}% of network)\n\n")
        
        f.write("## Top 5 Nodes by Degree Centrality\n\n")
        for i, (node, centrality) in enumerate(sorted(degree_centrality.items(), 
                                                  key=lambda x: x[1], reverse=True)[:5]):
            f.write(f"{i+1}. Node {node}: degree={G.degree(node)}, ")
            f.write(f"centrality={centrality:.4f}\n")
        
        if 'top_communities' in locals() and not top_communities.empty:
            f.write("\n## Top 5 Communities\n\n")
            for i, row in top_communities.head(5).iterrows():
                f.write(f"{i+1}. Community {row['community']}: {row['nodes']:,} nodes, ")
                f.write(f"internal ratio={row['internal_ratio']:.2f}, ")
                f.write(f"density={row['density']:.4f}\n")
    
    print(f"Network structure analysis completed. Results saved to {output_dir}")
    return G, communities

def analyze_community_structure(G, communities, output_dir):
    """
    Analyze the structural properties of communities
    
    """
    print("Analyzing detailed community structure...")
    
 
    comm_dir = os.path.join(output_dir, "community_analysis")
    os.makedirs(comm_dir, exist_ok=True)
    
    # Get unique community IDs
    community_ids = sorted(set(communities.values()))
    

    community_stats = []
    for comm_id in community_ids:
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        
        # Skipping empty communities
        if not comm_nodes:
            continue
        
        # Creating subgraph for this community
        subgraph = G.subgraph(comm_nodes)
        
        # Calculating basic metrics
        size = len(comm_nodes)
        edges = subgraph.number_of_edges()
        density = nx.density(subgraph)
        
        # Counting internal vs external edges
        internal_edges = edges
        
        # Counting external edges 
        external_edges = 0
        for node in comm_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in comm_nodes:
                    external_edges += 1
        
        # Calculating homophily (ratio of internal to total edges)
        total_edges = internal_edges + external_edges
        homophily = internal_edges / total_edges if total_edges > 0 else 0
        
        # Calculating centrality metrics
        try:
            # Degree centrality
            degree_centrality = np.mean([G.degree(n) for n in comm_nodes])
            
            # Clustering coefficient
            clustering = nx.average_clustering(subgraph)
            
            # Distance metrics 
            if nx.is_connected(subgraph) and len(comm_nodes) > 1:
                diameter = nx.diameter(subgraph)
                avg_path_length = nx.average_shortest_path_length(subgraph)
            else:
                diameter = float('nan')
                avg_path_length = float('nan')
                
        except Exception as e:
            print(f"  Error calculating metrics for community {comm_id}: {e}")
            degree_centrality = clustering = diameter = avg_path_length = float('nan')
        
        # Identifying bridge nodes 
        bridge_nodes = []
        for node in comm_nodes:
            external_connections = 0
            for neighbor in G.neighbors(node):
                if communities.get(neighbor) != comm_id:
                    external_connections += 1
            
            if external_connections > 0:
                bridge_nodes.append((node, external_connections))
        
        # Sorting bridge nodes by external connections
        bridge_nodes.sort(key=lambda x: x[1], reverse=True)
        top_bridges = bridge_nodes[:min(5, len(bridge_nodes))]
        
        # Adding to community stats
        community_stats.append({
            'community_id': comm_id,
            'size': size,
            'internal_edges': internal_edges,
            'external_edges': external_edges,
            'density': density,
            'homophily': homophily,
            'avg_degree': degree_centrality,
            'clustering': clustering,
            'diameter': diameter,
            'avg_path_length': avg_path_length,
            'bridge_count': len(bridge_nodes),
            'top_bridges': [n for n, _ in top_bridges]
        })
    
    # Save community statistics
    # Converting top_bridges to string for CSV saving
    for stat in community_stats:
        stat['top_bridges'] = str(stat['top_bridges'])
    
    df = pd.DataFrame(community_stats)
    df.to_csv(os.path.join(comm_dir, 'community_structure.csv'), index=False)
    
    
    # Community size vs homophily
    plt.figure(figsize=(10, 6))
    plt.scatter(df['size'], df['homophily'], s=df['size']/5, alpha=0.7)
    plt.xlabel('Community Size (nodes)')
    plt.ylabel('Homophily (internal edge ratio)')
    plt.title('Community Size vs Homophily')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(comm_dir, 'community_size_vs_homophily.png'), dpi=300)
    plt.close()
    
    # Community metrics comparison
    metrics_to_plot = ['homophily', 'density', 'avg_degree', 'clustering', 'bridge_count']
    
    # Normalizing metrics for comparison
    normalized_df = df.copy()
    for metric in metrics_to_plot:
        if metric in normalized_df.columns:
            max_val = normalized_df[metric].max()
            if max_val > 0:
                normalized_df[metric] = normalized_df[metric] / max_val
    
    # Plotting top 10 largest communities
    top10_df = normalized_df.nlargest(10, 'size')
    
    plt.figure(figsize=(12, 8))
    for metric in metrics_to_plot:
        if metric in top10_df.columns:
            plt.plot(top10_df['community_id'], top10_df[metric], 'o-', label=metric)
    
    plt.xlabel('Community ID')
    plt.ylabel('Normalized Value')
    plt.title('Structural Properties of Top 10 Largest Communities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(top10_df['community_id'])
    plt.savefig(os.path.join(comm_dir, 'community_metrics_comparison.png'), dpi=300)
    plt.close()
    
    # Creating inter-community connection map
    intercomm_connections = {}
    
    for node, comm_id in communities.items():
        for neighbor in G.neighbors(node):
            neigh_comm = communities.get(neighbor)
            if neigh_comm != comm_id:
                key = (comm_id, neigh_comm)
                intercomm_connections[key] = intercomm_connections.get(key, 0) + 1
    
    # Creating connection matrix
    matrix_size = max(community_ids) + 1
    connection_matrix = np.zeros((matrix_size, matrix_size))
    
    for (source, target), count in intercomm_connections.items():
        if 0 <= source < matrix_size and 0 <= target < matrix_size:
            connection_matrix[source, target] = count
    
    # Visualizing connection matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(connection_matrix, cmap='viridis', 
               xticklabels=range(matrix_size),
               yticklabels=range(matrix_size))
    plt.xlabel('Target Community')
    plt.ylabel('Source Community')
    plt.title('Inter-Community Connections')
    plt.savefig(os.path.join(comm_dir, 'intercomm_connections.png'), dpi=300)
    plt.close()
    
    # Creating bridge node analysis
    bridge_analysis = {}
    for stat in community_stats:
        comm_id = stat['community_id']
        bridge_analysis[comm_id] = {
            'community_size': stat['size'],
            'homophily': stat['homophily'],
            'bridge_count': stat['bridge_count'],
            'bridge_fraction': stat['bridge_count'] / stat['size'] if stat['size'] > 0 else 0
        }
    
    # Visualizing bridge fraction vs homophily
    bridge_df = pd.DataFrame.from_dict(bridge_analysis, orient='index')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(bridge_df['homophily'], bridge_df['bridge_fraction'], 
               s=bridge_df['community_size']/5, alpha=0.7)
    plt.xlabel('Community Homophily')
    plt.ylabel('Bridge Node Fraction')
    plt.title('Community Homophily vs Bridge Node Fraction')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(comm_dir, 'homophily_vs_bridge_fraction.png'), dpi=300)
    plt.close()
    
    # Saving bridge analysis
    bridge_df.reset_index(names=['community_id']).to_csv(
        os.path.join(comm_dir, 'bridge_analysis.csv'), index=False)
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze network structure")
    parser.add_argument("--network_file", required=True, help="Path to network file (edgelist format)")
    parser.add_argument("--output_dir", default="network_analysis", help="Directory to save results")
    
    args = parser.parse_args()
    
    analyze_network_structure(args.network_file, args.output_dir)