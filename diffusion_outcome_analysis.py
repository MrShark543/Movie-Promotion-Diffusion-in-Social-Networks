import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_diffusion_results(results_dir, network_file, output_dir):
    """
    Analyzing diffusion results from simulation
    
    """
    print(f"Analyzing diffusion results from {results_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Loading the original network
    G = nx.read_edgelist(network_file, nodetype=int)
    
    # Loading simulation results
    results_summary = pd.read_csv(os.path.join(results_dir, "results_summary.csv"))
    
    # Creating summary visualization - Final adoption rate by strategy
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Strategy', y='Final_Adoption_Rate', data=results_summary)
    plt.title('Final Adoption Rate by Strategy')
    plt.ylabel('Adoption Rate')
    plt.xticks(rotation=45)
    plt.tight_layout() 
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300)
    plt.close()
    
    strategies = results_summary['Strategy'].tolist()
    
    # Dictionary to store community membership of each node
    communities = {}
    
    # Extracting communities 
    all_graphml_files = []
    for strategy in strategies:
        graphml_files = [f for f in os.listdir(results_dir) 
                         if f.startswith(f"network_{strategy}_run") and f.endswith(".graphml")]
        all_graphml_files.extend([os.path.join(results_dir, f) for f in graphml_files])
    
    if all_graphml_files:
        # Using the first file to extract communities
        G_with_comm = nx.read_graphml(all_graphml_files[0])
        for node in G_with_comm.nodes():
            if 'community' in G_with_comm.nodes[node]:
                communities[int(node)] = int(G_with_comm.nodes[node]['community'])
    
    # Analyzing community-based diffusion patterns
    if communities:
        analyze_community_diffusion_patterns(G, communities, results_dir, strategies, output_dir)
    
    # Generating information
    with open(os.path.join(output_dir, 'diffusion_summary.txt'), 'w') as f:
        f.write("# Diffusion Analysis Summary\n\n")
        
        f.write("## Strategy Effectiveness\n\n")
        # Sorting strategies by effectiveness
        sorted_strategies = results_summary.sort_values('Final_Adoption_Rate', ascending=False)
        
        for i, row in sorted_strategies.iterrows():
            f.write(f"{i+1}. {row['Strategy']}: {row['Final_Adoption_Rate']:.4f} adoption rate ")
            f.write(f"(avg {row['Average_Steps']:.1f} steps)\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Comparing best vs worst strategy
        best = sorted_strategies.iloc[0]
        worst = sorted_strategies.iloc[-1]
        difference = best['Final_Adoption_Rate'] - worst['Final_Adoption_Rate']
        percentage = (difference / worst['Final_Adoption_Rate']) * 100
        
        f.write(f"- The best strategy ({best['Strategy']}) outperformed the worst ")
        f.write(f"({worst['Strategy']}) by {percentage:.1f}%\n")
        
        # Adding community-based insights if available
        if communities:
            f.write("\n## Community Effects\n\n")
            
            # Adding insights about community-targeted strategies
            community_strategies = [s for s in strategies if 'community' in s]
            if community_strategies:
                f.write("### Community-Targeted Strategies\n\n")
                for strategy in community_strategies:
                    if strategy in sorted_strategies['Strategy'].values:
                        rank = sorted_strategies[sorted_strategies['Strategy'] == strategy].index[0] + 1
                        rate = sorted_strategies[sorted_strategies['Strategy'] == strategy]['Final_Adoption_Rate'].values[0]
                        f.write(f"- {strategy}: ranked #{rank} with {rate:.4f} adoption rate\n")
                
                # Comparing with non-community strategies
                best_community = max([(s, sorted_strategies[sorted_strategies['Strategy'] == s]['Final_Adoption_Rate'].values[0]) 
                                     for s in community_strategies], key=lambda x: x[1])
                
                non_community = [s for s in strategies if 'community' not in s]
                best_non_community = max([(s, sorted_strategies[sorted_strategies['Strategy'] == s]['Final_Adoption_Rate'].values[0]) 
                                         for s in non_community], key=lambda x: x[1])
                
                f.write(f"\nBest community strategy ({best_community[0]}) vs. ")
                f.write(f"best non-community strategy ({best_non_community[0]}): ")
                
                diff = (best_community[1] - best_non_community[1]) / best_non_community[1] * 100
                if diff > 0:
                    f.write(f"{abs(diff):.1f}% improvement\n")
                else:
                    f.write(f"{abs(diff):.1f}% worse performance\n")
    
    print(f"Diffusion analysis completed. Results saved to {output_dir}")

def analyze_community_diffusion_patterns(G, communities, results_dir, strategies, output_dir):
    """
    Analyze detailed diffusion patterns across communities
    
    """
    print("Analyzing detailed community diffusion patterns...")
    
    # Creating output directory for detailed analysis
    detail_dir = os.path.join(output_dir, "community_analysis")
    os.makedirs(detail_dir, exist_ok=True)
    
    # Getting unique community IDs
    community_ids = sorted(set(communities.values()))
    
    # Counting nodes in each community
    community_sizes = {}
    for comm_id in community_ids:
        community_sizes[comm_id] = sum(1 for n, c in communities.items() if c == comm_id)
    
    # Analyzing diffusion patterns for each strategy
    strategy_community_data = {}
    
    for strategy in strategies:
        graphml_files = [f for f in os.listdir(results_dir) 
                       if f.startswith(f"network_{strategy}_run") and f.endswith(".graphml")]
        
        if not graphml_files:
            print(f"  No GraphML file found for strategy: {strategy}")
            continue
            
        graphml_file = os.path.join(results_dir, graphml_files[0])
        
        try:
            # Loading the network with diffusion results
            G_diff = nx.read_graphml(graphml_file)
            
            # Getting diffusion data
            seed_nodes = set()
            adopted_nodes = set()
            timestamps = {}
            
            for node in G_diff.nodes():
                if G_diff.nodes[node]['is_seed'] == 'True':
                    seed_nodes.add(int(node))
                if G_diff.nodes[node]['adopted'] == 'True':
                    adopted_nodes.add(int(node))
                
                timestamp = int(G_diff.nodes[node]['adoption_timestamp'])
                if timestamp >= 0:  # -1 indicates no adoption
                    timestamps[int(node)] = timestamp
            
            # Analyzing seed distribution across communities
            seed_distribution = {}
            for comm_id in community_ids:
                comm_nodes = [n for n, c in communities.items() if c == comm_id]
                seed_distribution[comm_id] = sum(1 for n in comm_nodes if n in seed_nodes)
            
            # Analyzing adoption by time step for each community
            community_adoption_over_time = {}
            max_time = max(timestamps.values()) if timestamps else 0
            
            for comm_id in community_ids:
                comm_nodes = [n for n, c in communities.items() if c == comm_id]
                
                # Initializing time series for this community
                adoption_series = [0] * (max_time + 1)
                
                # Counting adoptions at each time step
                for node in comm_nodes:
                    if node in timestamps:
                        time_step = timestamps[node]
                        adoption_series[time_step] += 1
                
                # Convert to cumulative
                cumulative_series = []
                cumulative = 0
                for adoptions in adoption_series:
                    cumulative += adoptions
                    cumulative_series.append(cumulative)
                
                # Normalizing by community size
                normalized_series = [count / len(comm_nodes) for count in cumulative_series]
                
                community_adoption_over_time[comm_id] = normalized_series
            
            # Storing all data for this strategy
            strategy_community_data[strategy] = {
                'seed_distribution': seed_distribution,
                'adoption_over_time': community_adoption_over_time
            }
        
        except Exception as e:
            print(f"  Error analyzing {strategy} for community diffusion: {e}")
    
    # Creating comparative visualizations
    
    # 1. Adoption curves for top 5 communities by strategy
    top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
    top_comm_ids = [comm_id for comm_id, _ in top_communities]
    key_strategies = ['degree', 'preference', 'community_bridges', 'community_proportional', 'random']
    for strategy in strategies:
        if strategy not in key_strategies and strategy in strategy_community_data:
            key_strategies.append(strategy)
    
    for strategy in key_strategies:
        if strategy in strategy_community_data:
            data = strategy_community_data[strategy]
            plt.figure(figsize=(10, 6))
            
            for comm_id in top_comm_ids:
                adoption_series = data['adoption_over_time'].get(comm_id, [])
                if adoption_series:
                    plt.plot(range(len(adoption_series)), adoption_series, 
                           label=f'Community {comm_id} (size: {community_sizes[comm_id]})')
            
            plt.xlabel('Time Step')
            plt.ylabel('Adoption Rate')
            plt.title(f'Adoption Rate Over Time by Community ({strategy} strategy)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(detail_dir, f'adoption_time_by_community_{strategy}.png'), dpi=300)
            plt.close()
    
    # 2. Comparing seed distribution vs final adoption by community 
    adoption_by_community = {}
    
    for strategy, data in strategy_community_data.items():
        for comm_id in community_ids:
            # Getting final adoption rate
            adoption_series = data['adoption_over_time'].get(comm_id, [])
            final_rate = adoption_series[-1] if adoption_series else 0
            
            # Getting seed rate
            seed_count = data['seed_distribution'].get(comm_id, 0)
            seed_rate = seed_count / community_sizes[comm_id] if community_sizes[comm_id] > 0 else 0
            
            # Storing data
            if comm_id not in adoption_by_community:
                adoption_by_community[comm_id] = []
            
            adoption_by_community[comm_id].append({
                'strategy': strategy,
                'seed_rate': seed_rate,
                'final_rate': final_rate,
                'seed_count': seed_count,
                'community_size': community_sizes[comm_id]
            })
    
    # Creating comparison visualization for top 5 communities
    for comm_id in top_comm_ids:
        if comm_id in adoption_by_community:
            plt.figure(figsize=(10, 6))
            
            df = pd.DataFrame(adoption_by_community[comm_id])
            
            for i, row in df.iterrows():
                plt.scatter(row['seed_rate'], row['final_rate'], s=100, label=row['strategy'])
                plt.annotate(row['strategy'], (row['seed_rate'], row['final_rate']), 
                           xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel('Seed Rate (fraction of community)')
            plt.ylabel('Final Adoption Rate')
            plt.title(f'Seed Rate vs Final Adoption Rate for Community {comm_id}')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(detail_dir, f'seed_vs_adoption_comm{comm_id}.png'), dpi=300)
            plt.close()
    
    # 3. Strategy effectiveness by community size
    plt.figure(figsize=(12, 8))
    
    # Creating DataFrame with community sizes and adoption rates by strategy
    comp_data = []
    
    for comm_id in community_ids:
        comm_size = community_sizes[comm_id]
        
        for strategy, data in strategy_community_data.items():
            adoption_series = data['adoption_over_time'].get(comm_id, [])
            final_rate = adoption_series[-1] if adoption_series else 0
            
            comp_data.append({
                'community_id': comm_id,
                'community_size': comm_size,
                'strategy': strategy,
                'adoption_rate': final_rate
            })
    
    comp_df = pd.DataFrame(comp_data)
    
    sns.scatterplot(x='community_size', y='adoption_rate', hue='strategy', 
                   size='community_size', sizes=(20, 200), alpha=0.7, data=comp_df)
    
    plt.xscale('log')
    plt.xlabel('Community Size (log scale)')
    plt.ylabel('Adoption Rate')
    plt.title('Strategy Effectiveness by Community Size')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(detail_dir, 'strategy_effectiveness_by_community_size.png'), dpi=300)
    plt.close()
    
    # 4. Community adoption by strategy bar chart
    plt.figure(figsize=(12, 8))
    
    # Preparing data for bar chart
    bar_data = []
    for comm_id in top_comm_ids:
        if comm_id in adoption_by_community:
            for item in adoption_by_community[comm_id]:
                bar_data.append({
                    'community_id': comm_id,
                    'strategy': item['strategy'],
                    'adoption_rate': item['final_rate']
                })
    
    bar_df = pd.DataFrame(bar_data)
    
    # Creating bar chart
    sns.barplot(x='community_id', y='adoption_rate', hue='strategy', data=bar_df)
    plt.title('Adoption Rate by Community and Strategy (Top 5 Largest Communities)')
    plt.xlabel('Community ID')
    plt.ylabel('Adoption Rate')
    plt.legend(title='Strategy')
    plt.savefig(os.path.join(detail_dir, 'community_adoption_by_strategy.png'), dpi=300)
    plt.close()
    

    comp_df.to_csv(os.path.join(detail_dir, 'strategy_community_comparison.csv'), index=False)
    
    # Creating summary document
    with open(os.path.join(detail_dir, 'community_analysis_summary.txt'), 'w') as f:
        f.write("# Community-Based Diffusion Analysis\n\n")
        
        f.write(f"Total communities: {len(community_ids)}\n")
        f.write(f"Largest community: {top_communities[0][0]} with {top_communities[0][1]} nodes\n\n")
        
        f.write("## Seeding Strategies\n\n")
        
        # Comparing seeding strategies
        for strategy in strategies:
            if strategy in strategy_community_data:
                data = strategy_community_data[strategy]
                
                # Calculating seeding concentration
                seed_counts = [count for comm_id, count in data['seed_distribution'].items()]
                top_seeds = sum(sorted(seed_counts, reverse=True)[:3])
                total_seeds = sum(seed_counts)
                
                concentration = top_seeds / total_seeds if total_seeds > 0 else 0
                
                f.write(f"### {strategy}\n")
                f.write(f"- Seed concentration in top 3 communities: {concentration:.2f}\n")
                
                # Finding communities with most seeds
                top_seed_comms = sorted([(comm_id, count) for comm_id, count in data['seed_distribution'].items()], 
                                      key=lambda x: x[1], reverse=True)[:3]
                
                f.write("- Top seeded communities:\n")
                for comm_id, count in top_seed_comms:
                    f.write(f"  - Community {comm_id}: {count} seeds ")
                    f.write(f"({count/community_sizes[comm_id]:.1%} of community)\n")
                
                f.write("\n")
        
        f.write("## Community Adoption Patterns\n\n")
        
        # Analyzing adoption patterns by community
        for comm_id, size in top_communities:
            f.write(f"### Community {comm_id} (size: {size})\n")
            
            # Comparing strategies for this community
            if comm_id in adoption_by_community:
                comm_data = adoption_by_community[comm_id]
                
                # Sorting by final adoption rate
                sorted_data = sorted(comm_data, key=lambda x: x['final_rate'], reverse=True)
                
                f.write("Strategy effectiveness:\n")
                for data in sorted_data:
                    f.write(f"- {data['strategy']}: {data['final_rate']:.2f} adoption rate ")
                    f.write(f"(from {data['seed_rate']:.2f} seed rate)\n")
                
                # Calculating impact ratio (adoption rate / seed rate)
                for data in sorted_data:
                    if data['seed_rate'] > 0:
                        impact = data['final_rate'] / data['seed_rate']
                        f.write(f"  - Impact ratio: {impact:.2f}x\n")
            
            f.write("\n")
    
    print(f"Community-based diffusion analysis completed. Results saved to {detail_dir}")
    return strategy_community_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze diffusion results")
    parser.add_argument("--results_dir", required=True, help="Directory containing diffusion results")
    parser.add_argument("--network_file", required=True, help="Path to original network file")
    parser.add_argument("--output_dir", default="diffusion_analysis", help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    analyze_diffusion_results(args.results_dir, args.network_file, args.output_dir)