# ----- Imports & Paramaters -----

import json 
import pandas as pd
import numpy as np
import copy
import openpyxl
import os
import csv
from sklearn.metrics import log_loss
import ot
from tqdm.notebook import tqdm
from scipy.stats import norm
from scipy.stats import kstest
import matplotlib.pyplot as plt

from pymatgen.io.jarvis import JarvisAtomsAdaptor
from jarvis.core.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN

from data_loader import load_local_data, histog, build_noisy_circular_graph
import networkx as nx
from graph import graph_colors, draw_rel, draw_transp, Graph, wl_labeling
from ot_distances import Fused_Gromov_Wasserstein_distance, Wasserstein_distance
from tqdm import tqdm 
from scipy.sparse.csgraph import shortest_path

print("Modules imported; the program is ready to run.")


# ======================================
#          Defining Functions 
# ======================================

# Convert a Graph() object into a dictionary with node and edge info. 
# If label='petti', it uses Pettifor numbers from 'petti_num'

def graph2dict(G, label='skip'):
    atom_names = [site.label for site in G.structure.sites]

    nodeArr = []
    for ii, node in enumerate(list(G.graph.nodes(data=True))):
        if label == 'skip':
            node[1]['skip'] = skip_vecs[atom_names[ii]].to_list()
        elif label == 'petti':
            node[1]['petti'] = [petti_num[atom_names[ii]]]
        nodeArr.append(node)

    edgeArr = []
    for edge in list(G.graph.edges()): 
        edgeArr.append(edge)

    graphDict = { 'nodes': nodeArr,
                  'edges': edgeArr}

    return graphDict

# Reconstruct a Graph() object from a dictionary representation.
# To generate graphical representations in a chosen feature space.

def dict2graph(D, label='skip'):
    g = Graph()
    node_dict = {}
    for i, node in enumerate(D['nodes']):
        if label == 'skip':
            node_dict[i] = node[1]['skip']
        elif label == 'petti':
            node_dict[i] = node[1]['petti']

    g.add_attibutes(node_dict)
    for edge in D['edges']:
        g.add_edge(edge)
    return g

# Generate FGW distance matrix among a list of graph() objects. 
# Alpha is the weighting for FGW

def getfusedvalue(graphs, alpha): 
    output = np.zeros([len(graphs),len(graphs)])
    if len(graphs) > 0:
          print(f"First graph has {len(graphs[0].nodes())} nodes")
          print(f"Node attributes shape: {dir(graphs[0])}")
          print(f"Sample node attribute: {graphs[0].nodes_attributes[0] if hasattr (graphs[0], 'nodes_attributes') and len(graphs[0].nodes_attributes) > 0 else 'No attributes'}")
    for j in range(len(graphs)): 
        for k in range(len(graphs)):
            try:
                output[j,k]=Fused_Gromov_Wasserstein_distance(
                    alpha=alpha,
                    features_metric='euclidean',
                    method='harmonic_distance'
                ).graph_d(graphs[j], graphs[k])        
            except Exception as e: 
                print(f"Error calculating distance for graph {j} and {k}: {e}")
                if j < 5 and k < 5:
                        print(f"Graph {j} nodes: {len(graphs[j].nodes)}, sample attr: {graphs[j].nodes_attributes[0] if len(graphs[j].nodes_attributes) > 0 else 'No attributes'}")
                        print(f"Graph {k} nodes: {len(graphs[k].nodes)}, sample attr: {graphs[k].nodes_attributes[0] if len(graphs[k].nodes_attributes) > 0 else 'No attributes'}")
                output[j,k]=0
    non_zeros = np.count_nonzero(output)
    print(f"Number of non-zero distances: {non_zeros} out of {len(graphs)*len(graphs)} pairs")
    return output

# Save all the info in an Excel file

def save_side_by_side_excel(filename, run_name, metrics_dict):
    new_df = pd.DataFrame(list(metrics_dict.items()), columns=['Key', run_name])
    new_df.set_index('Key', inplace=True)

    if not os.path.isfile(filename):
        with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
            new_df.to_excel(writer, sheet_name='GMM_Analysis_Results')
    else: 
        old_df = pd.read_excel(filename, sheet_name='GMM_Analysis_Results', index_col=0)
        if run_name in old_df.columns:
            # Add a suffix to avoid column name conflict
            suffix_num = 1
            while f"{run_name}_{suffix_num}" in old_df.columns:
                suffix_num += 1
            run_name = f"{run_name}_{suffix_num}"
            new_df.columns = [run_name]
        merged_df = old_df.join(new_df, how='outer')
        with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
            merged_df.to_excel(writer, sheet_name='GMM_Analysis_Results')


# ======================================
#            Main Execution
# ======================================

def main():

    '''
    1. loads structures from JSON
    2. Builds graphs via pymatgen + CrystalNN
    3. Import Pettifor numbers
    4. Computes FGW distance matrix, normalise, and saves to Excel
    5. Histogram + normal distribution fit 
    6. Check normal fit (KS test), if normal fit is poor, do BIC-based GMM
    7. GMM histogram
    8. Save GMM results to Excel'
    9. Run t-SNE on the normalised FGW distances for 2D visualisation'
    '''
   
    # --------  1. Load data -------- 
    
    with open('/Users/y.h.jollin/FGW/MPtrj_1K_structures.json', 'r') as f:
        data = json.load(f)
    print("Number of entries in JSON:", len(data))

    data = data[:400]

    structures = []
    for entry in tqdm(data):
        try:
            struct_dict = entry["structure"]
            structure = Structure.from_dict(struct_dict)
            structures.append(structure)
        except KeyError as e:
            print(f"Missing key in entry: {e}")
            continue
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
    print(f"Successfully loaded {len(structures)} structures")

    # -------- 2. Build graphs --------
    
    struct_graphs = []
    for structure in tqdm(structures):
        try:
            structure.add_oxidation_state_by_guess()
            graph = StructureGraph.from_local_env_strategy(structure, CrystalNN())
            struct_graphs.append(graph)
        except Exception as e:
            print(f"Error creating graph: {e}")
            continue
    
    # -------- 3. Import Pettifor numbers --------
    
    global petti_num
    petti_num = pd.read_json('/Users/y.h.jollin/FGW/mod_petti.json', typ='series')

    # Generate Pettifor feature vectors for Fabinni data from the intitial graphical representations
    dicts = []
    dicts = [graph2dict(g, label='petti') for g in struct_graphs]
    
    if len(dicts) > 0:
        print("Sample dict keys:", dicts[0].keys())
        print("Number of nodes in first dict:", len(dicts[0]["nodes"]))
        print("Number of edges in first dict:", len(dicts[0]["edges"]))
    
    graphs = [dict2graph(d, label='petti') for d in dicts]
    if len(graphs) > 0:
        first_graph = graphs[0]
        print("First graph has", len(graphs[0].nodes()), "nodes.")
    else:
        print("No graphs found:(")

    # -------- 4. Compute FGW distance matrix --------

    fusedGW = getfusedvalue(graphs, alpha=0.5)
    #print("fusedGW matrix shape:", fusedGW.shape)
    if fusedGW.size > 0:
       print("First row of fusedGW:", fusedGW[0])

    # Normalise the matrix
    x = np.max(fusedGW)
    #print("Max distance in fusedGW:", x)
    if x > 0:
        fusedGWNorm = fusedGW /x
        print(f"Normalised matrix with maximum value: {x}")
    else:
        print("Ohno: Maximum value in fusedGW is zero, skipping normalisation")
        fusedGWNorm = fusedGW

    # Convert to Dataframe 
    fusedGWDF = pd.DataFrame(fusedGW)
    fusedGWNormDF = pd.DataFrame(fusedGWNorm)

    # Get Excel 
    with pd.ExcelWriter('pettiFusedGW.xlsx', mode='w', engine="openpyxl") as writer:
        fusedGWDF.to_excel(writer, sheet_name="FusedGW")
        fusedGWNormDF.to_excel(writer, sheet_name="NormalisedFusedGW")
    print("Fused GW matrix saved to 'pettiFusedGW.xlsx")

    # -------- 5. Histogram + normal distribution fit --------
    
    N = fusedGWNorm.shape[0]
    idx = np.triu_indices(N, k=1)
    pairwise_distances = fusedGWNorm[idx]

    # Basic data
    median_d = np.median(pairwise_distances)
    min_d = np.min(pairwise_distances)
    max_d = np.max(pairwise_distances)
    print("Diversity statistic - nomralised FGW:")
    print(" Median distance:", median_d)
    print(" Min distance:", min_d)
    print(" Max distance:", max_d)
    
    # Mean + std
    mean = np.mean(pairwise_distances)
    std = np.std(pairwise_distances)
    print(f" Mean of distance: {mean:.4f}")
    print(f" Standard deviation of distance: {std:.4f}")

    # Histogram with normal fit
    plt.figure(figsize=(6,4))
    count, bins, patches = plt.hist(
            pairwise_distances,
            bins = 100, 
            density = True,
            color='orange',
            alpha=0.5,
            edgecolor='gray'
        )
    xmin, xmax = bins[0], bins[-1]
    x = np.linspace(xmin, xmax, 200)
    normal_pdf = norm.pdf(x, loc=mean, scale=std)
    plt.plot(x, normal_pdf, 'k-', linewidth=2, label='Normal PDF')

    plt.title("Histogram of FGW Distances with Normal Fit")
    plt.xlabel("Normalised FGW Distance")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("Histogram of FGW Distances with Normal Fit.png")
    plt.close()
    print("Histogram (nomral fit) has been saved in .png format")

    # -------- 6. Check normal fit (KS test), if fit is poor do GMM --------
    
    D, p_value = kstest(pairwise_distances, 'norm', args=(mean, std))
    print(f"K-S test statistic: {D:.3f}")
    print(f"K-S test p-value: {p_value:.3e}")
    if p_value > 0.5:
        print("Normal Distribution is a good fit")
    else:
        print("Normal fit is poor, let's do GMM")
        
        # Fit a GMM model 
        from sklearn.mixture import GaussianMixture
        reshaped = pairwise_distances.reshape(-1, 1)
        # the following is used to make sure how many Gaussians needs to be fitted
        lowest_bic = np.inf
        best_n = None
        for n in range(1,6):
            gmm = GaussianMixture(n_components=n, covariance_type='full')
            gmm.fit(reshaped)
            bic = gmm.bic(pairwise_distances.reshape(-1, 1))
            if bic < lowest_bic: # if the current bic score is better (lower) than what i thought is the best one 
                lowest_bic = bic
                best_n = n
        print(f"Best number of components: {best_n}")

        # -------- 7. GMM histogram --------
        
        gmm = GaussianMixture(n_components=best_n, covariance_type='full')
        gmm.fit(reshaped)

         # Converged status, log likelihood score, weights, means, covars
        converged_status = gmm.converged_
        log_likelihood_score = gmm.lower_bound_
        weights = gmm.weights_
        means = gmm.means_.flatten()
        covars = gmm.covariances_.flatten()

        # Plot GMM
        plt.figure(figsize=(10,6))
        count, bins, patches = plt.hist(
            pairwise_distances,
            bins = 100, 
            density = True,
            color='orange',
            alpha=0.6,
            edgecolor='gray'
            )
        
        xmin, xmax = min(pairwise_distances), max(pairwise_distances)
        x = np.linspace(xmin, xmax, 200)

        colors = ["red", "blue", "green", "orange", "purple"]
        
        for i, (w, mu, var_) in enumerate(zip(weights, means, covars)):
            single_pdf = w * norm.pdf(x, loc=mu, scale=np.sqrt(var_))
            plt.plot(
                x, single_pdf,
                color = colors[i % len(colors)],
                linestyle='-', 
                linewidth=2,
                label=f"GMM Component {i+1} (μ={mu:.3f}, w={w * 100:.2f}%, Σ={var_:.3f})"
            )

            plt.vlines(
                x=mu,
                ymin=0,
                ymax=max(single_pdf),
                color= colors[i % len(colors)],
                linestyle='--',
                alpha=0.7
            )

        plt.title("Histogram of FGW Distances with GMM fit")
        plt.xlabel("Normalised FGW Distance")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("Histogram with GMM.png")
        plt.close()
        print("GMM is fitted and saevd in .png format")

    # -------- 8. Save GMM results to Excel --------
    
    metrics_dict = {
        "KS_Stat": D,
        "p_value": p_value,
        "GMM_BestN": best_n,
        "GMM_BIC": lowest_bic,
        "Converged_Status": converged_status,
        "Log_Likelihood_Score": log_likelihood_score,
        "GMM_Weights": ", ".join([f"{w:.4f}" for w in weights]),
        "GMM_Means": ", ".join([f"{m:.4f}" for m in means]),
        "GMM_Covars": ", ".join([f"{c:.4f}" for c in covars])
    }
    
    save_side_by_side_excel(
        filename='GMM_Analaysis_Results.xlsx',
        run_name='50',
        metrics_dict=metrics_dict
    )

    # -------- 9. Run t-SNE --------
    
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2, 
        metric="precomputed", 
        perplexity=40,
        init="random",
        n_iter=1000, 
        random_state=42
    )
    coords = tsne.fit_transform(fusedGWNorm)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=20, alpha=0.7)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE 2D Embedding of Structures (Normalised FGW Distances)")
    plt.show()

if __name__ == "__main__":
    main()

