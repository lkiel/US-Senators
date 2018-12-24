import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_adjacency(adjacency):
    plt.figure(figsize=(9,9))
    plt.spy(adjacency, markersize=3)
    plt.title('adjacency matrix')
    plt.show()


def cosine_similarity(x,y):
    """
    Given two vectors of the same length, return the cosine similarity between the two vectors.
    """
    return 0.5 * (1 + (np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y)))
    
    
def gaussian_kernel(distance_matrix):
    """
    Given a matrix calculate the gaussian kernel of it. The variance is given by the mean of the matrix. 
    """
    kernel_width = distance_matrix.mean()
    res = np.exp(-distance_matrix**2 / kernel_width**2)
    return res
    

def euclidean_distance(x,y):
    """
    Given two vectors of the same length, return the euclidean distance between the two vectors.
    """
    return np.sqrt(np.sum(np.power(x-y,2)))
    
    
def get_adjacency_multidistance(features_list, weights, distance_function, kernel_function, sparsify):
    '''
    Construct an adjacency matrix matrix using  the features passed in parameter.
    The final result is computed as a weighted sum of the distances matrices obtained using the different features 
    '''
    number_of_nodes = features_list[0].shape[0]
    result = np.zeros((number_of_nodes, number_of_nodes))
    meshgrid_nodes = np.mgrid[:number_of_nodes,:number_of_nodes]
    
    for i, features in enumerate(features_list):
        distance_matrix = np.apply_along_axis(lambda x: distance_function(features[x[0]],features[x[1]]) ,0,meshgrid_nodes)
        kernel = kernel_function(distance_matrix)
        np.fill_diagonal(kernel, 0)
        
        result += weights[i] * sparsify[i](kernel)
        
    return result  
    
def sparsify_with_limit(adjacency, limit = 0.35):
    """
    Sparsify a matrix by putting each of element of the adjacency matrix to 0
    if it is below the limit.
    """
    res = adjacency.copy()
    res[res < limit] = 0
    return res

def sparsify_with_max_neighbors(adjacency, max_neighbors = 45):
    """
    Sparsify a matrix by greedily adding the links of higher weight such that each node
    doesn't have more neighbors than max_neighbors 
    """
    number_of_nodes = adjacency.shape[0]    
    
    index_sort = np.argsort(adjacency,axis=None)[::-1]
    flatten_adjacency = adjacency.flatten()
    res = np.zeros(adjacency.shape)
    counter = 0

    for i in range(len(index_sort)):
        if counter >= max_neighbors * number_of_nodes:
            break
        
        node_1 = index_sort[i] % number_of_nodes
        node_2 = int(index_sort[i] / number_of_nodes)

        if(np.count_nonzero(res[node_1]) < max_neighbors and np.count_nonzero(res[node_2]) < max_neighbors):
            res[node_1,node_2] = flatten_adjacency[index_sort[i]]
            res[node_2,node_1] = flatten_adjacency[index_sort[i]]
            counter += 1
                   
    return res

def plot_graph(G, node_color, edge_threshold=0.5, scale=None, highlight_node=[], ax=None, colormap=plt.get_cmap('Set1'), positions=None):
    
    pos = nx.spring_layout(G,seed=2018, iterations=400, pos=positions, k=1.5)
    
    e_weights = nx.get_edge_attributes(G,'weight')
    e_weights = np.array(list(e_weights.values()))
    e_weights = 0.5 + 2 * (e_weights - np.min(e_weights))/(np.max(e_weights) - np.min(e_weights))
    
    nx.draw_networkx_edges(G, pos, width=e_weights, alpha=0.2, ax=ax, style='dotted')
    
    
    if scale is not None:
        vmin = scale[0]
        vmax = scale[1]
    else:
        vmin= None
        vmax= None
        
    node_size = [2000 if i in highlight_node else 300  for i,_ in enumerate(G.nodes())]
    node_color = np.array([0.6 if i in highlight_node else node_color[i] for i,_ in enumerate(G.nodes())])
    
    
    normal_nodes = [n for i,n in enumerate(G.nodes()) if i not in highlight_node]
    normal_nodes_indices = [i for i,n in enumerate(G.nodes()) if i not in highlight_node]
    special_nodes = [n for i,n in enumerate(G.nodes()) if i in highlight_node]
    
    print(special_nodes)
    
    
    nc = nx.draw_networkx_nodes(G,pos,with_labels=True, nodelist=normal_nodes, node_color=node_color[normal_nodes_indices], cmap=colormap,
                               vmin=vmin,vmax=vmax,node_size=300,ax=ax, edgecolors='black', alpha=0.6)
                               
    nx.draw_networkx_labels(G, pos, alpha=0.7, color='gray', ax=ax)
                            
    nc = nx.draw_networkx_nodes(G,pos,with_labels=True, nodelist=special_nodes, node_color=[0.6], cmap=colormap,
                               vmin=vmin,vmax=vmax,node_size=2000,ax=ax, edgecolors='black', alpha=1.0, node_shape='H')
                               
    
    
    
    plt.axis('off')
    
    if ax is not None:
        ax.axis('off')
        
    return nc
    
    
def plot_signal(adjacency, signal, labels=None, **kwargs):
    adjacency_temp = adjacency.copy()
    G = nx.from_numpy_matrix(adjacency_temp)    
    
    if labels is not None:
        G = nx.relabel_nodes(G, lambda x : labels[x])
    
    return plot_graph(G,np.round(signal,5), edge_threshold = np.max(adjacency) * 0.9, **kwargs)