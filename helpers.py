import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def cosine_similarity(x,y):
    """
    Given two vectors of the same length, return the cosine similarity betweem the two vectors.
    """
    return 0.5 * (1 + (np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y)))
	
def get_adjacency_multidistance(features_list, weights, distance_function, kernel_function, sparsify):
    '''
    Construct an adjecency matrix matrix using  the features passed in parameter.
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

def plot_graph(G, node_color, edge_threshold=0.5, scale=None, highlight_node=[], ax=None, colormap=plt.get_cmap('coolwarm')):
    pos = nx.spring_layout(G,seed=2019, iterations=200)
    
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > edge_threshold]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= edge_threshold]

    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=0.5, alpha=0.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=elarge,
                       width=1, alpha=0.5, ax=ax)
    
    if scale is not None:
        vmin = scale[0]
        vmax = scale[1]
    else:
        vmin= None
        vmax=None
        
    if len(highlight_node)>0:
        node_size=[900 if x in highlight_node else 300  for x in G.nodes()]
    else:
        node_size = 300
    nc = nx.draw_networkx_nodes(G,pos,with_labels=True,node_color= node_color, cmap=colormap,
                               vmin=vmin,vmax=vmax,node_size=node_size,ax=ax)
    nx.draw_networkx_labels(G, pos, alpha=0.5,ax=ax)
    plt.axis('off')
    if ax is not None:
        ax.axis('off')
    return nc
	
	
def plot_signal(adjacency, signal, **kwargs):
    adjacency_temp = adjacency.copy()
    G = nx.from_numpy_matrix(adjacency_temp)    
    return plot_graph(G,np.round(signal,5), edge_threshold = np.max(adjacency) * 0.9, **kwargs)