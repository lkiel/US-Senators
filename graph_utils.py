import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def cosine_similarity(x,y):
    """
    Given two vectors of the same length, return the cosine similarity between the two vectors.
    """
    return 0.5 * (1 + (np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y)))
    

def euclidean_distance(x,y):
    """
    Given two vectors of the same length, return the euclidean distance between the two vectors.
    """
    return np.sqrt(np.sum(np.power(x-y,2)))
    
    
def gaussian_kernel(distance_matrix):
    """
    Given a matrix calculate the gaussian kernel of it. The variance is given by the mean of the matrix. 
    """
    kernel_width = distance_matrix.mean()
    res = np.exp(-distance_matrix**2 / kernel_width**2)
    return res
  
  
def identity_kernel(distance_matrix):
    return distance_matrix
    
    
def get_adjacency_matrix(features, distance_function, kernel_function, sparsification_function):
    '''
    Construct an adjacency matrix matrix by computing the similarities between feature vectors passed as parameters.
    '''
    number_of_nodes = features.shape[0]
    result = np.zeros((number_of_nodes, number_of_nodes))
    meshgrid_nodes = np.mgrid[:number_of_nodes,:number_of_nodes]
    
    distance_matrix = np.apply_along_axis(lambda x: distance_function(features[x[0]],features[x[1]]) , 0, meshgrid_nodes)
    kernel = kernel_function(distance_matrix)
    np.fill_diagonal(kernel, 0)
    
    result = sparsification_function(kernel)
        
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
    
    
def print_graph_specs(adjacency):
    print("Clustering coefficient: " + str(nx.average_clustering(nx.from_numpy_matrix(adjacency))))
    print("Diameter: " + str(nx.diameter(nx.from_numpy_matrix(adjacency))))
    plt.figure(figsize=(5,3))
    plt.hist(adjacency.sum(1), bins=range(20), color='silver', lw=1, edgecolor='black')
    plt.title("Node degree distribution")
    plt.savefig("deg_distribution.png")
    plt.show()

