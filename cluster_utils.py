import numpy as np
import scipy.sparse.linalg
from sklearn.cluster import KMeans

def embed(laplacian,d):
    eigenvalues,eigenvectors = scipy.linalg.eigh(laplacian)
    indexes = np.argsort(eigenvalues)
    
    return eigenvectors[:,indexes][:,:d]

def gen_kmeans(eigenvectors,k=3,random_state = 42):
    means = KMeans(n_clusters=k, random_state=random_state).fit(eigenvectors)
    return means.labels_,means.cluster_centers_