import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_adjacency(adjacency):
    plt.figure(figsize=(9,9))
    plt.spy(adjacency, markersize=3)
    plt.title('adjacency matrix')
    plt.show()


def plot_graph(G, node_color, edge_threshold=0.5, scale=None, highlight_node=[], ax=None, colormap=plt.get_cmap('Set1'), positions=None):
    
    pos = nx.spring_layout(G,seed=2018, iterations=400, pos=positions, k=2.25)
    
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
        
    
    nc = nx.draw_networkx_nodes(G,pos,with_labels=True, nodelist=normal_nodes, node_color=node_color[normal_nodes_indices], cmap=colormap,
                               vmin=vmin,vmax=vmax,node_size=300,ax=ax, edgecolors='black', alpha=0.6)
                               
    nx.draw_networkx_labels(G, pos, alpha=0.7, color='gray', ax=ax)
                            
    nc = nx.draw_networkx_nodes(G,pos,with_labels=True, nodelist=special_nodes, node_color=[0.6]*len(highlight_node), cmap=colormap,
                               vmin=vmin,vmax=vmax,node_size=2000,ax=ax, edgecolors='black', alpha=1.0, node_shape='H')
                               
    
    
    
    plt.axis('off')
    
    if ax is not None:
        ax.axis('off')
        
    return nc
    
    
def plot_signal(adjacency, signal, labels=None, **kwargs):

    plt.figure(figsize=(20,15))

    adjacency_temp = adjacency.copy()
    G = nx.from_numpy_matrix(adjacency_temp)    
    
    if labels is not None:
        G = nx.relabel_nodes(G, lambda x : labels[x])
    
    return plot_graph(G,np.round(signal,5), edge_threshold = np.max(adjacency) * 0.9, **kwargs)
    
    
def show_political_spectrum(embedding, n, colors, senators_party):
    fig = plt.figure(figsize=(15,2))
    ax = plt.subplot(111)
    ax.set_title("Your position on the political spectrum")

    for i in range(n-1):
        ax.scatter(embedding[i,0], np.random.normal(.5,.05), alpha=0.25, color=colors[senators_party[i]])
        
    #ax.scatter(embedding[n-1,0], embedding[n-1,1], color='black', s=100, marker='H')
    xs = np.linspace(np.min(embedding[:,0]),np.max(embedding[:,0]),100)

    #ax.plot(xs, density(xs))
    ax.set_xticks([np.min(embedding[:,0]),np.max(embedding[:,0])])
    ax.set_ylim([0,1])
    ax.set_xticklabels(['< liberal','conservative >'])
    plt.tick_params(axis='x', which='both', bottom=False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot([embedding[n-1,0], embedding[n-1,0]], [0, 0.5], 'k-', lw=1)

    plt.text(embedding[n-1,0] - 0.0035, 0.65, "You", fontsize=12)
    plt.show()