import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.lines as mlines

def plot_adjacency(adjacency):
    plt.figure(figsize=(9,9))
    plt.spy(adjacency, markersize=3)
    plt.title('adjacency matrix')
    plt.show()
    
    
def plot_prediction(G_pyGSP, sol, labels, mask):
    
    G = nx.from_numpy_matrix(G_pyGSP.W.todense())
    
    true_pos = np.argwhere((sol == labels) & (sol == 1))[:,0].tolist()
    true_neg = np.argwhere((sol == labels) & (sol == -1))[:,0].tolist()
    false_pos = np.argwhere((sol != labels) & (sol == 1))[:,0].tolist()
    false_neg = np.argwhere((sol != labels) & (sol == -1))[:,0].tolist()
    measured_pos = np.argwhere((mask==1) & (labels==1))[:,0].tolist()
    measured_neg = np.argwhere((mask==1) & (labels==-1))[:,0].tolist()
    
    fig = plt.figure(figsize=(20,15))
    
    pos = nx.spring_layout(G,seed=2018, iterations=500, k=5.5)
        
    # Draw edges
    e_weights = nx.get_edge_attributes(G,'weight')
    e_weights = np.array(list(e_weights.values()))
    e_weights = 0.5 + 2 * (e_weights - np.min(e_weights))/(np.max(e_weights) - np.min(e_weights))
    nx.draw_networkx_edges(G, pos, width=e_weights, alpha=0.2, style='dotted')
    
    # Draw true positives
    if len(true_pos) > 0:
        nc = nx.draw_networkx_nodes(G, pos, nodelist=true_pos, node_color='limegreen', linewidths=4, edgecolors='forestgreen')
    if len(true_neg) > 0:
        nc = nx.draw_networkx_nodes(G, pos, nodelist=true_neg, node_color='salmon', linewidths=4, edgecolors='crimson')
    if len(false_pos) > 0:
        nc = nx.draw_networkx_nodes(G, pos, nodelist=false_pos, node_color='limegreen', linewidths=4, edgecolors='crimson')
    if len(false_neg) > 0:
        nc = nx.draw_networkx_nodes(G, pos, nodelist=false_neg, node_color='salmon', linewidths=4, edgecolors='forestgreen')
    if len(measured_pos) > 0:
        nc = nx.draw_networkx_nodes(G, pos, nodelist=measured_pos, node_color='forestgreen', linewidths=4, edgecolors='forestgreen')
    if len(measured_neg) > 0:
        nc = nx.draw_networkx_nodes(G, pos, nodelist=measured_neg, node_color='crimson', linewidths=4, edgecolors='crimson')

    plt.axis('off')
    
    tp = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='', markeredgewidth=4, markeredgecolor='forestgreen', 
                          markersize=20, label="True positive")
    tn = mlines.Line2D([], [], color='salmon', marker='o', linestyle='', markeredgewidth=4, markeredgecolor='crimson', 
                          markersize=20, label="True negative")
    fp = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='', markeredgewidth=4, markeredgecolor='crimson', 
                          markersize=20, label="False positive")
    fn = mlines.Line2D([], [], color='salmon', marker='o', linestyle='',  markeredgewidth=4, markeredgecolor='forestgreen',
                          markersize=20, label="False negative")
    mes = mlines.Line2D([], [], color='white', marker='s', linestyle='',  markeredgewidth=4, markeredgecolor='forestgreen',
                          markersize=0, label="Filled = measured")

    plt.legend(handles=[tp,tn,fp,fn,mes], prop={'size':20})
    fig.suptitle("Signal reconstruction results measuring {prop}% of realizations".format(prop = np.round(100*np.sum(mask) / len(mask),1)), fontsize=24)

        
    return nc


def plot_graph(G, node_color, edge_threshold=0.5, scale=None, highlight_node=[], ax=None, colormap=plt.get_cmap('Set1'), positions=None, k=5.5):
    
    pos = nx.spring_layout(G, weight='weight',  seed=2018, iterations=500, pos=positions, k=k)
    
    e_weights = nx.get_edge_attributes(G,'weight')
    e_weights = np.array(list(e_weights.values()))
    e_weights = 0.5 + 2 * (e_weights - np.min(e_weights))/(np.max(e_weights) - np.min(e_weights))
    
    nx.draw_networkx_edges(G, pos, width=e_weights, alpha=0.15, ax=ax, style='dotted')
    
    if scale is not None:
        vmin = scale[0]
        vmax = scale[1]
    else:
        vmin= None
        vmax= None
        
    node_color = np.array([0.6 if i in highlight_node else node_color[i] for i,_ in enumerate(G.nodes())])
    
    
    normal_nodes = [n for i,n in enumerate(G.nodes()) if i not in highlight_node]
    normal_nodes_indices = [i for i,n in enumerate(G.nodes()) if i not in highlight_node]
    special_nodes = [n for i,n in enumerate(G.nodes()) if i in highlight_node]
        
    
    nc = nx.draw_networkx_nodes(G,pos,with_labels=True, nodelist=normal_nodes, node_color=node_color[normal_nodes_indices], cmap=colormap,
                               vmin=vmin,vmax=vmax,node_size=300,ax=ax, edgecolors='black', alpha=0.6)
                               
    nx.draw_networkx_labels(G, pos, alpha=0.7, font_color='black', ax=ax)
                            
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
    
    return plot_graph(G, signal, edge_threshold = np.max(adjacency) * 0.9, **kwargs)
    
   
def show_2D_embedding(embedding, senators_party):

    colors = {'R':'red','D':'blue','I':'green'}
    plt.figure(figsize=(6,4))

    n = len(embedding)
    for i in range(n-1):
        plt.scatter(embedding[i,0], embedding[i,1], alpha=0.25, s=50, c=colors[senators_party[i]])
        
    plt.scatter(embedding[n-1,0], embedding[n-1,1], color='black', s=200, marker='x')
    plt.xlabel("Coordinate on first eigenvector")
    plt.ylabel("Coordinate on second eigenvector")
    plt.show()
    
    
def show_2D_embedding_ax(embedding, senators_party, ax):

    colors = {'R':'red','D':'blue','I':'green'}

    n = len(embedding)
    for i in range(n-1):
        ax.scatter(embedding[i,0], embedding[i,1], alpha=0.25, s=50, c=colors[senators_party[i]])
        
    ax.scatter(embedding[n-1,0], embedding[n-1,1], color='black', s=200, marker='x')
    
    
    
def show_political_spectrum(embedding, n, colors, senators_party):
    fig = plt.figure(figsize=(15,2))
    ax = plt.subplot(111)
    ax.set_title("Your position on the political spectrum (first eigenvector)")

    n = len(embedding)
    sen_embedding = embedding[:n-1,:]

    reps = sen_embedding[senators_party == 'R',:]
    dems = sen_embedding[senators_party == 'D',:]
    inds = sen_embedding[senators_party == 'I',:]

    ax.scatter(reps[:,0], np.random.normal(.5,.05, size=len(reps)), alpha=0.25, color='r')
    ax.scatter(dems[:,0], np.random.normal(.5,.05, size=len(dems)), alpha=0.25, color='b')
    ax.scatter(inds[:,0], np.random.normal(.5,.05, size=len(inds)), alpha=0.25, color='g')
        
    ax.set_xticks([np.min(embedding[:,0]),np.max(embedding[:,0])])
    ax.set_ylim([0,1])
    ax.set_xticklabels(['< liberal','conservative >'] if np.mean(reps) > np.mean(dems) else ['< conservative','liberal >'])
    plt.tick_params(axis='x', which='both', bottom=False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot([embedding[n-1,0], embedding[n-1,0]], [0, 0.5], 'k-', lw=1)

    plt.text(embedding[n-1,0] - 0.0035, 0.65, "You", fontsize=12)
    plt.show()
    
    
def show_portraits(similar_senators):
    # read images
    s = "data/senate_members/{id}.jpg"
    t = "{name} ({party})"

    # display images
    fig, ax = plt.subplots(1,3, figsize=(15,6))
    fig.suptitle("Senators whose voting positions are the most similar to you", fontsize=32)
    plt.subplots_adjust(top=0.75)

    for i in range(3):
        ax[i].axis('off')
        im = ax[i].imshow(mpimg.imread(s.format(id=similar_senators.index[i])));
        ax[i].set_title(t.format(name=similar_senators['name'].iloc[i], party=similar_senators['party'].iloc[i]), fontsize=24)
        patch = patches.FancyBboxPatch((0, 2.5), boxstyle="round,rounding_size=10", width=220, height=270, transform=ax[i].transData)
        im.set_clip_path(patch)
    
    plt.show()