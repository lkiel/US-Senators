from transducive_learning_utils import *
from utils import *
from sklearn.neighbors import NearestNeighbors

def predict_using_transductive_learning(votes_selection,votes_df,G_votation,number_of_trials = 1):
    """
    Uses transductive learning to predict missing votes for every senator and return the mean vote prediction accuracy
    on this task
    
    Parameters
    ----------
    votes_selection : nparray
        labeled votes indexes
    votes_df : Dataframe
        votes feature matrix
    G_votation : graph
        Bills similarity graph
    number_of_trials: int (optional)
        Define the number of time each prediction should be run. the final prediction for every vote is a average of
        "number_of_trials" predictions made for it 
    Returns
    -------
    float
        mean vote prediction accuracy
    """
    
    # Votes mask
    w = np.zeros(votes_df.shape[1])
    w[votes_selection] = 1
    accuracies_accumulator = 0
    # Iterate over senators
    for senator in range(votes_df.shape[0]):
        #run prediction
        votes_np = votes_df.iloc[senator,:].values
        available_votes_idx = np.nonzero(votes_np != 0)[0]
        ground_truth = get_thresholded_values(votes_np,0)
        sol, sol_bin = reconstruct_signal(G_votation, w, ground_truth,number_of_trials=number_of_trials)
        #Accumulate the obtained accuracy
        accuracies_accumulator += accuracy(ground_truth[available_votes_idx],sol_bin[available_votes_idx])
    #average accuracies
    accuracies_accumulator /= votes_df.shape[0]
    return accuracies_accumulator


def knn_predict(similar_senators_votes):
    """Predict votes based on k similar voters positions passed in parameters"""
    prediction = np.apply_along_axis(lambda a: np.histogram(a,  bins=[-1,0,1,2])[0], 0, similar_senators_votes)
#     print(prediction)
    prediction = np.argmax(prediction[[0,2],:],axis=0)
    prediction[prediction == 0] = -1
    return prediction
def predict_using_knn(votes_selection,votes_df,number_of_neighbors = 3):
    """
    Uses KNN to predict missing votes for every senator and return the mean vote prediction accuracy
    on this task
    
    Parameters
    ----------
    votes_selection : nparray
        labeled votes indexes
    votes_df : Dataframe
        votes feature matrix
    number_of_neighbors : int (optional)
        The number of similar senators to take into account for every prediction
    Returns
    -------
    float
        mean vote prediction accuracy
    """
    accuracies_accumulator = 0
    tmp = votes_df.iloc[:,votes_selection].values
    # compute pairwise distance between all the senators using the selected votes
    model = NearestNeighbors(n_neighbors=number_of_neighbors+1, algorithm='ball_tree').fit(tmp)
    distances, indices = model.kneighbors(tmp)
    distances = distances[:,1:]
    indices = indices[:,1:]
    selector = np.arange(votes_df.shape[1])
    selector = np.nonzero(~np.isin(selector,votes_selection))[0]
    total = 0
    for senator in range(votes_df.shape[0]):
        #select the most similar senators votes
        similar_senators = indices[senator]
        similar_senators_votes = votes_df.iloc[similar_senators].values[:,selector]
        
        #Select the ground truth for the predicted votes (used later for accuracy measures)
        ground_truth = votes_df.iloc[senator,selector].values
        available_votes_idx = np.nonzero(ground_truth != 0)[0]
        ground_truth = ground_truth[available_votes_idx]
        
        # Make predictions only if we have ground truth for the selected votes
        if  len(ground_truth):
            sol = knn_predict(similar_senators_votes)
            acc = accuracy(ground_truth,sol[available_votes_idx])
            accuracies_accumulator  += acc
            total+=1
    accuracies_accumulator /= total
    return accuracies_accumulator


def find_best_votes(selected_search_space,votes_df,G_bills,verbose=True):
    """
    print the best initial 3 votes combination using transductive learning and KNN
    
    Parameters
    ----------
    selected_search_space : list of lists
        selected votes combinations
    votes_df : Dataframe
        votes feature matrix
    G_bills : Graph
        Bills similary graph 
    verbose: Boolean (optional)
        Print best results
    Returns
    -------
    Obtained accuracies by each evaluation method
        
    """
    
    all_methods_accuracies_trans_learning = []
    all_methods_accuracies_knn = []
    
    for idx,votes_combination in enumerate(selected_search_space):
        if verbose:
            print(str(idx+1)+'/'+str(len(selected_search_space)),end='\r')
        # Make predictions
        all_methods_accuracies_trans_learning.append(predict_using_transductive_learning(votes_combination,votes_df,G_bills))
        all_methods_accuracies_knn.append(predict_using_knn(votes_combination,votes_df))

    # Find best votes combination
    all_methods_accuracies_trans_learning = np.array(all_methods_accuracies_trans_learning)
    all_methods_accuracies_knn = np.array(all_methods_accuracies_knn)
    
    best_method_1,best_method_2 = np.argmax(all_methods_accuracies_trans_learning),np.argmax(all_methods_accuracies_knn)
    combination_1,combination_2 = selected_search_space[best_method_1],selected_search_space[best_method_2]
    
    # Pretty print the best combinations
    if verbose:
        data = (best_method_1,best_method_2),(all_methods_accuracies_trans_learning[best_method_1],all_methods_accuracies_knn[best_method_2])
        log_best_votes(data,selected_search_space,votes_df)

    return all_methods_accuracies_trans_learning,all_methods_accuracies_knn

def log_best_votes(data,search_space,votes_df):
    """Pretty print 'find_best_votes' function results"""
    (m1,m2),(acc1,acc2) = data
    print('---------Transductive Learning benchmark------------')
    print('Best Method: {}'.format(m1 + 1))
    print('Accuracy: {}'.format(acc1))
    print('Votes idxs: {}'.format(search_space[m1]))
    print('Votes selected {}'.format([x for x in votes_df.iloc[:,search_space[m1]].columns]))
    
    print('---------KNN benchmark------------')
    print('Best Value Index: {}'.format(m2+ 1))
    print('Accuracy: {}'.format(acc2))
    print('Votes idxs: {}'.format(search_space[m2]))
    print('Votes selected {}'.format([x for x in votes_df.iloc[:,search_space[m2]].columns]))