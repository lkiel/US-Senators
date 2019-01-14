import scipy
import numpy as np
from pyunlocbox import functions, solvers
from scipy import sparse
from utils import accuracy

def graph_pnorm_interpolation(gradient, P, x0=None, p=1., **kwargs):
    r"""
    Solve an interpolation problem via gradient p-norm minimization.

    A signal :math:`x` is estimated from its measurements :math:`y = A(x)` by solving
    :math:`\text{arg}\underset{z \in \mathbb{R}^n}{\min}
    \| \nabla_G z \|_p^p \text{ subject to } Az = y` 
    via a primal-dual, forward-backward-forward algorithm.

    Parameters
    ----------
    gradient : array_like
        A matrix representing the graph gradient operator
    P : callable
        Orthogonal projection operator mapping points in :math:`z \in \mathbb{R}^n` 
        onto the set satisfying :math:`A P(z) = A z`.
    x0 : array_like, optional
        Initial point of the iteration. Must be of dimension n.
        (Default is `numpy.random.randn(n)`)
    p : {1., 2.}
    kwargs :
        Additional solver parameters, such as maximum number of iterations
        (maxit), relative tolerance on the objective (rtol), and verbosity
        level (verbosity). See :func:`pyunlocbox.solvers.solve` for the full
        list of options.

    Returns
    -------
    x : array_like
        The solution to the optimization problem.

    """
    
    grad = lambda z: gradient.dot(z)
    div = lambda z: gradient.transpose().dot(z)

    # Indicator function of the set satisfying :math:`y = A(z)`
    f = functions.func()
    f._eval = lambda z: 0
    f._prox = lambda z, gamma: P(z)

    # :math:`\ell_1` norm of the dual variable :math:`d = \nabla_G z`
    g = functions.func()
    g._eval = lambda z: np.sum(np.abs(grad(z)))
    g._prox = lambda d, gamma: functions._soft_threshold(d, gamma)

    # :math:`\ell_2` norm of the gradient (for the smooth case)
    h = functions.norm_l2(A=grad, At=div)

    stepsize = (0.9 / (1. + scipy.sparse.linalg.norm(gradient, ord='fro'))) ** p

    solver = solvers.mlfbf(L=grad, Lt=div, step=stepsize)

    if p == 1.:
        problem = solvers.solve([f, g, functions.dummy()], x0=x0, solver=solver, **kwargs)
        return problem['sol']
    if p == 2.:
        problem = solvers.solve([f, functions.dummy(), h], x0=x0, solver=solver, **kwargs)
        return problem['sol']
    else:
        return x0
        
        
def P(a):
    b = labels_bin * w + (1-w) * a
    return b
    
    
def P_wrapper(mask, labels_bin):
    return lambda a: labels_bin * mask + (1-mask) * a
    
    
def get_thresholded_values(v,threshold, epsilon = 1e-2):
    v_bin = v.copy()
    v_bin[v_bin > threshold + epsilon] = 1
    v_bin[v_bin < threshold - epsilon] = -1
    v_bin[np.logical_and((v_bin <= threshold + epsilon) , (v_bin >= threshold - epsilon))] = 0
    return v_bin
    
    
def get_mask(n,m):
    idx = np.random.choice(np.arange(n), m, replace=False)
    w = np.zeros(n)
    w[idx] = 1
    return w

    
def reconstruct_signal(G, mask, labels_bin, threshold = 0, number_of_trials=100, verbose = 'NONE'):

    sols = []
    
    for _ in range(number_of_trials):
        sols.append(graph_pnorm_interpolation(
            G.D,
            P_wrapper(mask, labels_bin),
            np.random.normal(loc=0, scale=0.1,size=len(labels_bin)), 
            1.,
            verbosity=verbose
        ))
        
    sols = np.mean(sols,axis=0)
    return sols, get_thresholded_values(sols, threshold)
    
    
def compare_outcome(pred, labels):
    true_results = dict(zip(*np.unique(labels.astype(int), return_counts=True)))
    true_outcome = true_results.get(1,0) > true_results.get(-1,0)
    
    pred_results = dict(zip(*np.unique(pred.astype(int), return_counts=True)))
    pred_outcome = pred_results.get(1,0) > pred_results.get(-1,0)
    
    print("True: "+str(true_results) + " Pred: " + str(pred_results) + " Correct: " +str(pred_outcome == true_outcome))
    
    return pred_outcome == true_outcome
    
    
def predict_and_compare(G, df, senator_selection):
    individual_accuracies = []
    outcome_comparison = []

    sencount, votecount = df.shape
    
    for i in range(votecount):
        labels_bin = get_thresholded_values(df.values[:,i], 0.0)
        mask = np.zeros(sencount)
        mask[senator_selection] = 1
        _, pred = reconstruct_signal(G, mask, labels_bin, number_of_trials=100)
        individual_accuracies.append(accuracy(pred,labels_bin))
        outcome_comparison.append(compare_outcome(pred, labels_bin))
        
    print("Outcome accuracy: " + str(np.mean(outcome_comparison)))
    
    return individual_accuracies, outcome_comparison