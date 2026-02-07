from scipy import special
import numpy as np
from collections import defaultdict
import numbers



def treeprob_worsetime(model, x, semivalue, class_index=None, test=False):
    # if class_index is None, model would be treated as regression trees
    n_players = len(x)    
    if hasattr(model, 'estimators_'):
        # for models trained using sklearn.ensemble.GradientBoostingClassifier/GradientBoostingRegressor
        shape = np.shape(model.estimators_)
        result = np.empty((shape[0], n_players), dtype=np.float64)
        for i, stage in enumerate(model.estimators_):
            if shape[1] == 1:
                result[i] = treeprob_(stage[0].tree_, x, semivalue, 0, test)
            else:
                result[i] = treeprob_(stage[class_index].tree_, x, semivalue, 0, test)
            
        outcome = model.learning_rate * result.sum(axis=0)
        if shape[1] == 1 and class_index == 0:
            outcome = -outcome           
    else:
        # for models trained using sklearn.tree.DecisionTreeClassifier/DecisionTreeRegressor
        outcome = treeprob_(model.tree_, x, semivalue, class_index or 0, test)
        
    return outcome
    
    
    

def treeprob_(tree, x, semivalue, value_index, test): 
    if test:
        D = tree.max_depth
    else:
        D = min(tree.max_depth, len(x))
    
    height = retrieve_height(tree)  
    nodes = np.exp(1j * 2 * np.pi / D * np.arange(D), dtype=np.complex128)
    weights = np.zeros((D, D), dtype=np.complex128)
    for i in range(D):
        tmp = compute_weights(semivalue, i+1)
        weights[i, :i+1] = np.linalg.pinv(np.vander(nodes[:i+1], increasing=True).T).dot(tmp)

    
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    n_node_samples = tree.n_node_samples
    value = tree.value[:, 0, value_index]
    quotient = n_node_samples / n_node_samples[0]     
    M_scaling = np.vander(nodes+1, increasing=True)
     
    features_seen = defaultdict(list)
    gammas = np.full(len(value), -1, dtype=np.float64)
    polynomials = dict()
    
    def traverse(node, n_samples_parent, feature_parent, activation, p=None):
        if p is None:
            p = np.ones(D, dtype=np.complex128)
        
        n_samples_current = n_node_samples[node]
        gamma = n_samples_parent / n_samples_current * activation
        if len(features_seen[feature_parent]):
            gamma_ancestor = gammas[features_seen[feature_parent][-1]]
            p /= 1 + gamma_ancestor * nodes
            gamma *= gamma_ancestor
            p *= 1 + gamma * nodes
        else:
            p *= 1 + gamma * nodes
        
        left, right = children_left[node], children_right[node]
        if left == right:                
            p *= value[node] * quotient[node]
        else:
            gammas[node] = gamma
            features_seen[feature_parent].append(node)
            polynomials[node] = np.zeros(D, dtype=np.complex128)
            
            feature_current = feature[node]
            if x[feature_current] <= threshold[node]:
                activation_left, activation_right = 1, 0
            else:
                activation_left, activation_right = 0, 1         
            p_a = traverse(left, n_samples_current, feature_current, activation_left, p.copy())
            p_b = traverse(right, n_samples_current, feature_current, activation_right, p.copy())
            if height[left] > height[right]:
                p_b *= M_scaling[:, height[left] - height[right]]
            elif height[left] < height[right]:
                p_a *= M_scaling[:, height[right] - height[left]]
            p = p_a + p_b
            features_seen[feature_parent].pop()
        
            
        if len(features_seen[feature_parent]):
            node_ancestor = features_seen[feature_parent][-1]
            if height[node] < height[node_ancestor]:
                polynomials[node_ancestor] -= p * M_scaling[:, height[node_ancestor] - height[node]]      
            else:
                polynomials[node_ancestor] -= p      
        
        if left == right:
            p_current = p.copy()
        else:
            p_current = polynomials.pop(node)
            p_current += p
        p_current /= 1 + gamma * nodes
        phi[feature_parent] += (gamma - 1) * np.dot(p_current, weights[height[node]-1]).real
        
        return p
        
    phi = np.zeros(len(x), dtype=np.float64)
    left, right = children_left[0], children_right[0]
    feature_root = feature[0]
    n_samples_root = n_node_samples[0]
    if x[feature_root] <= threshold[0]:
        activation_left, activation_right = 1, 0
    else:
        activation_left, activation_right = 0, 1
    traverse(left, n_samples_root, feature_root, activation_left)
    traverse(right, n_samples_root, feature_root, activation_right)
    
    return phi



def retrieve_height(tree):
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    
    height = np.zeros_like(children_left, dtype=np.int64)
    
    features_seen = defaultdict(list)
    def traverse(node, feature_parent, degree=0):
        if len(features_seen[feature_parent]) == 0:
            degree += 1
            
        left, right = children_left[node], children_right[node]
        if left == right:                
            height[node] = degree
        else:
            features_seen[feature_parent].append(node)
            
            feature_current = feature[node]       
            degree_a = traverse(left,  feature_current, degree)
            degree_b = traverse(right, feature_current, degree)
            degree = max(degree_a, degree_b)
            height[node] = degree
            features_seen[feature_parent].pop()
            
        return degree
    
    left, right = children_left[0], children_right[0]
    feature_root = feature[0]
    traverse(left, feature_root)
    traverse(right, feature_root)
    return height
            


def compute_weights(semivalue, n):
    tmp = np.arange(n, dtype=np.float64)
    if isinstance(semivalue, tuple):
        alpha, beta = semivalue
        assert alpha > 0 and beta > 0
        denominator = special.beta(beta, alpha)
        weights = special.beta(tmp+beta, tmp[::-1]+alpha) / denominator
    else:
        assert isinstance(semivalue, numbers.Number)
        weights = semivalue ** tmp
        weights *= (1-semivalue) ** tmp[::-1]
        
    return weights
    