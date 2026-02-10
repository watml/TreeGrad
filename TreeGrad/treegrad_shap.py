import numpy as np
from collections import defaultdict



def treegrad_shap(model, x, semivalue, class_index=None, test=False):
    # if class_index is None, model would be treated as regression trees
    n_players = len(x)
    if hasattr(model, 'estimators_'):
        # for models trained using sklearn.ensemble.GradientBoostingClassifier/GradientBoostingRegressor
        shape = np.shape(model.estimators_)
        result = np.empty((shape[0], n_players), dtype=np.float64)
        for i, stage in enumerate(model.estimators_):
            if shape[1] == 1:
                result[i] = treegrad_shap_(stage[0].tree_, x, semivalue, 0, test)
            else:
                assert class_index is not None
                result[i] = treegrad_shap_(stage[class_index].tree_, x, semivalue, 0, test)
            
        outcome = model.learning_rate * result.sum(axis=0)
        if shape[1] == 1 and class_index == 0:
            outcome = -outcome           
    else:
        # for models trained using sklearn.tree.DecisionTreeClassifier/DecisionTreeRegressor
        outcome = treegrad_shap_(model.tree_, x, semivalue, class_index or 0, test)
        
    return outcome



def treegrad_shap_(tree, x, semivalue, value_index, test):
    if test:
        D = tree.max_depth
    else:
        D = min(tree.max_depth, len(x))
    
    alpha, beta = semivalue
    D += alpha + beta - 2
    n_points = -(-D // 2)
    points, weights = np.polynomial.legendre.leggauss(n_points)
    points += 1
    points /= 2
    weights /= 2
    
    tmp_alpha = np.arange(1, alpha, dtype=np.float64)
    tmp_beta = np.arange(1, beta, dtype=np.float64)
    tmp = np.arange(1, alpha+beta, dtype=np.float64)[::-1]
    init = ((tmp[:beta-1] * tmp[-1] / tmp_beta)[:,None] * points[None,:]).prod(axis=0)
    init *= ((tmp[beta-1:-1] / tmp_alpha)[:,None] * (1-points)[None,:]).prod(axis=0)
    
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    n_node_samples = tree.n_node_samples
    value = tree.value[:, 0, value_index]
    quotient = n_node_samples / n_node_samples[0]
    
    features_seen = defaultdict(list)
    gammas = np.full(len(value), -1, dtype=np.float64)
    ss = dict()  
 
    
    def traverse(node, n_samples_parent, feature_parent, activation, s=None):
        if s is None:
            s = init.copy()
        
        n_samples_current = n_node_samples[node]
        gamma = n_samples_parent / n_samples_current * activation

        if len(features_seen[feature_parent]):
            gamma_ancestor = gammas[features_seen[feature_parent][-1]]
            s /= 1 - points + points * gamma_ancestor
            gamma *= gamma_ancestor
            s *= 1 - points + points * gamma
        else:
            s *= 1 - points + points * gamma
        
        left, right = children_left[node], children_right[node]
        if left == right:
            s *= value[node] * quotient[node]
        else:
            gammas[node] = gamma
            features_seen[feature_parent].append(node)
            ss[node] = np.zeros(n_points, dtype=np.float64)
            
            feature_current = feature[node]
            if x[feature_current] <= threshold[node]:
                activation_left, activation_right = 1, 0
            else:
                activation_left, activation_right = 0, 1         
            s_a = traverse(left, n_samples_current, feature_current, activation_left, s.copy())
            s_b = traverse(right, n_samples_current, feature_current, activation_right, s.copy())
            s = s_a + s_b
            
            features_seen[feature_parent].pop()
            
        if len(features_seen[feature_parent]):
            node_ancestor = features_seen[feature_parent][-1]
            ss[node_ancestor] -= s      
        
        if left == right:
            s_current = s.copy()
        else:
            s_current = ss.pop(node)
            s_current += s
        s_current /= 1 - points + points * gamma
        phi[feature_parent] += np.dot((gamma - 1) * s_current, weights)
        
        return s
        
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