import numpy as np
from collections import defaultdict


def treegrad(model, x, z, class_index=None):
    # if class_index is None, model would be treated as regression trees
    n_players = len(x)
    if hasattr(model, 'estimators_'):
        # for models trained using sklearn.ensemble.GradientBoostingClassifier/GradientBoostingRegressor
        shape = np.shape(model.estimators_)
        result = np.empty((shape[0], n_players), dtype=np.float64)
        for i, stage in enumerate(model.estimators_):
            if shape[1] == 1:
                result[i] = treegrad_(stage[0].tree_, x, z, 0)
            else:
                assert class_index is not None
                result[i] = treegrad_(stage[class_index].tree_, x, z, 0)
            
        outcome = model.learning_rate * result.sum(axis=0)
        if shape[1] == 1 and class_index == 0:
            outcome = -outcome           
    else:
        # for models trained using sklearn.tree.DecisionTreeClassifier/DecisionTreeRegressor
        outcome = treegrad_(model.tree_, x, z, class_index or 0)
        
    return outcome



def treegrad_(tree, x, z, value_index):
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
            s = np.float64(1)
        
        n_samples_current = n_node_samples[node]
        gamma = n_samples_parent / n_samples_current * activation
        z_current = z[feature_parent]
        
        check = 1 - z_current + z_current * gamma
        if check == 0:
            if len(features_seen[feature_parent]):
                gamma_ancestor = gammas[features_seen[feature_parent][-1]]
                s /= 1 - z_current + z_current * gamma_ancestor
                
            left, right = children_left[node], children_right[node]
            if left == right:      
                s *= value[node] * quotient[node]
            else:               
                feature_current = feature[node]
                if x[feature_current] <= threshold[node]:
                    activation_left, activation_right = 1, 0
                else:
                    activation_left, activation_right = 0, 1         
                s_a = traverse_zero(left, n_samples_current, feature_current, activation_left, s, feature_parent)
                s_b = traverse_zero(right, n_samples_current, feature_current, activation_right, s, feature_parent)
                s = s_a + s_b
            
            gradient[feature_parent] -= s
            return 0
        else:        
            if len(features_seen[feature_parent]):
                gamma_ancestor = gammas[features_seen[feature_parent][-1]]
                s /= 1 - z_current + z_current * gamma_ancestor
                gamma *= gamma_ancestor
                s *= 1 - z_current + z_current * gamma
            else:
                s *= 1 - z_current + z_current * gamma
            
            left, right = children_left[node], children_right[node]
            if left == right:     
                s *= value[node] * quotient[node]
            else:
                gammas[node] = gamma
                features_seen[feature_parent].append(node)
                ss[node] = np.float64(0)
                
                feature_current = feature[node]
                if x[feature_current] <= threshold[node]:
                    activation_left, activation_right = 1, 0
                else:
                    activation_left, activation_right = 0, 1         
                s_a = traverse(left, n_samples_current, feature_current, activation_left, s)
                s_b = traverse(right, n_samples_current, feature_current, activation_right, s)
                s = s_a + s_b
                
                features_seen[feature_parent].pop()
                
            if len(features_seen[feature_parent]):
                node_ancestor = features_seen[feature_parent][-1]
                ss[node_ancestor] -= s      
            
            if left == right:
                s_current = s
            else:
                s_current = ss.pop(node)
                s_current += s
            s_current /= 1 - z_current + z_current * gamma
            gradient[feature_parent] += (gamma - 1) * s_current
        
        return s
    
    
    def traverse_zero(node, n_samples_parent, feature_parent, activation, s, feature_zero):
        n_samples_current = n_node_samples[node]
        if feature_parent != feature_zero:
            gamma = n_samples_parent / n_samples_current * activation            
            z_current = z[feature_parent]
            
            check = 1 - z_current + z_current * gamma
            if check == 0:
                return 0
                       
            if len(features_seen[feature_parent]):
                gamma_ancestor = gammas[features_seen[feature_parent][-1]]
                s /= 1 - z_current + z_current * gamma_ancestor
                gamma *= gamma_ancestor
                s *= 1 - z_current + z_current * gamma
            else:
                s *= 1 - z_current + z_current * gamma
        
        left, right = children_left[node], children_right[node]
        if left == right:      
            s *= value[node] * quotient[node]
        else:
            if feature_parent != feature_zero:
                gammas[node] = gamma
                features_seen[feature_parent].append(node)
            
            feature_current = feature[node]
            if x[feature_current] <= threshold[node]:
                activation_left, activation_right = 1, 0
            else:
                activation_left, activation_right = 0, 1         
            s_a = traverse_zero(left, n_samples_current, feature_current, activation_left, s, feature_zero)
            s_b = traverse_zero(right, n_samples_current, feature_current, activation_right, s, feature_zero)
            s = s_a + s_b
            
            if feature_parent != feature_zero: 
                features_seen[feature_parent].pop()
            
        return s
    
        
    gradient = np.zeros(len(x), dtype=np.float64)
    left, right = children_left[0], children_right[0]
    feature_root = feature[0]
    n_samples_root = n_node_samples[0]
    if x[feature_root] <= threshold[0]:
        activation_left, activation_right = 1, 0
    else:
        activation_left, activation_right = 0, 1
    traverse(left, n_samples_root, feature_root, activation_left)
    traverse(right, n_samples_root, feature_root, activation_right)
    
    return gradient