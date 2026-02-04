from. treegrad import treegrad
import numpy as np


def treegrad_ranker(model, x, class_index, optimizer, lr, T_max, 
                    beta1=0.9, beta2=0.999, epsilon=1e-8): # for Adam
    assert optimizer in ['GA', 'Adam']
    
    z = np.full(len(x), 0.5, dtype=np.float64)
    attr_scores = np.zeros_like(z, dtype=np.float64)
    
    if optimizer == 'GA':
        for T in range(T_max):       
            gradient = treegrad(model, x, z, class_index) / 2
            gradient += treegrad(model, x, 1-z, class_index) / 2
         
            attr_scores *= T / (T + 1)
            attr_scores += gradient / (T + 1)
            
            z += lr * gradient 
            z = np.clip(z, 0, 1)
    else:
        m = np.zeros_like(z, dtype=np.float64)
        v = np.zeros_like(z, dtype=np.float64)
        for T in range(T_max):       
            gradient = treegrad(model, x, z, class_index) / 2
            gradient += treegrad(model, x, 1-z, class_index) / 2
                
            m = beta1*m + (1-beta1)*gradient
            v = beta2*v + (1-beta2)*np.square(gradient)
            
            m_cur = m / (1-beta1**(T+1))
            v_cur = v / (1-beta2**(T+1))
        
            attr_scores *= T / (T + 1)
            attr_scores += gradient / (T + 1)
            
            z += lr * (m_cur / (v_cur**0.5 + epsilon))
            z = np.clip(z, 0, 1)
            
    return attr_scores