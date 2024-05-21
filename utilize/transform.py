import numpy as np
    
def select_target_labels(X, y, M, target_labels, label_names, drop_all_zero = True):
    li = []
    for i, label_name in enumerate(label_names):
        if label_name in target_labels:
            li.append(i)

    sample_index = (np.sum(y[:, li], axis = 1) != 0)
    if drop_all_zero:
        return X[sample_index, :], y[sample_index, :][:, li], M[sample_index, :][:, li]
    else:
        return X[:,:], y[:, li], M[:, li]


if __name__ == 'main': 

    None