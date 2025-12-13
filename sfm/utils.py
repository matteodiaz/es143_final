import numpy as np

# helper functions from previous classes/labs

def in2hom(X):
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)

def hom2in(Xh):
    return Xh[:, :2] / Xh[:, 2:]

def normalizing_transform(X):
    D = X.shape[1]
    centroid = X.mean(axis=0, keepdims=True)
    denom = np.mean(np.sqrt(np.sum((X - centroid)**2, axis=1)))
    s = np.sqrt(D) / denom
    t = -s * centroid[0, :]
    t = np.concatenate((t, np.array([1])))
    return np.concatenate((s*np.eye(D+1, D+1)[:, 0:D], np.expand_dims(t, 1)), axis=1)
