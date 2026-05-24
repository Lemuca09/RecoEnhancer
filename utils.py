import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]


def euclidean(a, b):
    return np.linalg.norm(
        np.array(a) - np.array(b)
    )