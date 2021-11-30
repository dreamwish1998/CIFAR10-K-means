import numpy as np
import math

def vectorDistance(v1, v2):
    """
    this function calculates de euclidean distance between two
    vectors.
    """
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) ** 2  # sum the distance
    return sum ** 0.5

def compute_Si(i, x, pre, clusters, nc):
    """
    this function calculates SI.
    """
    s = 0
    sel = np.where(np.array(pre)==i)[0].tolist()  # same prediction
    norm_c = len(sel)  # the total sum to division
    for t in range(len(sel)):
        s += vectorDistance(x[sel[t]], clusters[i])  # sum the distance
    return s / norm_c

def compute_Rij(i, j, x, pre, clusters, nc):
    """
    this function calculates Rij.
    """
    Mij = vectorDistance(clusters[i], clusters[j])  # sum the distance
    Rij = (compute_Si(i, x, pre, clusters, nc) + compute_Si(j, x, pre, clusters, nc)) / Mij  # compute the Rij
    return Rij

def compute_Di(i, x, pre, clusters, nc):
    """
    this function calculates DI.
    """
    list_r = []
    for j in range(nc):
        if i != j:
            temp = compute_Rij(i, j, x, pre, clusters, nc)
            list_r.append(temp)
    return max(list_r)  # get the DI

def compute_DB_index(x, pre, clusters, nc):
    """
    this function calculates DBI.
    """
    sigma_R = 0.0
    for i in range(nc):
        print("calculating {0} cluster".format(i))
        sigma_R = sigma_R + compute_Di(i, x, pre, clusters, nc)  # sum the DBI
    DB_index = float(sigma_R) / float(nc)
    return DB_index

def compute_SSE(x, pre, clusters, nc):
    """
    this function calculates SSE.
    """
    sigma_R = 0.0
    for i in range(nc):
        sigma_R = sigma_R + compute_Si(i, x, pre, clusters, nc)  # sum the SSE
    return sigma_R

