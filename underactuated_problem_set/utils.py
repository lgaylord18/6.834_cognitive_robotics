import numpy as np
import scipy.linalg

def lqr(A, B):
    Q = np.identity(3)
    R = np.identity(2)
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    eigvals, eigvecs = scipy.linalg.eig(A-B*K)
    return K, X, eigvals
