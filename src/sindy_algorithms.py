import torch

# Solve the sparse optimization problem min( ||Ax - b|| + ||x||1 )
class STLSQ:
    def __init__(self, threshold, reg=0):
        self.threshold = threshold
        self.k = reg
        self.max_iter = 20

    def fit(self, X, X_dot, lib):
        A, b = lib.theta(X), X_dot 
        x = None
        for _ in range(self.max_iter):
            x = self.stlsq(A,b,x)
        return x

    def stlsq(self, A, b, x):
        if x is None:
            return self.lsq(A,b)
        _, num_columns = b.shape
        small = torch.abs(x) < self.threshold
        x[small] = 0
        for i in range(num_columns):
            big = ~small[:,i]
            x[big,i] = self.lsq(A[:,big], b[:,i])
        return x

    def lsq(self, A, b):
        _, n = A.shape
        return torch.linalg.inv(A.T @ A + self.k * torch.eye(n)) @ (A.T @ b)

