import numpy as np
import matplotlib.pyplot as plt
import torch

def add_noise(X, amount=1):
    mean_col_std = torch.mean(torch.std(X,dim=0)).item()
    noise_std = mean_col_std / 50 * amount
    return X + torch.normal(0, std=noise_std, size=X.size())

def plot(X):
    _, m = X.size()
    plt.figure(figsize=(8,6), dpi=100)
    if m == 1:
        plt.plot(X)
    elif m == 2:
        plt.plot(X[:,0], X[:,1])
    elif m == 3:
        ax = plt.axes(projection='3d')
        ax.plot3D(X[:,0], X[:,1], X[:,2]);

def plot_phase(
    figsize,
    dpi,
    scatter,
    marker,
    markersize,
    dim,
    trajectories,
    vector_field
):
    plt.figure(figsize=figsize, dpi=dpi)
    if dim == 2:
        if vector_field is not None:
            x,y = vector_field.x[:,0], vector_field.x[:,1]
            dx,dy = vector_field.x_dot[:,0], vector_field.x_dot[:,1]
    elif dim == 3:
        ax = plt.axes(projection='3d')
        if vector_field is not None:
            x,y,z = vector_field.x[:,0], vector_field.x[:,1], vector_field.x[:,2]
            dx,dy,dz = vector_field.x_dot[:,0], vector_field.x_dot[:,1], vector_field.x_dot[:,2]
    else:
        print('must have dim=2 or dim=3 to plot phase portrait')
        return
    for traj in trajectories:
        X = traj.x
        if dim == 2:
            plt.plot(X[:,0], X[:,1])
        elif dim == 3:
            ax.plot3D(X[:,0], X[:,1], X[:,2]);
    if vector_field is not None:
        color=(.2,.9,.9,.7)
        if dim == 2:
            plt.quiver(x,y,dx,dy,color=color)
            plt.axis('scaled')
        elif dim == 3:
            plt.quiver(x,y,z,dx,dy,dz,color=color,length=2,normalize=True)


    
    
    
    


def differentiate(X,t):
    n,m = X.size()
    T   = torch.stack(tuple(t for _ in range(m)), axis=1)
    dX  = torch.gradient(X,dim=0)[0]
    dT  = torch.gradient(T,dim=0)[0]
    return dX/dT


def differentiate_1(X,t):
    n,m = X.size()
    dt = (t[1] - t[0]).item()
    X_dot_0   = (X[1] - X[0]) / dt                     
    X_dot_mid = (X[1:-1] - X[0:-2]) / (2*dt)           
    X_dot_f   = (X[-1] - X[-2]) / dt                   
    return torch.vstack((X_dot_0, X_dot_mid, X_dot_f))   
    
    


    
        
