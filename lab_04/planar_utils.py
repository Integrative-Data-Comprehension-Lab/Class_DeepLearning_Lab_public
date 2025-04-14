import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    return X, Y

def plot_decision_boundary(estimator, X, y=None, *,
                           grid_resolution=1000,
                           eps=1.0,
                           plot_method="contourf",
                           xlabel='x1',
                           ylabel='x2',
                           ax=None,
                           **kwargs):

    # Use current axis if none provided
    if ax is None:
        ax = plt.gca()
        
    # Define the grid boundaries with padding
    x0, x1 = X[:, 0], X[:, 1]
    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps
    
    # Create a grid with np.linspace for a given resolution
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution)
    )
    
    # Flatten the grid to pass through the estimator
    X_grid = np.c_[xx0.ravel(), xx1.ravel()]
    
    # Get the estimator's response; support both callable lambdas and objects with a forward() method.
    if hasattr(estimator, "forward"):
        response = estimator.forward(X_grid)
    else:
        response = estimator(X_grid)
    
    # Reshape the response to match the grid shape
    Z = response.reshape(xx0.shape)
    
    # Plot the decision boundary using the selected plot method
    if plot_method == "contourf":
        ax.contourf(xx0, xx1, Z, cmap=plt.cm.Spectral, **kwargs)
    elif plot_method == "contour":
        ax.contour(xx0, xx1, Z, cmap=plt.cm.Spectral, **kwargs)
    else:
        raise ValueError(f"Unknown plot_method: {plot_method}. Use 'contourf' or 'contour'.")
    
    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Overlay the training data if labels are provided
    if y is not None:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
    
    return ax

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure