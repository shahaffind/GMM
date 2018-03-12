import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats


def run_em_nd(X, k_clusters):
    m = len(X)
    start_points = np.random.randint(0, len(X), k_clusters)
    mu = X[start_points]

    start_cov = np.cov(X, rowvar=False)
    sigma = np.array([start_cov.copy()] * k_clusters)

    phi = np.ones(k_clusters) / k_clusters

    g = np.zeros([m, k_clusters])
    W = np.zeros([m, k_clusters])

    for it in xrange(0, 1000):
        # E - step:
        for i in xrange(0, m):
            for j in xrange(0, k_clusters):
                g[i][j] = stats.multivariate_normal.pdf(X[i], mu[j], sigma[j]) * phi[j]
            sum_W = g[i].sum()
            W[i] = g[i] / sum_W

        # M - step
        prev_mu = mu.copy()
        for j in xrange(0, k_clusters):
            phi[j] = W[:, j].mean()
            mu[j] = np.dot(W[:, j], X) / W[:, j].sum()

            sigma[j] = np.cov(X, rowvar=False, aweights=W[:, j])

        if np.allclose(prev_mu, mu):
            print "%d iterations" % it
            break

    return mu, sigma


if __name__ == "__main__":

    clusters_examples = [np.random.multivariate_normal([0, 0], [[6, 0.1], [0.1, 1]], 30),
                         np.random.multivariate_normal([5, 5], [[2, 0.7], [0.7, 1]], 30),
                         np.random.multivariate_normal([-5, 3], [[1, 0.4], [0.4, 3]], 30),
                         np.random.multivariate_normal([-5, -7], [[5, 1], [1, 2]], 40)]

    k_clusters = len(clusters_examples)
    X = np.concatenate(clusters_examples)

    mu, sigma = run_em_nd(X, k_clusters)

    plt.scatter(X[:, 0], X[:, 1])

    delta = 0.025
    x = np.arange(min(X[:, 0]) - 1, max(X[:, 0]) + 1, delta)
    y = np.arange(min(X[:, 1]) - 1, max(X[:, 1]) + 1, delta)
    X, Y = np.meshgrid(x, y)

    for k in xrange(0, k_clusters):
        plt.scatter(mu[:, 0], mu[:, 1], c='r', marker='x')

        Z = mlab.bivariate_normal(X, Y, sigma[k][0, 0], sigma[k][1, 1], mu[k][0], mu[k][1], sigma[k][0, 1])
        plt.contour(X, Y, Z)

        print "cluster %d:" % k
        print "mu:\n", mu[k]
        print "cov:\n", sigma[k], '\n'

    plt.show()
