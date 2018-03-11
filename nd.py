import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats


def simulate_nd_gmm(X, k_clusters):
    m = len(X)
    start_points = np.random.randint(0, len(X), k_clusters)
    mu = X[start_points]

    start_cov = np.cov(X, rowvar=False)
    sigma = np.array([copy.deepcopy(start_cov)] * k_clusters)

    phi = np.ones(k_clusters) / k_clusters

    g = np.zeros([m, k_clusters])
    W = np.zeros([m, k_clusters])

    for it in xrange(0, 1000):
        # E - step:
        for i in xrange(0, m):
            for j in xrange(0, k_clusters):
                g[i][j] = stats.multivariate_normal.pdf(X[i], mu[j], sigma[j]) * phi[j]
            sum_W = np.sum(g[i])
            W[i] = g[i] / sum_W

        # M - step
        prev_mu = copy.deepcopy(mu)
        for j in xrange(0, k_clusters):
            phi[j] = np.mean(W[:, j])
            mu[j] = np.dot(W[:, j], X) / np.sum(W[:, j])

            sigma[j] = np.cov(X, rowvar=False, aweights=W[:, j])

        if np.sum(prev_mu - mu) == 0:
            print "%d iterations" % it
            break

    return mu, sigma


if __name__ == "__main__":

    examples_cluster = np.array([np.random.multivariate_normal([0, 0], np.array([[6, 0.1], [0.1, 1]]), 30),
                                 np.random.multivariate_normal([5, 5], np.array([[2, 0.7], [0.7, 1]]), 30),
                                 np.random.multivariate_normal([-5, 3], np.array([[1, 0.4], [0.4, 3]]), 30)])

    k_clusters = len(examples_cluster)
    X = examples_cluster.reshape(-1, 2)

    mu, sigma = simulate_nd_gmm(X, k_clusters)

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
        print "mu:\n\t", mu[k]
        print "cov:\n\t", sigma[k]
        print "\n"

    plt.show()
