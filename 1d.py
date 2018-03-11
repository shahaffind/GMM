import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats


def simulate_gmm(X, k_clusters):
    n = len(X)
    mu = np.random.choice(X, k_clusters)
    sigma = np.ones(k_clusters) * np.sqrt(np.var(X))

    phi = np.ones(k_clusters) / k_clusters

    g = np.zeros([n, k_clusters])
    W = np.zeros([n, k_clusters])

    for it in xrange(0, 1000):
        # E - step:
        for i in xrange(0, n):
            for j in xrange(0, k_clusters):
                g[i][j] = stats.multivariate_normal.pdf(X[i], mu[j], sigma[j]) * phi[j]
            sum_W = np.sum(g[i])
            W[i] = g[i] / sum_W

        # M - step
        prev_mu = copy.deepcopy(mu)
        for j in xrange(0, k_clusters):
            phi[j] = np.mean(W[:, j])
            mu[j] = np.dot(W[:, j], X) / np.sum(W[:, j])
            variance = np.dot(W[:, j], np.power(X - mu[j], 2)) / np.sum(W[:, j])
            sigma[j] = np.sqrt(variance)

        if np.sum(prev_mu - mu) == 0:
            print "%d iterations" % it
            break

    plt.scatter(X, np.zeros(n))
    for k in xrange(0, k_clusters):
        gaussian = np.linspace(mu[k] - 3*sigma[k], mu[k] + 3*sigma[k], 100)
        plt.plot(gaussian, mlab.normpdf(gaussian, mu[k], sigma[k]))
        print "cluster %d:\n\tmu: %f\n\tsigma: %f\n" % (k, mu[k], sigma[k])

    plt.show()


if __name__ == "__main__":

    examples_cluster = np.array([np.random.randn(30) * 1 + 0,
                                 np.random.randn(30) * 3 + 9,
                                 np.random.randn(30) * 2 - 6])

    k_clusters = len(examples_cluster)
    X = np.ravel(examples_cluster)

    simulate_gmm(X, k_clusters)
