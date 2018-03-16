import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def calc_log_weighted_prob(X, mu, Q, phi):
    mean_diff = X - mu
    dim = len(X)

    l_k = - 0.5 * np.dot(np.dot(mean_diff, Q), mean_diff)
    w_k = phi / np.sqrt(np.power(2 * np.pi, dim) * (1 / np.linalg.det(Q)))
    log_weighted_prob = l_k + np.log(w_k)

    return log_weighted_prob


def run_em_nd(X, k_clusters):
    m = len(X)
    start_points = np.random.randint(0, len(X), k_clusters)
    mu = X[start_points]

    start_cov = np.cov(X, rowvar=False)
    sigma = np.array([start_cov.copy()] * k_clusters)
    start_Q = np.linalg.inv(start_cov)
    Q = np.array([start_Q] * k_clusters)

    phi = np.ones(k_clusters) / k_clusters

    weighted_log_prob = np.zeros(k_clusters)
    W = np.zeros([m, k_clusters])

    for it in xrange(0, 1000):
        # E - step:
        for i in xrange(0, m):
            for j in xrange(0, k_clusters):
                weighted_log_prob[j] = calc_log_weighted_prob(X[i], mu[j], Q[j], phi[j])
            max_log_prob = np.max(weighted_log_prob)
            shifted_probs = np.exp(weighted_log_prob - max_log_prob)
            W[i] = shifted_probs / shifted_probs.sum()

        # M - step
        prev_mu = mu.copy()
        for j in xrange(0, k_clusters):
            phi[j] = W[:, j].mean()
            mu[j] = np.dot(W[:, j], X) / W[:, j].sum()

            sigma[j] = np.cov(X, rowvar=False, aweights=W[:, j])
            Q[j] = np.linalg.inv(sigma[j])

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
