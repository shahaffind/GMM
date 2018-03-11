import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def calc_expectation(point, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-np.power(point - mu, 2) / (2 * np.power(sigma, 2)))


k_clusters = 3
points_per_cluster = 30

examples_cluster = np.array([np.random.randn(points_per_cluster) * 1 + 1,
                             np.random.randn(points_per_cluster) * 2 - 5,
                             np.random.randn(points_per_cluster) * 1.5 + 6])

X = np.ravel(examples_cluster)

m_points = len(X)

mu = np.random.choice(X, k_clusters)
sigma = np.ones(k_clusters) * np.sqrt(np.var(X))

phi = np.ones(k_clusters) / k_clusters

g = np.zeros([m_points, k_clusters])
W = np.zeros([m_points, k_clusters])

for it in xrange(0, 1000):
    # E - step:
    for i in xrange(0, m_points):
        for j in xrange(0, k_clusters):
            g[i][j] = calc_expectation(X[i], mu[j], sigma[j]) * phi[j]
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

plt.scatter(X, np.zeros(m_points))
for k in xrange(0, k_clusters):
    gaussian = np.linspace(mu[k] - 3*sigma[k], mu[k] + 3*sigma[k], 100)
    plt.plot(gaussian, mlab.normpdf(gaussian, mu[k], sigma[k]))
plt.show()

print mu, sigma