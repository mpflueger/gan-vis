# Generate multi-modal 2d data

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def sample(mean, std_dev, samples):
    # Get 'samples' samples from a normal distritubtion
    rands = np.random.normal(0, std_dev, samples * len(mean))
    rands = np.reshape(rands, [-1, len(mean)])
    return rands + mean

def sample_multi_mode(modes, samples):
    # Sample a set of 2D datapoints with modes spaced evenly around a unit
    # circle
    std_dev = 0.05

    data = sample([1, 0], std_dev, samples)
    for mode in range(1, modes):
        theta = float(mode) / modes * 2 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        rot = np.matrix([[c, -s], [s, c]])
        mean = (rot * np.matrix([[1], [0]])).transpose().tolist()[0]
        data = np.append(data, sample(mean, std_dev, samples), axis=0)

    return data

if __name__ == "__main__":
    data = sample_multi_mode(3, 10000)

    np.savetxt('data.csv', data)

    H, x_edges, y_edges = np.histogram2d(
        data[:,0], data[:,1], bins=200, normed=True)
    H = H.transpose()
    plt.imshow(H, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    plt.show()

