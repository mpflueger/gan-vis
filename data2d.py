import numpy as np

class Data2D(object):
    def __init__(self, datafile):
        self.data = np.loadtxt(datafile)
        self.data_size = self.data.shape[0]

    def get_batch(self, n):
        # Return a random batch of n items
        return self.data[np.random.choice(self.data_size, n, replace=False)]

    def get_data(self):
        return self.data
