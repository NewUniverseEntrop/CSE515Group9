import numpy as np
import math

class lshash:
    def __init__(self, L, k):
        self.L = L
        self.k = k
        self.p = []
        self.b = []
        self.mu, self.sigma = 0, math.sqrt(k) # mean and standard deviation
        self.buckets = {}

    def index(self, vectors):
        self.dim = len(vectors[0]) # dimension of the vectors
        self.w = .05 # window
        for _ in range(self.L):
            level = []
            for _ in range(self.k):
                level.append(np.random.normal(self.mu, self.sigma, self.dim))
            self.p.append(level)
            self.b.append(np.random.uniform(0, self.w, self.k))

        for idx, input_point in enumerate(vectors):
            input_point = np.array(input_point)
            for i in range(self.L):
                h = []
                for j in range(self.k):
                    h.append(math.floor((np.dot(self.p[i][j], input_point) + self.b[i][j]) / self.w))
                if (i, tuple(h)) not in self.buckets:
                    self.buckets[(i, tuple(h))] = [idx]
                else:
                    self.buckets[(i, tuple(h))].append(idx)

    def query(self, input_point):
        input_point = np.array(input_point)
        candidates = set()
        overall = 0
        for i in range(self.L):
            h = []
            for j in range(self.k):
                h.append(math.floor((np.dot(self.p[i][j], input_point) + self.b[i][j]) / self.w))
            try:
                for candidate in self.buckets[(i, tuple(h))]:
                    overall += 1
                    candidates.add(candidate)
            except:
                pass
        return candidates, overall, len(candidates)