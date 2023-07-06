import numpy as np
import random, time


def eu_dist(a, b):
    return np.linalg.norm(a - b)


class Kmeans:
    def __init__(self, features, cls=16, max_epoch=10) -> None:
        self.samples = features
        self.cls = cls
        self.max_epoch = max_epoch
        self.centroids = np.zeros((self.cls, self.samples.shape[-1]))

        self.random_centroids()
        self.clustering()

    def random_centroids(self):
        index = random.sample(range(0, self.samples.shape[0]), self.cls)
        self.centroids = self.samples[index]

    def clustering(self):
        num_samples = self.samples.shape[0]
        cluster_tag = np.zeros(num_samples)
        temp_centroids = np.zeros(self.centroids.shape)

        epoch = 0

        while True:
            epoch += 1

            for i in range(num_samples):
                min_dist = np.inf
                for j in range(self.cls):
                    dist = eu_dist(self.samples[i], self.centroids[j])
                    if dist < min_dist:
                        min_dist = dist
                        cluster_tag[i] = j

            for k in range(self.cls):
                cls_samples = self.samples[cluster_tag == k]
                temp_centroids[k] = np.mean(cls_samples, axis=0)

            if np.equal(temp_centroids, self.centroids).all() or epoch == self.max_epoch:
                self.centroids = temp_centroids.copy()
                break

            self.centroids = temp_centroids.copy()

    def calc_centroid_index(self, vec_matrix):
        if len(vec_matrix.shape) == 1:
            index = 0
            min_dist = np.inf
            for k in range(self.cls):
                dist = eu_dist(vec_matrix, self.centroids[k])
                if dist < min_dist:
                    min_dist = dist
                    index = k
            return index
        else:
            indices = []
            for vec in vec_matrix:
                index = 0
                min_dist = np.inf
                for k in range(self.cls):
                    dist = eu_dist(vec, self.centroids[k])
                    if dist < min_dist:
                        min_dist = dist
                        index = k
                indices.append(index)
            return indices


class Quantizer:
    def __init__(self, input_matrix, weight_matrix, j_indices=None, num_space=None, bias=None, max_epoch=10):
        self.kmeans = []
        self.j_indices = j_indices
        self.num_space = num_space if num_space else len(self.j_indices)
        self.new_weight = weight_matrix[[j for index in self.j_indices for j in index]]
        self.bias = bias
        self.table = []

        for i in range(self.num_space):
            kmeans = Kmeans(input_matrix[:, self.j_indices[i]], max_epoch=max_epoch)
            self.kmeans.append(kmeans)
            self.table.append(np.dot(kmeans.centroids, weight_matrix[self.j_indices[i]]))

        self.table = np.stack(self.table, axis=0)
        if self.bias is not None:
            self.table[0] += self.bias

    def calc(self, inputs):
        result = []
        if len(inputs.shape) == 1:
            for i in range(self.num_space):
                index = self.kmeans[i].calc_centroid_index(inputs[self.j_indices[i]])
                result.append(self.table[i, index])
        else:
            for i in range(self.num_space):
                index = self.kmeans[i].calc_centroid_index(inputs[:, self.j_indices[i]])
                result.append(self.table[i, index])
        return np.stack(result, axis=0).sum(axis=0)

    def save(self):
        dic = {}
        for i in range(self.num_space):
            dic[i] = (self.kmeans[i].centroids, self.j_indices[i])
        np.save('pq_model.npy', dic)
        np.save('q_lookup_table.npy', self.table)


if __name__ == '__main__':
    print('{:-^50s}'.format(' Loading Data '))
    train_A = np.load('train_matrices.npy')
    test_A = np.load('test_matrices.npy')
    weight_B = np.load('weight.npy')
    bias_B = np.load('bias.npy')

    j_indices = [list(range(i * 32, i * 32 + 32)) for i in range(16)]

    max_epoch = 200

    print('{:-^50s}'.format(' Training '))
    start_time = time.time()
    quantizer = Quantizer(train_A, weight_B, j_indices, 16, bias_B, max_epoch)
    end_time = time.time()
    print('training cost time: %.3f' % (end_time - start_time))

    print('{:-^50s}'.format(' Approximating '))
    start_time = time.time()
    approx_test_A = quantizer.calc(test_A)
    end_time = time.time()
    print('approximate matrix shape: {}'.format(approx_test_A.shape))
    print('approximating cost time: %.3f' % (end_time - start_time))

    print('{:-^50s}'.format(' Saving '))
    np.save('q_approx_test_matrices.npy', approx_test_A)
    quantizer.save()
