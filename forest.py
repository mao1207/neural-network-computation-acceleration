import numpy as np
import time
import random

np.random.seed(1)

# from main import HashingTree


class HashingTree:
    def __sort_based_col(self, X, col, num_row=None):
        n = num_row if num_row else X.shape[0]
        sort_elem = X[:, col]
        sort_list = np.concatenate((sort_elem[:, np.newaxis], np.arange(n)[:, np.newaxis]), axis=1).tolist()
        sort_list.sort(key=(lambda x: x[0]))
        return X[np.array(sort_list, dtype='int')[:, 1]]

    def __cumulative_sse(self, X, reverse=False):
        n, d = X.shape
        x = np.flip(X, axis=0) if reverse else X.copy()
        out = [0] * n
        cum_x = [0] * d
        cum_x2 = [0] * d
        for c_index in range(d):
            cum_x[c_index] = x[0, c_index]
            cum_x2[c_index] = x[0, c_index] ** 2
        for r_index in range(1, n):
            for c_index in range(d):
                cum_x[c_index] += x[r_index, c_index]
                cum_x2[c_index] += x[r_index, c_index] ** 2
                out[r_index] += cum_x2[c_index] - cum_x[c_index] ** 2 / (r_index + 1)
        return np.array(out)

    def __optim_split_thr(self, index, b):
        x = np.array(b)
        x = self.__sort_based_col(x, index)
        sse_head = self.__cumulative_sse(x, reverse=False)
        sse_tail = self.__cumulative_sse(x, reverse=True)
        losses = sse_head
        losses[:-1] += np.flip(sse_tail[:-1])
        n = np.argmin(losses)
        return (x[n, index] + x[n + 1, index]) / 2, losses[n]

    def __apply_split(self, index, threshold, bucket):
        b_below, b_above = [], []
        if threshold is not None:
            for vector in bucket:
                if vector[index] <= threshold:
                    b_below.append(vector)
                else:
                    b_above.append(vector)
        return b_below, b_above

    def __next_level(self, buckets, vec_len=4):
        num_b = len(buckets)
        loss_min, j_min, v_min = float('inf'), None, None
        for j in range(vec_len):
            l, v = 0, []
            for i in range(num_b):
                if len(buckets[i]) == 0 or len(buckets[i]) == 1:
                    v.append(None)
                    continue
                v_i, l_i = self.__optim_split_thr(j, buckets[i])
                v.append(v_i)
                l += l_i
            if l < loss_min:
                loss_min = l
                j_min = j
                v_min = v
        next_bucket = []
        for i in range(num_b):
            b_below, b_above = self.__apply_split(j_min, v_min[i], buckets[i])
            next_bucket.append(b_below)
            next_bucket.append(b_above)
        return next_bucket, loss_min, j_min, v_min

    def __init__(self, input_matrix, num_level=None, with_optim=False):
        self.tree_buckets = []
        self.tree_thresholds = []
        self.tree_comp_indices = []
        self.prototypes = []
        self.prototypes_ = []
        self.table = []
        self.num_level = num_level

        self.vec_len = input_matrix.shape[-1]

        _A = [a for a in input_matrix]
        self.tree_buckets.append(_A)
        if self.num_level:
            self.construct(num_level, with_optim=with_optim)

    def construct(self, num_level, with_proto=False, with_optim=False):
        if not self.num_level:
            self.num_level = num_level
        for t in range(self.num_level):
            # next_bucket, _, j_min, v_min = self.__next_level(self.tree_buckets[pow(2, t) - 1: pow(2, t + 1) - 1])
            # self.tree_buckets += next_bucket
            next_bucket, _, j_min, v_min = self.__next_level(self.tree_buckets, vec_len=self.vec_len)
            self.tree_buckets = next_bucket
            self.tree_comp_indices.append(j_min)
            self.tree_thresholds.append(v_min)
        if with_proto:
            self.proto(with_optim)
            # for index, bucket in enumerate(self.tree_buckets[pow(2, self.num_level) - 1:]):
            #     if len(bucket):
            #         self.prototypes.append(np.average(np.array(bucket), axis=0))
            #     else:
            #         level = self.num_level - 1
            #         pre_index = int(index / 2)
            #         while len(self.tree_buckets[pow(2, level) - 1 + pre_index]) == 0:
            #             level -= 1
            #             pre_index = int(pre_index / 2)
            #         self.prototypes.append(np.average(np.array(self.tree_buckets[pow(2, level) - 1 + pre_index]), axis=0))
            # if with_optim:
            #     self.optim_proto()

    def proto(self, with_optim=False):
        # for index, bucket in enumerate(self.tree_buckets[pow(2, self.num_level) - 1:]):
        for index, bucket in enumerate(self.tree_buckets):
            if len(bucket):
                self.prototypes.append(np.average(np.array(bucket), axis=0))
            else:
                # level = self.num_level - 1
                # pre_index = int(index / 2)
                # while len(self.tree_buckets[pow(2, level) - 1 + pre_index]) == 0:
                #     level -= 1
                #     pre_index = int(pre_index / 2)
                # self.prototypes.append(np.average(np.array(self.tree_buckets[pow(2, level) - 1 + pre_index]), axis=0))
                self.prototypes.append(None)
        if with_optim:
            self.optim_proto()

    def optim_proto(self):
        a_tilde = []
        g_matrix = []
        # for index, bucket in enumerate(self.tree_buckets[pow(2, self.num_level) - 1:]):
        for index, bucket in enumerate(self.tree_buckets):
            a_tilde += bucket
            g_matrix += [index] * len(bucket)
        a_tilde = np.array(a_tilde)
        n = a_tilde.shape[0]
        # m = len(self.tree_buckets[pow(2, self.num_level) - 1:])
        m = len(self.tree_buckets)
        g_matrix = np.eye(n, m)[g_matrix]
        prototypes = np.linalg.inv(g_matrix.T @ g_matrix + np.eye(g_matrix.shape[1])) @ g_matrix.T @ a_tilde
        self.prototypes_ = [p for p in prototypes]

    def create_table(self, weight_matrix, with_optim=False):
        if len(self.prototypes) == 0 or (with_optim and len(self.prototypes_) == 0):
            self.table = []
        else:
            if with_optim:
                for prot in self.prototypes_:
                    self.table.append((prot @ weight_matrix).tolist())
            else:
                for prot in self.prototypes:
                    self.table.append((prot @ weight_matrix).tolist())
        return self.table

    def calc_prot_and_index(self, vec_matrix, with_proto=False, with_optim=False):
        if len(vec_matrix.shape) == 1:
            indices = 0
            for t in range(self.num_level):
                threshold = self.tree_thresholds[t][indices]
                temp = 0 if threshold is None or vec_matrix[self.tree_comp_indices[t]] <= threshold else 1
                indices = 2 * indices + temp
            if with_proto:
                prototypes = self.prototypes_[indices] if with_optim else self.prototypes[indices]
                return indices, prototypes
        else:
            indices = []
            for vec in vec_matrix:
                i = 0
                for t in range(self.num_level):
                    threshold = self.tree_thresholds[t][i]
                    temp = 0 if threshold is None or vec[self.tree_comp_indices[t]] <= threshold else 1
                    i = 2 * i + temp
                indices.append(i)
            if with_proto:
                prototypes = [self.prototypes_[i] for i in indices] if with_optim else [self.prototypes[i] for i in indices]
                return indices, prototypes
        return indices, None

    def approximate_calc(self, vec_or_matrix, with_optim=False):
        m = vec_or_matrix.copy() if len(vec_or_matrix.shape) == 2 else vec_or_matrix[np.newaxis, :]
        result = []
        for vec in m:
            _, index = self.calc_prot_and_index(vec, with_optim)
            result.append(self.table[index])
        return np.array(result)

    def get_thr(self):
        return self.tree_thresholds

    def get_indices(self):
        return self.tree_comp_indices

    def get_buckets(self):
        return self.tree_buckets

    def get_prototypes(self):
        return self.prototypes

    def get_prototypes_(self):
        return self.prototypes_


class Forest:
    def __init__(self, input_matrix, weight_matrix, j_indices=None, num_space=None, bias=None):
        self.trees = []
        self.j_indices = j_indices
        self.num_space = num_space if num_space else len(self.j_indices)
        self.new_weight = weight_matrix[[j for index in self.j_indices for j in index]]
        self.num_level = 4
        self.bias = bias

        self.k = pow(2, self.num_level)
        self.start_pos = [0]
        mask = np.zeros((self.k * self.num_space, input_matrix.shape[1]))

        # a_tilde = []
        g_matrix = [] #G
        for i in range(self.num_space):
            mask[i * self.k:(i + 1) * self.k, self.start_pos[i]:self.start_pos[i] + len(self.j_indices[i])] = 1.
            self.start_pos.append(self.start_pos[i] + len(self.j_indices[i]))
            self.trees.append(HashingTree(input_matrix[:, self.j_indices[i]], self.num_level))
        #     a = []
        #     g = []
        #     buckets = self.trees[-1].get_buckets()[pow(2, self.num_level) - 1:]
        #     for index, bucket in enumerate(buckets):
        #         a += bucket
        #         g += [index] * len(bucket)
        #     a_tilde.append(np.array(a))
        #     g_matrix.append(np.eye(self.k)[g])
        # a_tilde = np.concatenate(a_tilde, axis=-1)
        # g_matrix = np.concatenate(g_matrix, axis=-1)
            index, _ = self.trees[-1].calc_prot_and_index(input_matrix[:, self.j_indices[i]])
            g_matrix.append(np.eye(self.k)[index])
        a_tilde = input_matrix[:, [n for m in self.j_indices for n in m]]
        g_matrix = np.concatenate(g_matrix, axis=-1)
        self.prototypes = np.dot(np.dot(np.linalg.inv(np.dot(g_matrix.T, g_matrix) + np.eye(g_matrix.shape[1])), g_matrix.T), a_tilde) * mask
        self.table = np.dot(self.prototypes, self.new_weight).reshape((self.num_space, self.k, -1))
        if self.bias is not None:
            self.table[0] += self.bias

    def calc(self, inputs):
        result = []
        if len(inputs.shape) == 1:
            for i in range(self.num_space):
                index, _ = self.trees[i].calc_prot_and_index(inputs[self.j_indices[i]])
                result.append(self.table[i, index])
            return np.array(result).sum(axis=0)
        else:
            for i in range(self.num_space):
                index, _ = self.trees[i].calc_prot_and_index(inputs[:, self.j_indices[i]])
                result.append(self.table[i, index])
            return np.stack(result, axis=0).sum(axis=0)

    def save(self):
        dic = {}
        for i in range(self.num_space):
            dic[i] = (self.trees[i].get_thr(), self.trees[i].get_indices(), self.j_indices[i])
        np.save('hashing_model_s.npy', dic)
        np.save('lookup_table_s.npy', self.table)


def mse_loss(approx, truth):
    x = np.linalg.norm(truth - approx, ord=2, axis=1) / approx.shape[-1]
    return np.mean(x)


if __name__ == '__main__':
    print('{:-^50s}'.format(' Loading Data '))
    train_A = np.load('train_matrices.npy')
    test_A = np.load('test_matrices.npy')
    weight_B = np.load('weight.npy')
    bias_B = np.load('bias.npy')

    ori_list = list(range(train_A.shape[-1]))
    random.shuffle(ori_list)
    j_indices = [ori_list[i * 32: i * 32 + 32] for i in range(16)]
    # j_indices = [list(range(i * 32, i * 32 + 32)) for i in range(16)]

    print('{:-^50s}'.format(' Training '))
    start_time = time.time()
    forest = Forest(train_A, weight_B, j_indices, 16, bias_B)
    end_time = time.time()
    print('training cost time: %.3f' % (end_time - start_time))

    print('{:-^50s}'.format(' Approximating '))
    start_time = time.time()
    approx_test_A = forest.calc(test_A)
    end_time = time.time()
    print('approximate matrix shape: {}'.format(approx_test_A.shape))
    print('approximating cost time: %.3f' % (end_time - start_time))

    print('{:-^50s}'.format(' Saving '))
    np.save('approx_test_matrices.npy', approx_test_A)
    forest.save()

    # A = np.random.randint(0, 10, (128, 12))
    # B = np.random.randint(0, 10, (12, 5))
    #
    # train_truth = A[:96] @ B
    # test_truth = A[96:] @ B
    #
    # j_indices = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    # forest = Forest(A[:96], B, j_indices, 3)
    # approx = forest.calc(A[96:])
    # print(approx)
    # print(test_truth)
    # print(mse_loss(approx, test_truth))

    # # print(train_truth)
    # # print(test_truth)
    #
    # tree = HashingTree(A[:96], 4, True)
    # # print(A[:96])
    # print(tree.get_thr())
    # print(tree.get_indices())
    # print(tree.get_buckets())
    # print(tree.get_prototypes())
    # print(tree.get_prototypes_())
    # print(tree.create_table(B, True))
    # approx = tree.approximate_calc(A[96:], True)
    # print(approx)
    # print(test_truth)
    # print(mse_loss(approx, test_truth))

    # 3889572
