import numpy as np

def mul(inputs, weights):
    m = inputs.shape[0]
    n = inputs.shape[1]
    t = weights.shape[1]
    result = np.empty(shape=(m, t))
    for i in range(m):
        for k in range(t):
            temp = 0
            for j in range(n):
                temp += inputs[i][j] * weights[j][k]
            result[i][k] = temp
    return result

inputs = np.array([[1, 2]])
weights = np.array([[3, 4], [5, 6]])

print(mul(inputs, weights))