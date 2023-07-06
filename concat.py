import torch

a1 = torch.load("train_matrices_signal_1.npy")
a2 = torch.load("train_matrices_signal_2.npy")
a3 = torch.load("train_matrices_signal_3.npy")
a4 = torch.load("train_matrices_signal_4.npy")

a = torch.concat([a1, a2, a3, a4], axis=0)
print(a.shape, a[0, 0, 0, 0], a[7200, 0, 0, 0], a[14400, 0, 0, 0], a[21600, 0, 0, 0])

torch.save(a, "train_matrices_signal.npy")