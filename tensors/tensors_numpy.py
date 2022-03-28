# Bridge with NumPy
import torch
import numpy as np

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)  # default: float32
print(f"t: {t}")
print(t[0].dtype)
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


n = np.ones(5)
t = torch.from_numpy(n)
print(f"t: {t}")
print(f"n: {n}")

np.add(1, n, out=n)
print(f"t: {t}")
print(f"n: {n}")
