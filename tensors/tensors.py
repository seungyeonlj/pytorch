# LEARN THE BASICS
# https://pytorch.org/tutorials/beginner/basics/intro.html
# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from data: \n {x_data}\n")
print(f"Tensor from np array: \n {x_np}\n")

# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float) #float32
print(f"Ones Tensor: \n {x_ones} \n")
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
shape_ = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
ones_tensor_ = torch.ones(shape_)
zeros_tensor = torch.zeros(shape)

print(ones_tensor == ones_tensor_)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Random Tensor: \n {ones_tensor_} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


A = torch.rand(3, 4)
print(f"Shape of tensor: {A.shape}")
print(f"Datatype of tensor: {A.dtype}")
print(f"Device tensor is stored on: {A.device}")

if torch.cuda.is_available():
    A = A.to("cuda")

shape = (3,3,3,)
def get_tensor(shape):
    tensor = torch.ones(shape)
    count = 0

    if len(shape) > 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    tensor[i][j][k] = count
                    count += 1
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                    tensor[i][j] = count
                    count += 1

    return tensor

A = get_tensor(shape)
print(f"tensor: {A}")
print(f"First row: {A[0]}")
print(f"First column: {A[:, 0]}")
print(f"Last column: {A[..., -1]}")
A[:, 1, :] = 0
print(A)
A = torch.ones(3, 3, 3)
A[..., 1] = 0
print(A)


A = torch.ones(2, 3, 1, 5)
print(A)

# concatenate: shape gets longer at given dimension
t1 = torch.cat([A, A, A], dim=1)
print(t1)

A = get_tensor((2, 2, 3))
print(A)
t1 = torch.cat([A, A], dim=0)
t2 = torch.cat([A, A], dim=1)
t3 = torch.cat([A, A], dim=2)
t_default = torch.cat([A, A])
print(t1, t1.shape)
print(t2, t2.shape)
print(t3, t3.shape)
print(t_default, t_default.shape)  # default: dim=0

# cat vs stack
# -- cat: Concatenates the given sequence of seq tensors in the given dimension.
# -- stack: Concatenates sequence of tensors along a new dimension.
A = get_tensor((2, 2, 3))
t1 = torch.stack([A, A])
t2 = torch.cat([A, A])
t3 = torch.stack([A, A], dim=1)
t4 = torch.stack([A, A], dim=2)
t5 = torch.stack([A, A], dim=3)
print(t1, t1.shape)  # (<2>,2,2,3)  # 2 = numer of stackes tensors
print(t2, t2.shape)  # (4,2,3)
print(t3, t3.shape)  # (2,<2>,2,3)
print(t4, t4.shape)  # (2,2,<2>, 3)
print(t5, t5.shape)  # (2,2,3,<2>)


# arithmetic operations
# matrix multiplication
# transpose is needed when matmul
A = get_tensor((2, 2))
print(A)
y1 = A @ A  # AB^T
y2 = A @ A.T  # AB
print(y1)
print(y2)
y3 = torch.rand_like(A)
torch.matmul(A, A.T, out=y3)
print(y3)
y4 = A.matmul(A.T)
print(y4)

# element-wise
z1 = A * A
z2 = A.mul(A)
z3 = torch.rand_like(A)
torch.mul(A, A, out=z3)

# single-element tensors -- convert it to Python numerical value using item()
print(A)
agg = A.sum()
agg_item = agg.item()
print(agg, type(agg))
print(agg_item, type(agg_item))
# print(agg[0], type(agg[0])) # IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
print(agg.shape) #torch.Size([])
agg_proper_tensor = torch.tensor([agg])
print(agg_proper_tensor, type(agg_proper_tensor), agg_proper_tensor.shape)
print(A.shape)


# in-place operations
# Operations that store the result into the operand are called in-place.
# They are denoted by a _ suffix.
# For example: x.copy_(y), x.t_(), will change x
tensor = get_tensor((2,3))
print(f"{tensor}\n")
tensor.add_(5)
print(f"{tensor}\n")


# numpy
# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
print(t.dtype)
n = np.ones(5)
print(f"n: {n}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
print(f"t: {t}")
print(f"n: {n}")

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
