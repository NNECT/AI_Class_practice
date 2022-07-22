import numpy as np


print(np.zeros(3))
print(np.zeros(3).dtype)
print(np.zeros([3, 5]))
print(np.zeros([3, 5]).dtype)

print(np.full([2, 3], -1))

print(np.random.rand(3))
print(np.random.rand(2, 3))     # 균등분포(uniform)
print(np.random.randn(2, 3))    # n: normal distribution

a = np.arange(12).reshape(3, 4)
print(a)

print(a[0][0], a[-1][-1])
print(a[0, 0], a[-1, -1])       # fancy indexing

print(a[-1])
print(a[-1, :])
print(a[:, -1])

print(a[::-1, ::-1])

# 테두리 채우기
b = np.full([5, 5], fill_value=1)
# b[0, :] = 0
# b[-1, :] = 0
# b[:, 0] = 0
# b[:, -1] = 0
b[(0, -1), :] = 0
b[:, (0, -1)] = 0
print(b)

# 테두리를 뺀 안쪽 채우기
b[1:-1, 1:-1] = 2
print(b)

# 대각선 채우기
for i, row in enumerate(b):
    row[[i, -1 - i]] = 3

for i in range(len(b)):
    b[i, [i, -1 - i]] = 3

b[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]] = 5
b[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]] = 5

b[range(5), range(5)] = 6
b[range(5), tuple(reversed(range(5)))] = 6

print(b)