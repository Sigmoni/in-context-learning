from samplers import IntSampler
from tasks import Lis

import numpy as np
import matplotlib.pyplot as plt

def lis(arr):
    res = [1]
    ans = 0
    for i in range(1, len(arr)):
        tmp = 0
        for j in range(i):
            if arr[i] >= arr[j]:
                tmp = max(tmp, res[j])
        res.append(tmp + 1)
        ans = max(ans, tmp + 1)

    #print(arr, res)

    return ans

cnt = np.zeros(21)

# sampler = IntSampler(n_dims=20, low=0, high=40, resort=True)
# x = sampler.sample_xs(n_points=1, b_size=10000)
# for batch in x:
#     for point in batch:
#         cnt[lis(point)] += 1

# plt.bar(range(21), cnt)
# plt.savefig("distribution.png")

# x = sampler.sample_xs(n_points=1, b_size=1)
# for batch in x:
#     for point in batch:
#         plt.figure()
#         plt.scatter(range(20), point)
#         plt.savefig("input.png")

sampler = IntSampler(n_dims=1, low=0, high=40, resort=False)
x = sampler.sample_xs(n_points=20, b_size=1)
task = Lis(n_dims=1, batch_size=1)
y = task.evaluate(x)

print(x)
print(y)
