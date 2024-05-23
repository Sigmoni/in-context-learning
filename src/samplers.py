import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "int": IntSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class IntSampler(DataSampler):
    def __init__(self, n_dims, low=0, high=10):
        super().__init__(n_dims)
        self.low = low
        self.high = high

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        def random_select_and_sort(arr, n):
            indices = torch.randint(low=0, high=self.n_dims, size=(n, ))
            indices, _ = torch.sort(indices)
            samples = arr[indices]
            samples, _ = torch.sort(samples)
            arr[indices] = samples

        if seeds is None:
            xs_b = torch.randint(low=self.low, high=self.high, size=(b_size, n_points, self.n_dims))
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randint(low=self.low, high=self.high, size=(n_points, self.n_dims), generator=generator)
        
        for i in range(b_size):
            for j in range(n_points):
                n = torch.randint(low=1, high=self.n_dims, size=()).item()
                xs_b[i][j], _ = torch.sort(xs_b[i][j])
                random_select_and_sort(xs_b[i][j], n)

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b