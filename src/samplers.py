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
    def __init__(self, n_dims=1, low=0, high=10, resort=False):
        super().__init__(n_dims)
        self.low = low
        self.high = high
        self.resort = resort

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        def random_select_and_sort(arr, n):
            indices = torch.randperm(len(arr))[:n]
            indices, _ = torch.sort(indices)
            samples = arr[indices]
            samples, _ = torch.sort(samples)
            arr[indices] = samples

        def random_numbers(n, generator=None, resort=False):
            arr = None
            if generator is None:
                arr = torch.randint(low=self.low, high=self.high, size=(n,))
            else:
                arr = torch.randint(low=self.low, high=self.high, size=(n,), generator=generator)
            
            if resort:
                n = torch.randint(low=1, high=n, size=()).item()
                arr, _ = torch.sort(arr, descending=True)
                random_select_and_sort(arr, n)

            return arr

        xs_b = torch.zeros(b_size, n_points, self.n_dims)
        if seeds is None:
            for i in range(b_size):
                xs_b[i] = random_numbers(n_points * self.n_dims, resort=self.resort).reshape(n_points, self.n_dims)
        else:
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = random_numbers(n_points * self.n_dims, generator=generator, resort=self.resort).reshape(n_points, self.n_dims)

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b