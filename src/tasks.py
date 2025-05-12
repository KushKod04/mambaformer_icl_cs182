import math
import torch


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()
ce_loss = torch.nn.CrossEntropyLoss()

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def absolute_error(ys_pred, ys):
    return (ys - ys_pred).abs()

def mean_absolute_error(ys_pred, ys):
    return (ys - ys_pred).abs().mean()

def squared_norm_error(ys_pred, ys, dim=-1):
    return (ys - ys_pred).square().sum(dim=dim).unsqueeze(dim=dim)

def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()

def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "retrieval": Retrieval,
        "sparse_parity": SparseParity,
        "token_induction_head": InductionHead,
        "filter_linear_regression": FilterLinearRegression,
        "filter_relu_2nn_regression": FilterRelu2nnRegression,
        "filter_scale_linear_regression": FilterScaleLinearRegression,
        "filter_ortho_linear_regression": FilterOrthoLinearRegression,
        "sinusoidal_regression": SinusoidalRegression,
        "long_term_dependency": LongTermDependency,
        "modulo_classification": ModuloClassification,
        "euclidean_distance": EuclideanDistance,
        "l1_distance": L1Distance,
        "vector_manipulation": VectorManipulation,
        "high_frequency": HighFrequency,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        if "token_induction_head" in task_name:
            return lambda **args: task_cls(
                batch_size,
                pool_dict,
                **args,
                **kwargs
            )
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class Retrieval(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        super(Retrieval, self).__init__(n_dims, batch_size, pool_dict, seeds)

    def evaluate(self, xs_b):
        # xs_b: (self.b_size, n_points, self.n_dims)
        assert xs_b.shape[0] == self.b_size and xs_b.shape[2] == self.n_dims
        assert xs_b.shape[1] % 2 == 1

        keys_b, values_b = xs_b[:, :-1:2, :], xs_b[:, 1::2, :]
        query_b = xs_b[:, -1, :].unsqueeze(dim=1)
        inner_products = torch.bmm(query_b, keys_b.transpose(1, 2)).squeeze(1)
        _, retrieval_inds = torch.max(inner_products, dim=1)
        retrieval_inds = retrieval_inds.view(-1, 1, 1).expand(-1, -1, values_b.size(-1))
        ys_b = torch.gather(values_b, 1, retrieval_inds).squeeze(1)

        # ys_b: (b_size, n_dims)
        return ys_b
    
    def classify(self, xs_b, ys_b, pred):
        keys_b, values_b = xs_b[:, :-1:2, :], xs_b[:, 1::2, :]
        query_b, pred_b = ys_b.unsqueeze(dim=1), pred.unsqueeze(dim=1)

        inner_products = torch.bmm(query_b, keys_b.transpose(1, 2)).squeeze(1)
        _, gold_inds = torch.max(inner_products, dim=1)
        inner_products = torch.bmm(pred_b, keys_b.transpose(1, 2)).squeeze(1)
        _, pred_inds = torch.max(inner_products, dim=1)
        match_error = (1-(gold_inds == pred_inds).float()).unsqueeze(1)
        return match_error

    @staticmethod
    def get_metric():
        return squared_norm_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class InductionHead(Task):
    def __init__(self, batch_size, vocab_size=128, chain_depth=1, pool_dict=None, seeds=None):
        super(InductionHead, self).__init__(1, batch_size, pool_dict, seeds)
        self.vocab_size = vocab_size
        self.chain_depth = chain_depth

    def evaluate(self, xs_b):
        # xs_b: (self.b_size, n_points). The last point should always be 0 (retrieval token)
        assert xs_b.shape[0] == self.b_size and len(xs_b.shape) == 2
        assert xs_b.shape[1] % 2 == 1

        ys_b = xs_b[:, -1]
        for i in range(self.chain_depth):
            # only look at keys (not values)
            mask = xs_b[:, ::2] == ys_b[:, None]
            # Finding the first index of zero in each 1D tensor
            retrieved_inds = torch.argmax(mask.int(), dim=1) * 2 + 1
            ys_b = xs_b[torch.arange(self.b_size), retrieved_inds]

        return ys_b
    
    def classify(self, xs_b, ys_b, pred):
        match_error = (1-(ys_b == pred).float()).unsqueeze(1)
        return match_error

    @staticmethod
    def get_metric():
        return ce_loss

    @staticmethod
    def get_training_metric():
        return ce_loss
    

class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class SinusoidalRegression(Task):
    def __init__(self, n_dims=1, batch_size=32, pool_dict=None, seeds=None, ampl_range=(-2, 2), freq_range=(0.5, 1.4), phase_range=(0, 2 * math.pi)):
        super(SinusoidalRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.ampl_range = ampl_range
        self.freq_range = freq_range
        self.phase_range = phase_range

        if pool_dict is None and seeds is None:
            self.ampls = torch.empty(batch_size).uniform_(*ampl_range)
            self.freqs = torch.empty(batch_size).uniform_(*freq_range)
            self.phases = torch.empty(batch_size).uniform_(*phase_range)
        elif seeds is not None:
            self.ampls = torch.zeros(batch_size)
            self.freqs = torch.zeros(batch_size)
            self.phases = torch.zeros(batch_size)
            for i, seed in enumerate(seeds):
                g = torch.Generator().manual_seed(seed)
                self.ampls[i] = torch.empty(1, generator=g).uniform_(*ampl_range)
                self.freqs[i] = torch.empty(1, generator=g).uniform_(*freq_range)
                self.phases[i] = torch.empty(1, generator=g).uniform_(*phase_range)
        else:
            self.ampls = pool_dict["ampls"]
            self.freqs = pool_dict["freqs"]
            self.phases = pool_dict["phases"]

    def evaluate(self, xs_b):
        # xs_b: [B, S, D]
        ampls = self.ampls[:, None, None].to(xs_b.device)   # [B, 1, 1]
        freqs = self.freqs[:, None, None].to(xs_b.device)   # [B, 1, 1]
        phases = self.phases[:, None, None].to(xs_b.device) # [B, 1, 1]
        return ampls * torch.sin(freqs * xs_b + phases)  

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, ampl_range=(-2, 2), freq_range=(0.8, 1.2), phase_range=(0, 2 * math.pi), **kwargs):
        ampls = torch.empty(num_tasks).uniform_(*ampl_range)
        freqs = torch.empty(num_tasks).uniform_(*freq_range)
        phases = torch.empty(num_tasks).uniform_(*phase_range)
        return {"ampls": ampls, "freqs": freqs, "phases": phases}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class L1Distance(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the distances."""
        super(L1Distance, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            # sample random w vectors
            self.w = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            # draw from existing pool
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w = pool_dict["w"][indices]

    def evaluate(self, xs):
        """
        xs: Tensor of shape [batch_size, num_points, n_dims]
        returns: Tensor of shape [batch_size, num_points] where
                 output[b, i] = scale * sum_j (w[b,j] - xs[b,i,j])**2
        """
        # bring w to same device and shape
        w = self.w.to(xs.device).squeeze(-1)           # [B, n_dims]
        w = w.unsqueeze(1)                              # [B, 1, n_dims]
        diff = xs - w                                   # [B, P, n_dims]
        l1_dist = diff.abs().sum(dim=2)  # [B, P]
        return self.scale * l1_dist

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    

class EuclideanDistance(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the distances."""
        super(EuclideanDistance, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            # sample random w vectors
            self.w = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            # draw from existing pool
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w = pool_dict["w"][indices]
    def evaluate(self, xs_b):
        w_b = self.w.to(xs_b.device)
        w_reshaped = w_b.squeeze(-1).unsqueeze(1)  # [B, 1, n_dims]
        ys_b = self.scale * torch.norm(xs_b - w_reshaped, p=2, dim=2)  # [B, P]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class HighFrequency(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """
        A task that projects inputs onto high-frequency alternating weights (+1/-1 pattern).
        
        Args:
            n_dims: Dimensionality of the input.
            batch_size: Number of weight vectors to sample.
            pool_dict: Dictionary of pre-generated weights.
            seeds: Random seeds for reproducibility.
            scale: A constant by which to scale the weights.
        """
        super(HighFrequency, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = 10

        if pool_dict is None and seeds is None:
            self.w_b = self._generate_weights(self.b_size, self.n_dims)
        elif seeds is not None:
            self.w_b = self._generate_weights(self.b_size, self.n_dims)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
    
    def _generate_weights(self, batch_size, n_dims):
        """Generate alternating +1/-1 weights."""
        # Create batch_size copies of the alternating pattern
        weights = torch.ones(batch_size, n_dims, 1)
        weights[:, 1::2, :] = -1  # Set odd indices to -1
        return weights * self.scale
        
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        weights = torch.ones(num_tasks, n_dims, 1)
        weights[:, 1::2, :] = -1  # Set odd indices to -1
        return {"w": weights}
        
    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class FilterLinearRegression(LinearRegression):
    pass

class FilterScaleLinearRegression(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, dummy_scale=0.01):
        super(FilterScaleLinearRegression, self).__init__(n_dims, batch_size, pool_dict=None, seeds=None, scale=1)
        self.dummy_scale = dummy_scale

class FilterOrthoLinearRegression(LinearRegression):
    pass

class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class FilterRelu2nnRegression(Relu2nnRegression):
    pass


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class SparseParity(Task):
    # k = num_indices
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, k=None):
        super(SparseParity, self).__init__(n_dims, batch_size, pool_dict, seeds)
        assert k is not None
        self.k = k
        self.indices = torch.randperm(n_dims)[:self.k]

    # xs_b: (self.b_size, n_points, self.n_dims)
    def evaluate(self, xs_b):
        assert xs_b.shape[2] >= self.k

        ys_b = torch.ones(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for idx in self.indices:
            ys_b *= xs_b[:, :, idx]
        return ys_b
        
    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class LongTermDependency(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        """
        A task where the output depends on a specific index of the input
        that is randomly selected during initialization.
        """
        super(LongTermDependency, self).__init__(n_dims, batch_size, pool_dict, seeds)
        
        if pool_dict is None and seeds is None:
            # Randomly select an index for each batch item
            self.selected_indices = torch.randint(0, self.n_dims, (self.b_size,))
        elif seeds is not None:
            # Use seeds to deterministically select indices
            self.selected_indices = torch.zeros(self.b_size, dtype=torch.long)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.selected_indices[i] = torch.randint(0, self.n_dims, (1,), generator=generator)
        else:
            # Select indices from the provided pool
            assert "indices" in pool_dict
            indices = torch.randperm(len(pool_dict["indices"]))[:batch_size]
            self.selected_indices = pool_dict["indices"][indices]
            
    def evaluate(self, xs_b):
        # Extract the values at the selected indices for each batch item
        # xs_b shape: [batch_size, sequence_length, n_dims]
        batch_indices = torch.arange(xs_b.shape[0])
        # Extract the value at the selected index for each sample in the batch
        selected_values = xs_b[batch_indices, :, self.selected_indices]
        
        # Return the selected values
        # This creates a dependency where the output depends solely on the selected index
        return selected_values

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {"indices": torch.randint(0, n_dims, (num_tasks,))}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class ModuloClassification(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, modulo=3, modulo_choices=[3, 5, 7]):
        """
        Classify the sum of vector elements modulo a small integer (e.g. 3).
        Output is a class label in {0, ..., modulo-1}.
        """
        super(ModuloClassification, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.modulo = modulo
        self.modulo_choices = modulo_choices

        if pool_dict is None and seeds is None:
            # Randomly assign a modulo per n_point from the set
            self.modulos = torch.tensor([
                modulo_choices[torch.randint(0, len(modulo_choices), (1,)).item()]
                for _ in range(self.n_dims)
            ], device='cuda')
        elif seeds is not None:
            # Deterministically assign modulos using seeds
            self.modulos = torch.zeros(self.n_dims, dtype=torch.long, device='cuda')
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.modulos[i] = self.modulo_choices[torch.randint(0, len(self.modulo_choices), (1,), generator=generator).item()]
        else:
            # Use pool_dict
            assert "modulos" in pool_dict
            self.modulos = pool_dict["modulos"][:self.n_dims]

    def evaluate(self, xs_b):
        """
        Compute the label as sum(xs) % modulo.
        Inputs:
            xs_b: Tensor of shape (batch_size, n_points, n_dims)
        Outputs:
            ys_b: Tensor of shape (batch_size, n_points) with class labels [0, modulo-1]
        """
        _, n_points, n_dims = xs_b.shape
        # Sum over last dimension (dim=-1) -> length of vector for each sample point
        summed = xs_b.sum(dim=-1)  # shape: (batch_size, n_points)
        if n_dims < n_points:
            modulos = self.modulos.repeat(n_points // n_dims + 1)[:n_points]  # (n_points,)
        else:
            modulos = self.modulos[:n_points]  # (n_points,)
        # labels = (summed.long() % self.modulo)#.view(-1, 1)  # final output: (batch_size, n_points)
        labels = (summed.long() % modulos)  # final output: (batch_size, n_points)
        return labels

    def classify(self, xs_b, ys_b, pred):
        """
        Classification accuracy metric.
            pred: logits of shape [batch_size, n_points, modulo]
            ys_b: true labels of shape [batch_size, n_points]
        Returns: 0-1 loss tensor of shape [batch_size, n_points]
        """
        pred_class = pred.argmax(dim=-1)  # [batch_size, n_points]
        incorrect = (pred_class != ys_b).float()
        return incorrect  # 1 where incorrect, 0 where correct

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, modulo_choices=[3,5,7], **kwargs):
        modulos = torch.tensor([
            modulo_choices[torch.randint(0, len(modulo_choices), (1,)).item()]
            for _ in range(num_tasks)
        ], device='cuda')

        return {"modulos": modulos}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class VectorManipulation(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        """
        A task where the output depends on a specific index of the input
        that is randomly selected during initialization.
        """
        super(VectorManipulation, self).__init__(n_dims, batch_size, pool_dict, seeds)

    
    def evaluate(self, xs_b):
        #Divide xs_b into two equal size subvectors and get the intermediate vector sum.
        #Then, obtain the product of all elements in the intermediate vector sum.
        vec = xs_b[:,:,:xs_b.nonzero(as_tuple=True)[2][-1] + 1] #gets the actual input vector (ignoring zero padding for sampling)
        left, right = vec.split(vec.shape[2] // 2, dim=2)
        sum = left + right
        out = torch.prod(sum, dim=2, keepdim=True)
        out_padded = torch.nn.functional.pad(out, (0, xs_b.shape[2] - 1))
        return out_padded

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {"indices": torch.randint(0, n_dims, (num_tasks,))}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

