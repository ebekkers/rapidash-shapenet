import torch
import torch.nn as nn

from torch import Tensor
from torch.optim import SGD

from typing import Optional

import math

try:
    from torch_cluster import knn_graph, knn, fps

    def fps_edge_index(pos, batch, ratio):
        fps_index = fps(pos, batch, ratio)

        pos_fps = pos[fps_index]
        batch_fps = batch[fps_index]

        edge_index = knn(pos_fps, pos, 1, batch_fps, batch)

        return edge_index, pos_fps, batch_fps

except ModuleNotFoundError:
    import warnings

    warnings.warn(
        "Module `torch_cluster` not found. Defaulting to local clustering algorithms which may increase computation time."
    )

    def fps_edge_index(pos, batch, ratio):
        batch_size = batch.max().item() + 1
        all_target_nodes = []
        all_fps_indices = []

        num_samples_running = 0
        for b in range(batch_size):
            pos_b = pos[batch == b]
            num_nodes = pos_b.size(0)
            num_samples = max(int(ratio * num_nodes), 1)

            fps_indices = torch.zeros(num_samples, dtype=torch.long, device=pos.device)
            distances = torch.full((num_nodes,), float("inf"), device=pos.device)

            initial_index = torch.randint(0, num_nodes, (1,))
            fps_indices[0] = initial_index

            for i in range(1, num_samples):
                new_point = pos_b[fps_indices[i - 1]]
                current_distances = torch.norm(pos_b - new_point.unsqueeze(0), dim=1)
                distances = torch.min(distances, current_distances)
                fps_indices[i] = torch.argmax(distances)
            all_fps_indices.append(
                fps_indices + (batch == b).nonzero(as_tuple=True)[0].min()
            )

            # Compute edge_index: each source connected to nearest fps point
            dist_matrix = torch.cdist(pos_b, pos_b[fps_indices])
            nearest_indices = torch.argmin(dist_matrix, dim=1)
            target_nodes = nearest_indices + num_samples_running

            all_target_nodes.append(target_nodes)
            num_samples_running += num_samples

        source_nodes = torch.arange(pos.size(0), device=pos.device, dtype=torch.long)
        target_nodes = torch.cat(all_target_nodes)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        fps_indices = torch.cat(all_fps_indices)
        fps_pos = pos[fps_indices]
        fps_batch = batch[fps_indices]

        return edge_index, fps_pos, fps_batch

    def knn(x, y, k, batch_x=None, batch_y=None):
        """For every point in `y`, returns `k` nearest neighbors in `x`."""

        edge_index = y.new_empty(2, k * y.shape[0], dtype=torch.long)

        if batch_x is None:
            batch_x = x.new_zeros(x.shape[0], dtype=torch.long)

        if batch_y is None:
            batch_y = y.new_zeros(y.shape[0], dtype=torch.long)

        num_seen = 0

        for i, (b, b_size) in enumerate(
            zip(*torch.unique(batch_y, return_counts=True))
        ):
            x_b, y_b = x[batch_x == b], y[batch_y == b]

            batch_offset = i * b_size
            num_per_batch = k * b_size

            source = (
                torch.arange(b_size, device=b_size.device, dtype=torch.long)
            ).repeat_interleave(k) + batch_offset

            target = (
                torch.topk(torch.cdist(y_b, x_b), k, largest=False)[1].flatten()
                + batch_offset
            )

            edge_index[0, num_seen : num_seen + num_per_batch] = target
            edge_index[1, num_seen : num_seen + num_per_batch] = source

            num_seen += num_per_batch

        return edge_index

    def knn_graph(x, k, batch=None, loop=False, flow="source_to_target"):
        """
        For each point in `x`, calculates its `k` nearest neighbors.
        If `loop` is `True`, neighbors include self-connections.
        """
        assert flow in ["source_to_target", "target_to_source"]

        k += not loop

        edge_index = knn(x, x, k, batch, batch)

        if not loop:
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        if flow == "target_to_source":
            edge_index = edge_index.flip(0)

        return edge_index


def fully_connected_edge_index(batch_idx):
    edge_indices = []

    for batch_num in torch.unique(batch_idx):
        # Find indices of nodes in the current batch
        node_indices = torch.where(batch_idx == batch_num)[0]
        grid = torch.meshgrid(node_indices, node_indices, indexing="ij")
        edge_indices.append(
            torch.stack([grid[0].reshape(-1), grid[1].reshape(-1)], dim=0)
        )

    edge_index = torch.cat(edge_indices, dim=1)

    return edge_index


def scatter_add(src, index, dim_size):
    out_shape = [dim_size] + list(src.shape[1:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    dims_to_add = src.dim() - index.dim()
    for _ in range(dims_to_add):
        index = index.unsqueeze(-1)
    index_expanded = index.expand_as(src)
    return out.scatter_add_(0, index_expanded, src)


def scatter_softmax(src, index, dim_size):
    src_exp = torch.exp(src - src.max())
    sum_exp = scatter_add(src_exp, index, dim_size) + 1e-6
    return src_exp / sum_exp[index]


class GridGenerator(nn.Module):
    def __init__(
        self,
        dim: int,
        n: int,
        steps: int = 200,
        step_size: float = 0.01,
        device: torch.device = None,
    ):
        super(GridGenerator, self).__init__()
        self.dim = dim
        self.n = n
        self.steps = steps
        self.step_size = step_size
        self.device = device if device else torch.device("cpu")

    def forward(self) -> torch.Tensor:
        if self.dim == 2:
            return self.generate_s1()
        elif self.dim == 3:
            return self.generate_s2()
        else:
            raise ValueError("Only S1 and S2 are supported.")

    def generate_s1(self) -> torch.Tensor:
        angles = torch.linspace(
            start=0, end=2 * torch.pi - (2 * torch.pi / self.n), steps=self.n
        )
        x = torch.cos(angles)
        y = torch.sin(angles)
        return torch.stack((x, y), dim=1)

    def generate_s2(self) -> torch.Tensor:
        grid = self.random_s2((self.n,), device=self.device)
        return self.repulse(grid)

    def random_s2(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        x = torch.randn((*shape, 3), device=device)
        return x / torch.linalg.norm(x, dim=-1, keepdim=True)

    def repulse(self, grid: torch.Tensor) -> torch.Tensor:
        grid = grid.clone().detach().requires_grad_(True)
        optimizer = SGD([grid], lr=self.step_size)

        for _ in range(self.steps):
            optimizer.zero_grad()
            dists = torch.cdist(grid, grid, p=2)
            dists = torch.clamp(dists, min=1e-6)  # Avoid division by zero
            energy = dists.pow(-2).sum()  # Simplified Coulomb energy calculation
            energy.backward()
            optimizer.step()

            with torch.no_grad():
                # Renormalize points back to the sphere after update
                grid /= grid.norm(dim=-1, keepdim=True)

        return grid.detach()

    def fibonacci_lattice(
        n: int, offset: float = 0.5, device: Optional[str] = None
    ) -> Tensor:
        """
        Creating ~uniform grid of points on S2 using the fibonacci spiral algorithm.

        Arguments:
            - n: Number of points.
            - offset: Strength for how much points are pushed away from the poles.
                    Default of 0.5 works well for uniformity.
        """
        if n < 1:
            raise ValueError("n must be greater than 0.")

        i = torch.arange(n, device=device)

        theta = (math.pi * i * (1 + math.sqrt(5))) % (2 * math.pi)
        phi = torch.acos(1 - 2 * (i + offset) / (n - 1 + 2 * offset))

        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)

        return torch.stack((cos_theta * sin_phi, sin_theta * sin_phi, cos_phi), dim=-1)


class SeparableFiberBundleConv(nn.Module):
    """ """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_dim,
        bias=True,
        groups=1,
        attention=False,
    ):
        super().__init__()

        # Check arguments
        if groups == 1:
            self.depthwise = False
        elif groups == in_channels and groups == out_channels:
            self.depthwise = True
            self.in_channels = in_channels
            self.out_channels = out_channels
        else:
            assert ValueError(
                "Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)"
            )

        # Construct kernels
        self.kernel = nn.Linear(kernel_dim, in_channels, bias=False)
        self.fiber_kernel = nn.Linear(
            kernel_dim, int(in_channels * out_channels / groups), bias=False
        )
        self.attention = attention
        if self.attention:
            key_dim = 128
            self.key_transform = nn.Linear(in_channels, key_dim)
            self.query_transform = nn.Linear(in_channels, key_dim)
            nn.init.xavier_uniform_(self.key_transform.weight)
            nn.init.xavier_uniform_(self.query_transform.weight)
            self.key_transform.bias.data.fill_(0)
            self.query_transform.bias.data.fill_(0)

        # Construct bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            self.bias.data.zero_()
        else:
            self.register_parameter("bias", None)

        # Automatic re-initialization
        self.register_buffer("callibrated", torch.tensor(False))

    def forward(self, x, kernel_basis, fiber_kernel_basis, edge_index):
        """ """

        # 1. Do the spatial convolution
        message = x[edge_index[0]] * self.kernel(
            kernel_basis
        )  # [num_edges, num_ori, in_channels]
        if self.attention:
            keys = self.key_transform(x)
            queries = self.query_transform(x)
            d_k = keys.size(-1)
            att_logits = (keys[edge_index[0]] * queries[edge_index[1]]).sum(
                dim=-1, keepdim=True
            ) / math.sqrt(d_k)
            att_weights = scatter_softmax(att_logits, edge_index[1], x.size(0))
            message = message * att_weights
        x_1 = scatter_add(
            src=message, index=edge_index[1], dim_size=edge_index[1].max().item() + 1
        )

        # 2. Fiber (spherical) convolution
        fiber_kernel = self.fiber_kernel(fiber_kernel_basis)
        if self.depthwise:
            x_2 = (
                torch.einsum("boc,poc->bpc", x_1, fiber_kernel) / fiber_kernel.shape[-2]
            )
        else:
            x_2 = (
                torch.einsum(
                    "boc,podc->bpd",
                    x_1,
                    fiber_kernel.unflatten(-1, (self.out_channels, self.in_channels)),
                )
                / fiber_kernel.shape[-2]
            )

        # Re-callibrate the initializaiton
        if self.training and not (self.callibrated):
            self.callibrate(x.std(), x_1.std(), x_2.std())

        # Add bias
        if self.bias is not None:
            return x_2 + self.bias
        else:
            return x_2

    def callibrate(self, std_in, std_1, std_2):
        print("Callibrating...")
        with torch.no_grad():
            self.kernel.weight.data = self.kernel.weight.data * std_in / std_1
            self.fiber_kernel.weight.data = (
                self.fiber_kernel.weight.data * std_1 / std_2
            )
            self.callibrated = ~self.callibrated


class SeparableFiberBundleConvNext(nn.Module):
    """ """

    def __init__(
        self,
        in_channels,
        kernel_dim,
        out_channels=None,
        act=nn.GELU(),
        layer_scale=1e-6,
        widening_factor=4,
        attention=False,
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.conv = SeparableFiberBundleConv(
            in_channels,
            in_channels,
            kernel_dim,
            groups=in_channels,
            attention=attention,
        )

        self.act_fn = act

        self.linear_1 = nn.Linear(in_channels, widening_factor * in_channels)
        self.linear_2 = nn.Linear(widening_factor * in_channels, out_channels)

        if layer_scale is not None:
            self.layer_scale = nn.Parameter(torch.ones(out_channels) * layer_scale)
        else:
            self.register_buffer("layer_scale", None)

        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, kernel_basis, fiber_kernel_basis, edge_index):
        """ """
        input = x
        x = self.conv(x, kernel_basis, fiber_kernel_basis, edge_index)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)

        if self.layer_scale is not None:
            x = self.layer_scale * x

        if x.shape == input.shape:
            x = x + input

        return x


class PolynomialFeatures(nn.Module):
    def __init__(self, degree):
        super(PolynomialFeatures, self).__init__()

        self.degree = degree

    def forward(self, x):

        polynomial_list = [x]
        for it in range(1, self.degree + 1):
            polynomial_list.append(
                torch.einsum("...i,...j->...ij", polynomial_list[-1], x).flatten(-2, -1)
            )
        return torch.cat(polynomial_list, -1)


def invariant_attr_r2s1_fiber_bundle(pos, ori_grid, edge_index, separable=False):
    pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]]  # [num_edges, 3]
    rel_pos = pos_send - pos_receive  # [num_edges, 3]

    # Convenient shape
    rel_pos = rel_pos[:, None, :]  # [num_edges, 1, 3]
    ori_grid_a = ori_grid[None, :, :]  # [1, num_ori, 3]
    ori_grid_b = ori_grid[:, None, :]  # [num_ori, 1, 3]

    # Note ori_grid consists of tuples (ori[0], ori[1]) = (cos t, sin t)
    # A transposed rotation (cos t, sin t \\ - sin t, cos t) is then
    # acchieved as (ori[0], ori[1] \\ -ori[1], ori[0]):
    invariant1 = (
        rel_pos[..., 0] * ori_grid_a[..., 0] + rel_pos[..., 1] * ori_grid_a[..., 1]
    ).unsqueeze(-1)
    invariant2 = (
        -rel_pos[..., 0] * ori_grid_a[..., 1] + rel_pos[..., 1] * ori_grid_a[..., 0]
    ).unsqueeze(-1)
    invariant3 = (ori_grid_a * ori_grid_b).sum(
        dim=-1, keepdim=True
    )  # [num_ori, num_ori, 1]

    # Note: We could apply the acos = pi/2 - asin, which is differentiable at -1 and 1
    # But found that this mapping is unnecessary as it is monotonic and mostly linear
    # anyway, except close to -1 and 1. Not applying the arccos worked just as well.
    # invariant3 = torch.pi / 2 - torch.asin(invariant3.clamp(-1.,1.))

    if separable:
        return (
            torch.cat([invariant1, invariant2], dim=-1),
            invariant3,
        )  # [num_edges, num_ori, 2], [num_ori, num_ori, 1]
    else:
        invariant1 = invariant1[:, :, None, :].expand(
            -1, -1, ori_grid.shape[0], -1
        )  # [num_edges, num_ori, num_ori, 1]
        invariant2 = invariant2[:, :, None, :].expand(
            -1, -1, ori_grid.shape[0], -1
        )  # [num_edges, num_ori, num_ori, 1]
        invariant3 = invariant3[None, :, :, :].expand(
            invariant1.shape[0], -1, -1, -1
        )  # [num_edges, num_ori, num_ori, 1]
        return torch.cat(
            [invariant1, invariant2, invariant3], dim=-1
        )  # [num_edges, num_ori, num_ori, 3]


class Rapidash(nn.Module):
    """Steerable E(3) equivariant (non-linear) convolutional network"""

    _supported_edge_types = ["fc", "knn"]

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        edge_types=["fc"],
        ratios=[],
        output_dim_vec=0,
        dim=3,
        num_ori=12,
        basis_dim=None,
        degree=2,
        widening_factor=4,
        layer_scale=None,
        task_level="node",
        multiple_readouts=False,
        last_feature_conditioning=True,
        attention=False,
        fully_connected=False,
        residual_connections=True,
        global_basis=False,
        **kwargs
    ):
        super().__init__()

        self.fully_connected = fully_connected

        self.layers_per_scale = [num_layers] if type(num_layers) is int else num_layers

        n = len(self.layers_per_scale)

        self.edge_types = n * [edge_types] if type(edge_types) is str else edge_types
        self.ratios = n * [ratios] if type(ratios) is float else ratios + [1.0]
        self.hidden_dims = n * [hidden_dim] if type(hidden_dim) is int else hidden_dim
        self.hidden_dim = self.hidden_dims[-1]

        self._parse_edge_types()

        # if true, applies u-net like residual connections
        self.residual_connections = residual_connections
        # if false, each scale has its own basis
        self.global_basis = global_basis

        assert len(self.layers_per_scale) == len(self.edge_types) == len(self.ratios)

        self.num_scales = len(self.layers_per_scale)

        self.in_out_dims_per_layer = []

        if not (task_level == "graph") and self.num_scales > 1:
            self.up_sample = True
            self.effective_num_layers = (
                sum(self.layers_per_scale) + self.num_scales - 1
            ) * 2
            self.in_out_dims_per_layer += self.in_out_dims_per_layer[::-1]
        else:
            self.up_sample = False
            self.effective_num_layers = sum(self.layers_per_scale) + self.num_scales - 1

        self.dim = dim
        self.last_feature_conditioning = last_feature_conditioning
        self.grid_generator = GridGenerator(dim, num_ori, steps=1000)
        self.ori_grid = self.grid_generator()

        # Input output settings
        self.output_dim, self.output_dim_vec = output_dim, output_dim_vec
        self.global_pooling = task_level == "graph"

        # Activation function to use internally
        act_fn = nn.GELU()

        # Kernel basis functions and spatial window
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim

        self.basis_fn = nn.ModuleList(
            nn.Sequential(
                PolynomialFeatures(degree),
                nn.LazyLinear(self.hidden_dim),
                act_fn,
                nn.Linear(self.hidden_dim, basis_dim),
                act_fn,
            )
            for _ in range(1 if global_basis else len(self.ratios))
        )
        self.fiber_basis_fn = nn.ModuleList(
            nn.Sequential(
                PolynomialFeatures(degree),
                nn.LazyLinear(self.hidden_dim),
                act_fn,
                nn.Linear(self.hidden_dim, basis_dim),
                act_fn,
            )
            for _ in range(1 if global_basis else len(self.ratios))
        )

        # Initial node embedding
        self.x_embedder = nn.Linear(
            input_dim + last_feature_conditioning, self.hidden_dims[0], False
        )

        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        self.read_out_layers = nn.ModuleList()

        hidden_dims = list(
            map(
                int,
                torch.repeat_interleave(
                    torch.Tensor(self.hidden_dims), torch.tensor(self.layers_per_scale).int() + 1
                ).tolist(),
            )
        )

        if self.up_sample:
            hidden_dims = hidden_dims + hidden_dims[:-1][::-1]
            # hidden_dims = hidden_dims + hidden_dims_compress_up

        for i in range(self.effective_num_layers):
            in_channels, out_channels = hidden_dims[i : i + 2]
            print('layer:', i, 'in channels:', in_channels, 'out channels:', out_channels)
            self.interaction_layers.append(
                SeparableFiberBundleConvNext(
                    in_channels,
                    basis_dim,
                    out_channels=out_channels,
                    act=act_fn,
                    layer_scale=layer_scale,
                    widening_factor=widening_factor,
                    attention=attention,
                )
            )
            if multiple_readouts or i == (self.effective_num_layers - 1):
                self.read_out_layers.append(
                    nn.Linear(out_channels, output_dim + output_dim_vec)
                )
            else:
                self.read_out_layers.append(None)

    def _parse_edge_types(self):
        edge_types = []
        edge_types_kwargs = []

        for edge_type in self.edge_types:
            if edge_type.lower() == "fc":
                edge_types.append(edge_type.lower())
                edge_types_kwargs.append({})
                continue

            edge_type, edge_type_kwargs = edge_type.lower().split("-")

            if edge_type == "knn":
                edge_types.append(edge_type)
                edge_types_kwargs.append({"k": int(edge_type_kwargs)})
            else:
                raise ValueError("Given edge type not in:", self._supported_edge_types)

        self.edge_types = edge_types
        self.edge_types_kwargs = edge_types_kwargs

    def compute_invariants(self, ori_grid, pos_send, pos_receive):
        rel_pos = pos_send - pos_receive  # [num_edges, 3]
        rel_pos = rel_pos[:, None, :]  # [num_edges, 1, 3]
        ori_grid_a = ori_grid[None, :, :]  # [1, num_ori, 3]
        ori_grid_b = ori_grid[:, None, :]  # [num_ori, 1, 3]

        # Displacement along the orientation
        invariant1 = (rel_pos * ori_grid_a).sum(
            dim=-1, keepdim=True
        )  # [num_edges, num_ori, 1]
        # Displacement orthogonal to the orientation (take norm in 3D)

        if self.dim == 2:
            invariant2 = (rel_pos - invariant1 * ori_grid_a).sum(
                dim=-1, keepdim=True
            )  # [num_edges, num_ori, 1]
        elif self.dim == 3:
            invariant2 = (rel_pos - invariant1 * ori_grid_a).norm(
                dim=-1, keepdim=True
            )  # [num_edges, num_ori, 1]
        # Relative orientation
        invariant3 = (ori_grid_a * ori_grid_b).sum(
            dim=-1, keepdim=True
        )  # [num_ori, num_ori, 1]
        # Stack into spatial and orientaiton invariants separately
        spatial_invariants = torch.cat(
            [invariant1, invariant2], dim=-1
        )  # [num_edges, num_ori, 2]
        orientation_invariants = invariant3  # [num_ori, num_ori, 1]

        return spatial_invariants, orientation_invariants

    def precompute_interaction_layers(self, edge_type, edge_type_kwargs, pos, batch):
        if edge_type == "fc":
            return fully_connected_edge_index(batch)
        elif edge_type == "knn":
            return knn_graph(pos, batch=batch, loop=True, **edge_type_kwargs).flip(0)

    def precompute_interaction_transition_layers(
        self, pos, batch, ori_grid, spatial_cond=None
    ):
        data_per_layer = []
        data_per_layer_up = []

        basis_idx = 0

        for i in range(0, self.num_scales):
            edge_type, edge_type_kwargs = self.edge_types[i], self.edge_types_kwargs[i]

            ratio = self.ratios[i]

            edge_index = self.precompute_interaction_layers(
                edge_type, edge_type_kwargs, pos, batch
            )

            spatial_invariants, orientation_invariants = self.compute_invariants(
                ori_grid, pos[edge_index[0]], pos[edge_index[1]]
            )

            if spatial_cond is not None:
                cond = spatial_cond[edge_index[0]].repeat(1, ori_grid.shape[-2], 1)
                spatial_invariants = torch.cat((spatial_invariants, cond), dim=-1)

            kernel_basis = self.basis_fn[basis_idx](spatial_invariants)
            fiber_kernel_basis = self.fiber_basis_fn[basis_idx](orientation_invariants)

            data_per_layer += [
                (kernel_basis, fiber_kernel_basis, edge_index, batch)
            ] * self.layers_per_scale[i]

            if self.up_sample:
                data_per_layer_up = [
                    (kernel_basis, fiber_kernel_basis, edge_index, batch)
                ] * self.layers_per_scale[i] + data_per_layer_up

            # Transition layer
            if ratio < 1.0 and i < self.num_scales - 1:
                edge_index, fps_pos, fps_batch = fps_edge_index(pos, batch, ratio=ratio)
                spatial_invariants, orientation_invariants = self.compute_invariants(
                    ori_grid, pos[edge_index[0]], fps_pos[edge_index[1]]
                )

                if spatial_cond is not None:
                    cond = spatial_cond[edge_index[0]].repeat(1, ori_grid.shape[-2], 1)
                    spatial_invariants = torch.cat((spatial_invariants, cond), dim=-1)

                kernel_basis = self.basis_fn[basis_idx](spatial_invariants)
                fiber_kernel_basis = self.fiber_basis_fn[basis_idx](
                    orientation_invariants
                )

                pos, batch = fps_pos, fps_batch

                data_per_layer.append(
                    (kernel_basis, fiber_kernel_basis, edge_index, batch)
                )

                if self.up_sample:
                    data_per_layer_up = [
                        (kernel_basis, fiber_kernel_basis, edge_index.flip(0), batch)
                    ] + data_per_layer_up

                basis_idx += 0 if self.global_basis else 1

        return data_per_layer + data_per_layer_up

    def forward(self, x, pos, edge_index, batch=None):
        ori_grid = self.ori_grid.type_as(pos)

        # Precompute the interaction and transition layers
        data_per_layer = self.precompute_interaction_transition_layers(
            pos,
            batch,
            ori_grid,
            spatial_cond=x[..., None, -1:] if self.last_feature_conditioning else None,
        )

        # Initial feature embeding
        x = self.x_embedder(x)
        x = x.unsqueeze(-2).repeat_interleave(ori_grid.shape[-2], dim=-2)  # [B*N,O,C]

        # Interaction + transition + readout
        readouts = []
        residuals = []

        for i in range(self.effective_num_layers):
            residual = x

            kernel, fiber_kernel, edge_index, batch = data_per_layer[i]
            x = self.interaction_layers[i](x, kernel, fiber_kernel, edge_index)

            if self.residual_connections:
                # downsampling, so save residual
                if residual.shape[0] > x.shape[0]:
                    residuals.append(residual)
                # upsampling, so take residual
                elif residual.shape[0] < x.shape[0]:
                    residual = residuals.pop(-1)
                    x = x + residual

            if self.read_out_layers[i] is not None:
                readouts.append(self.read_out_layers[i](x))
        readout = sum(readouts) / len(readouts)

        # Read out the scalar and vector part of the output
        readout_scalar, readout_vec = torch.split(
            readout, [self.output_dim, self.output_dim_vec], dim=-1
        )

        # Read out scalar and vector predictions
        output_scalar = readout_scalar.mean(dim=-2)  # [B*N,C]
        output_vector = (
            torch.einsum("boc,od->bcd", readout_vec, ori_grid) / ori_grid.shape[-2]
        )  # [B*N,C,3]

        if self.global_pooling:
            output_scalar = scatter_add(
                src=output_scalar, index=batch, dim_size=batch.max().item() + 1
            )
            output_vector = scatter_add(
                src=output_vector, index=batch, dim_size=batch.max().item() + 1
            )

        # Return predictions
        return output_scalar, output_vector
