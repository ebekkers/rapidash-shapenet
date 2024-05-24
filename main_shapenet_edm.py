import argparse
import os
from tqdm import trange
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from datasets.shapenet import ShapeNetDataset, collate_fn
from datasets.qm9_bond_analyze import check_stability
# from models.ponita_ms import Ponita
from models.rapidash import Rapidash
from models.egnn import EGNN
from datasets.qm9_rdkit_utils import BasicMolecularMetrics
import wandb

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from chamferdist import ChamferDistance

NPOINTS = 2048

torch.set_float32_matmul_precision('medium')



def svd_normalization(point_clouds):
    B, N, D = point_clouds.shape
    centered_point_clouds = point_clouds - point_clouds.mean(dim=1, keepdim=True)
    cov_matrix = centered_point_clouds.transpose(1, 2) @ centered_point_clouds / (N - 1)
    U, S, V = torch.svd(cov_matrix)
    normalized_point_clouds = torch.bmm(centered_point_clouds, V)
    return normalized_point_clouds



class RandomSOd(torch.nn.Module):
        def __init__(self, d):
            """
            Initializes the RandomRotationGenerator.
            Args:
            - d (int): The dimension of the rotation matrices (2 or 3).
            """
            super(RandomSOd, self).__init__()
            assert d in [2, 3], "d must be 2 or 3."
            self.d = d

        def forward(self, n=None):
            """
            Generates random rotation matrices.
            Args:
            - n (int, optional): The number of rotation matrices to generate. If None, generates a single matrix.
            
            Returns:
            - Tensor: A tensor of shape [n, d, d] containing n rotation matrices, or [d, d] if n is None.
            """
            if self.d == 2:
                return self._generate_2d(n)
            else:
                return self._generate_3d(n)
        
        def _generate_2d(self, n):
            theta = torch.rand(n) * 2 * torch.pi if n else torch.rand(1) * 2 * torch.pi
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            rotation_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
            if n:
                return rotation_matrix.view(n, 2, 2)
            return rotation_matrix.view(2, 2)

        def _generate_3d(self, n):
            q = torch.randn(n, 4) if n else torch.randn(4)
            q = q / torch.norm(q, dim=-1, keepdim=True)
            q0, q1, q2, q3 = q.unbind(-1)
            rotation_matrix = torch.stack([
                1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
                2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1),
                2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)
            ], dim=-1)
            if n:
                return rotation_matrix.view(n, 3, 3)
            return rotation_matrix.view(3, 3)

        
def scatter_mean(src, index, dim, dim_size):
    # Step 1: Perform scatter add (sum)
    out_shape = [dim_size] + list(src.shape[1:])
    out_sum = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    dims_to_add = src.dim() - index.dim()
    for _ in range(dims_to_add):
        index = index.unsqueeze(-1)
    index_expanded = index.expand_as(src)
    out_sum.scatter_add_(dim, index_expanded, src)
    
    # Step 2: Count occurrences of each index to calculate the mean
    ones = torch.ones_like(src)
    out_count = torch.zeros(out_shape, dtype=torch.float, device=src.device)
    out_count.scatter_add_(dim, index_expanded, ones)
    out_count[out_count == 0] = 1  # Avoid division by zero
    
    # Calculate mean by dividing sum by count
    out_mean = out_sum / out_count

    return out_mean

def fully_connected_edge_index(batch_idx):
    edge_indices = []
    for batch_num in torch.unique(batch_idx):
        # Find indices of nodes in the current batch
        node_indices = torch.where(batch_idx == batch_num)[0]
        grid = torch.meshgrid(node_indices, node_indices, indexing='ij')
        edge_indices.append(torch.stack([grid[0].reshape(-1), grid[1].reshape(-1)], dim=0))
    edge_index = torch.cat(edge_indices, dim=1)
    return edge_index

def subtract_mean(pos, batch):
    means = scatter_mean(src=pos, index=batch, dim=0, dim_size=batch.max().item()+1)
    return pos - means[batch]


class EDMPrecond(torch.nn.Module):
    def __init__(self,
        model,
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, pos, edge_index, batch, sigma, rot=None):
        sigma = sigma.reshape(-1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Simply condition by concatenating the noise level to the input  
        # x_in = torch.cat([c_in * x, c_noise.expand(x.shape[0], 1)], dim=-1)
        # Specific for shapenet: no node feature diffusion (and thus also no conditioning)
        x = torch.ones_like(x)
        x_in = torch.cat([x, c_noise.expand(x.shape[0], 1)], dim=-1)
        pos_in = c_in * pos
        if rot is not None:
             # Transpose because the columns are the reference vectors (the basis)
             # vec should be a shape [batch, num_vec, 3] tensor
             # rot is in that sence formated as [3, num_vec], hence the transpose
            vec_in = rot.transpose(-2,-1).unsqueeze(0).expand(x_in.shape[0], -1, -1) 
        else:
            vec_in = None
        if isinstance(self.model, Rapidash):
            dx, dpos = self.model(x_in, pos_in, edge_index, batch, vec=vec_in)
            dpos = subtract_mean(dpos, batch)
            # dx and dpos denote denoising gradients, relative to the input,
            # so predicted denoised version of pre-conditioned input is:
            F_x = x_in[..., :-1] - dx
            F_pos = pos_in - dpos[:,0,:]
        elif isinstance(self.model, EGNN):
            dx, F_pos = self.model(x_in, pos_in, edge_index, None)
            F_x = x_in[..., :-1] - dx
            F_pos = pos_in + subtract_mean(F_pos - pos_in, batch)
        else: # Other models not supported
            raise NotImplementedError
        
        # Noise dependent skip connection
        # D_x = c_skip * x + c_out * F_x.to(torch.float32)
        D_x = torch.ones_like(x)  # Shapenet fix: has no node features
        D_pos = c_skip * pos + c_out * F_pos.to(torch.float32)
        return D_x, D_pos

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class EDMLoss:
    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5, 
        normalize_x_factor=4.
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.normalize_x_factor = normalize_x_factor
        self.chamferdistance = ChamferDistance()

    def __call__(
        self,
        net,
        inputs,
    ):
        
        # The point clouds and fully connected edge_index
        pos, x, edge_index, batch, rot = inputs['pos'], inputs['x'], inputs['edge_index'], inputs['batch'], inputs['rot']
        pos = subtract_mean(pos, batch)
        x = x / self.normalize_x_factor

        # Random noise level per point cloud in the batch
        rnd_normal = torch.randn([batch.max() + 1, 1], device=pos.device, dtype=torch.float32)
        rnd_normal = rnd_normal[batch]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Noised inputs
        # x_noisy = x + torch.randn_like(x) * sigma
        x_noisy = torch.ones_like(x)
        pos_noisy = pos + subtract_mean(torch.randn_like(pos), batch) * sigma

        # The network net is a the precondioned version of the model, including noise dependent skip-connections
        # D_x and D_pos are the denoised features and positions
        # For shapenet we ignore the features (which are one anyway)
        # Todo: take normal vectors as features?
        D_x, D_pos = net(x_noisy, pos_noisy, edge_index, batch, sigma, rot)
        error_pos = (D_pos - pos) ** 2
        loss = (weight * error_pos).mean()

        return loss, (D_x, D_pos)


def edm_sampler(
    net,
    pos_0,
    x_0,
    edge_index,
    batch,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=20,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    return_intermediate=False,
    random_rot=False
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=pos_0.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next, pos_next = x_0 * t_steps[0], pos_0 * t_steps[0]
    steps = [(x_next.cpu(), pos_next.cpu())]
    
    if random_rot:
        rotation_generator = RandomSOd(3)
        rot = rotation_generator().type_as(pos_0)
    else:  
        rot = torch.eye(3, device=pos_0.device)
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur, pos_cur = x_next, pos_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        pos_hat = pos_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(pos_cur)

        # Euler step.
        x_hat = torch.ones_like(x_hat)  # Shapenet fix: node features should always be ones
        x_denoised, pos_denoised = net(x_hat, pos_hat, edge_index, batch, t_hat, rot)
        dx_cur = (x_hat - x_denoised) / t_hat
        dpos_cur = (pos_hat - pos_denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * dx_cur
        pos_next = pos_hat + (t_next - t_hat) * dpos_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_next = torch.ones_like(x_next)  # Shapenet fix: node features should always be ones
            x_denoised, pos_denoised = net(x_next, pos_next, edge_index, batch, t_next, rot)
            x_denoised = torch.ones_like(x_denoised)  # Shapenet fix: node features should always be ones
            dx_prime = (x_next - x_denoised) / t_next
            dpos_prime = (pos_next - pos_denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * dx_cur + 0.5 * dx_prime)
            x_next = torch.ones_like(x_next)  # Shapenet fix: node features should always be ones
            pos_next = pos_hat + (t_next - t_hat) * (0.5 * dpos_cur + 0.5 * dpos_prime)

        steps.append((x_next.cpu(),pos_next.cpu()))

    if return_intermediate:
        return steps
    return x_next, pos_next


class DiffusionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # For rotation augmentations during training and testing
        self.rotation_generator = RandomSOd(3)

        in_channels_scalar = 1
        in_channels_vec = 3
        out_channels_scalar = 1
        out_channels_vec = 1

        model_type = self.hparams.model
        if model_type == 'rapidash':
            self.net = Rapidash(input_dim         = in_channels_scalar + in_channels_vec,
                                hidden_dim        = self.hparams.hidden_dim,
                                output_dim        = out_channels_scalar,
                                num_layers        = self.hparams.layers,
                                edge_types        = self.hparams.edge_types,
                                ratios            = self.hparams.ratios,
                                output_dim_vec    = out_channels_vec,
                                dim               = 3,
                                num_ori           = self.hparams.num_ori,
                                basis_dim         = self.hparams.basis_dim,
                                degree            = self.hparams.degree,
                                widening_factor   = self.hparams.widening_factor,
                                layer_scale       = self.hparams.layer_scale,
                                task_level        = 'node',
                                multiple_readouts = self.hparams.multiple_readouts,
                                last_feature_conditioning=True,
                                attention         = self.hparams.attention,
                                fully_connected   = True,
                                residual_connections=True,
                                global_basis=False)
        else: # Currently only 'rapidash' for shapenet
            raise NotImplementedError
        self.model = EDMPrecond(self.net, sigma_data=self.hparams.sigma_data)
        self.criterion = EDMLoss(sigma_data=self.hparams.sigma_data, normalize_x_factor=self.hparams.normalize_x_factor)

    def training_step(self, batch, batch_idx):
        if self.hparams.train_augm:
            rot = self.rotation_generator().type_as(batch['pos'])
            batch['pos'] = torch.einsum('ij, bj->bi', rot, batch['pos'])
            batch['rot'] = rot
        else:
            batch['rot'] = torch.eye(3, device=batch['pos'].device)
        loss, _ = self.criterion(self.model, batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch['batch'].max()+1)
        return loss

    def validation_step(self, batch, batch_idx):
        batch['rot'] = torch.eye(3, device=batch['pos'].device)
        loss, _ = self.criterion(self.model, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch['batch'].max()+1)
        rand_idx = np.random.randint(0, (batch['batch'].max()+1).item())
        wandb.log({"reference_point_cloud": wandb.Object3D(batch['pos'][batch['batch']==rand_idx].cpu().numpy())})
        return loss
    
    def on_validation_epoch_end(self):
        # Just to have some reference visualization
        samples = self.sample(self.hparams.batch_size, random_rot=False)
        rand_idx = np.random.randint(0, len(samples))
        # shape = samples[rand_idx][0].cpu().numpy()
        shape = samples[rand_idx][0]
        # shape = svd_normalization(shape[None])[0].cpu().numpy()
        shape = shape.cpu().numpy()
        wandb.log({"val_gen_point_cloud": wandb.Object3D(shape)})
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        return None
    
    def on_test_epoch_end(self):
        samples = []
        for i in range(10):
            samples += self.sample(self.hparams.batch_size, random_rot=False)
        # samples = self.sample(self.hparams.batch_size)
        for i, (pos, x) in enumerate(samples):
            wandb.log({"test_point_cloud": wandb.Object3D(pos.cpu().numpy())})
        return super().on_test_epoch_end()
  
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def sample(self, num_molecules=100, random_rot=False):
        self.eval()
        with torch.no_grad():
            num_atoms = torch.tensor([NPOINTS] * num_molecules, dtype=torch.long, device=self.device)
            batch_indices = torch.arange(len(num_atoms), device=self.device)
            batch_idx = torch.repeat_interleave(batch_indices, num_atoms)
            pos_0 = torch.randn([len(batch_idx), 3], device=self.device)
            pos_0 = subtract_mean(pos_0, batch_idx)
            x_0 = torch.randn([len(batch_idx), 1], device=self.device)
            x_0 = torch.ones_like(x_0)
            edge_index = None
            samples = edm_sampler(self.model, pos_0, x_0, edge_index, batch_idx, S_churn=self.hparams.S_churn, num_steps=self.hparams.num_steps, sigma_max=self.hparams.sigma_max, random_rot=random_rot)
        # Convert to list of molecules (!!!! AND WE SWAP THE ORDER TO POS, FEATURE, CHARGES !!!!)
        sample_list = []
        for i in range(batch_idx.max()+1):
            positions = samples[1][batch_idx==i]
            features = samples[0][batch_idx==i]
            sample_list.append((positions, features))
        return sample_list
    
    
def load_data(args):
    train_set = ShapeNetDataset(root=args.root, split='train', npoints=NPOINTS, categories=['Airplane'])
    val_set = ShapeNetDataset(root=args.root, split='val', npoints=NPOINTS, categories=['Airplane'])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    return train_loader, val_loader


def main(args):
    # Seed everything
    pl.seed_everything(42)

    # Load the data
    train_loader, val_loader = load_data(args)

    # Hardware settings
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()

    # Logging settings
    if args.log:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(project="PONITA-ShapeNet-EDM", name=args.model+"-EDM", config=args, save_dir=save_dir)
    else:
        logger = None

    # Pytorch lightning call backs and trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode = 'min', every_n_epochs = 1, save_last=True)
    callbacks = [checkpoint_callback]
    if args.log: callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, gradient_clip_val=0.5, 
                         accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar)

    # Do the training or testing
    if args.test_ckpt is None:
        model = DiffusionModel(args)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        trainer.test(model, val_loader, ckpt_path = checkpoint_callback.best_model_path)
    else:   
        model = DiffusionModel.load_from_checkpoint(args.test_ckpt)
        model.hparams.S_churn = args.S_churn
        model.hparams.sigma_max = args.sigma_max
        model.hparams.num_steps = args.num_steps
        model.hparams.batch_size = args.batch_size
        trainer.test(model, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments
    
    # Run parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-12)
    parser.add_argument('--log', type=eval, default=True)
    parser.add_argument('--enable_progress_bar', type=eval, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_ckpt', type=str, default=None)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    
    # Train settings
    parser.add_argument('--train_augm', type=eval, default=True)
    
    # QM9 Dataset
    parser.add_argument('--root', type=str, default="./datasets/shapenet")
    
    # Model class
    parser.add_argument('--model', type=str, default="rapidash")  # ponita or egnn

    # PONTA model settings
    parser.add_argument('--num_ori', type=int, default=12)
    parser.add_argument('--hidden_dim', type=eval, default=[256,256,256,256,256])
    parser.add_argument('--basis_dim', type=int, default=256)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--layers', type=eval, default=[0, 1, 1, 1, 1])
    parser.add_argument('--edge_types', type=eval, default=["knn-8","knn-8", "knn-8", "knn-8", "fc"])
    parser.add_argument('--ratios', type=eval, default=[0.25, 0.25, 0.25, 0.25])
    parser.add_argument('--widening_factor', type=int, default=4)
    parser.add_argument('--layer_scale', type=eval, default=None)
    parser.add_argument('--multiple_readouts', type=eval, default=False)
    parser.add_argument('--attention', type=eval, default=False)

    # Diffusion model settings
    parser.add_argument('--S_churn', type=float, default=10)
    parser.add_argument('--sigma_max', type=float, default=1)
    parser.add_argument('--num_steps', type=int, default=40)
    parser.add_argument('--sigma_data', type=float, default=1)
    parser.add_argument('--normalize_x_factor', type=float, default=4.0)
    parser.add_argument('--normalize_charge_factor', type=float, default=8.0)
    
    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=1, type=int)
    
    # Arg parser
    args = parser.parse_args()

    main(args)
