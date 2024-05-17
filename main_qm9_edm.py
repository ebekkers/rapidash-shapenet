import argparse
import os
from tqdm import trange
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from datasets.qm9 import QM9Dataset, collate_fn
from datasets.qm9_bond_analyze import check_stability
from models.ponita import Ponita
from models.rapidash import Rapidash
from models.egnn import EGNN
from datasets.qm9_rdkit_utils import BasicMolecularMetrics


torch.set_float32_matmul_precision('medium')



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

    def forward(self, x, pos, edge_index, batch, sigma):
        sigma = sigma.reshape(-1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Simply condition by concatenating the noise level to the input
        x_in = torch.cat([c_in * x, c_noise.expand(x.shape[0], 1)], dim=-1)
        pos_in = c_in * pos
        if isinstance(self.model, Ponita) or isinstance(self.model, Rapidash):
            dx, dpos = self.model(x_in, pos_in, edge_index, batch)
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
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
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
        normalize_x_factor=4.,
        normalize_charge_factor = 8.,
        use_charges=False
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.normalize_x_factor = normalize_x_factor
        self.normalize_charge_factor = normalize_charge_factor
        self.use_charges = use_charges

    def __call__(
        self,
        net,
        inputs,
    ):
        
        # The point clouds and fully connected edge_index
        pos, x, edge_index, batch = inputs['pos'], inputs['x'], inputs['edge_index'], inputs['batch']
        edge_index = fully_connected_edge_index(batch)
        pos = subtract_mean(pos, batch)
        if self.use_charges:
            x[:,:-1] = x[:,:-1] / self.normalize_x_factor
            x[:,-1] = x[:,-1] / self.normalize_charge_factor
        else:
            x = x / self.normalize_x_factor

        # Random noise level per point cloud in the batch
        rnd_normal = torch.randn([batch.max() + 1, 1], device=pos.device, dtype=torch.float32)
        rnd_normal = rnd_normal[batch]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Noised inputs
        x_noisy = x + torch.randn_like(x) * sigma
        pos_noisy = pos + subtract_mean(torch.randn_like(pos), batch) * sigma

        # The network net is a the precondioned version of the model, including noise dependent skip-connections
        D_x, D_pos = net(x_noisy, pos_noisy, edge_index, batch, sigma)
        error_x = (D_x - x) ** 2
        error_pos = (D_pos - pos) ** 2
        loss = (weight * error_x).mean() + (weight * error_pos).mean()

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
        x_denoised, pos_denoised = net(x_hat, pos_hat, edge_index, batch, t_hat)
        dx_cur = (x_hat - x_denoised) / t_hat
        dpos_cur = (pos_hat - pos_denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * dx_cur
        pos_next = pos_hat + (t_next - t_hat) * dpos_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_denoised, pos_denoised = net(x_next, pos_next, edge_index, batch, t_next)
            dx_prime = (x_next - x_denoised) / t_next
            dpos_prime = (pos_next - pos_denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * dx_cur + 0.5 * dx_prime)
            pos_next = pos_hat + (t_next - t_hat) * (0.5 * dpos_cur + 0.5 * dpos_prime)

        steps.append((x_next.cpu(),pos_next.cpu()))

    if return_intermediate:
        return steps
    return x_next, pos_next


class DiffusionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.num_atoms_sampler = None

        in_channels_scalar = 5 + 1 + 1 * self.hparams.use_charges # num_atom_ty[es + 1 (for charges) + 1 (for noise level)
        in_channels_vec = 0
        out_channels_scalar = 5 + 1 * self.hparams.use_charges
        out_channels_vec = 1

        model_type = self.hparams.model
        if model_type == 'ponita':
            self.net = Ponita( input_dim      = in_channels_scalar + in_channels_vec,
                            hidden_dim        = self.hparams.hidden_dim,
                            output_dim        = out_channels_scalar,
                            num_layers        = self.hparams.layers,
                            output_dim_vec    = out_channels_vec,
                            num_ori           = self.hparams.num_ori,
                            basis_dim         = self.hparams.basis_dim,
                            degree            = self.hparams.degree,
                            widening_factor   = self.hparams.widening_factor,
                            layer_scale       = self.hparams.layer_scale,
                            task_level        = 'node',
                            multiple_readouts = self.hparams.multiple_readouts,
                            last_feature_conditioning=True,
                            attention         = self.hparams.attention)
        elif model_type == 'rapidash':
            self.net = Rapidash( input_dim    = in_channels_scalar + in_channels_vec - 1,  # As rapidash assumes dim+1 when last_feature_conditioning=True
                            hidden_dim        = self.hparams.hidden_dim,
                            output_dim        = out_channels_scalar,
                            num_layers        = self.hparams.layers,
                            output_dim_vec    = out_channels_vec,
                            num_ori           = self.hparams.num_ori,
                            basis_dim         = self.hparams.basis_dim,
                            degree            = self.hparams.degree,
                            widening_factor   = self.hparams.widening_factor,
                            layer_scale       = self.hparams.layer_scale,
                            task_level        = 'node',
                            multiple_readouts = self.hparams.multiple_readouts,
                            last_feature_conditioning=True,
                            attention         = self.hparams.attention)
        elif model_type == 'egnn':
            self.net = EGNN(in_node_nf = in_channels_scalar,
                            in_edge_nf = 0,
                            hidden_nf = self.hparams.hidden_dim,
                            act_fn=torch.nn.SiLU(), 
                            n_layers=self.hparams.layers,
                            attention=False,
                            out_node_nf=out_channels_scalar,
                            tanh=False, 
                            coords_range=15, 
                            norm_constant=0, 
                            inv_sublayers=2,
                            sin_embedding=False, 
                            normalization_factor=100, 
                            aggregation_method='sum')
        else: # Only 'ponita', 'rapidash' or 'egnn' implemented
            raise NotImplementedError
        self.model = EDMPrecond(self.net, sigma_data=self.hparams.sigma_data)
        self.criterion = EDMLoss(sigma_data=self.hparams.sigma_data, normalize_x_factor=self.hparams.normalize_x_factor, use_charges=self.hparams.use_charges, normalize_charge_factor=self.hparams.normalize_charge_factor)
    
    def set_num_atoms_sampler(self, num_atoms_sampler):
        self.num_atoms_sampler = num_atoms_sampler

    def init_molecule_analyzer(self, dataset_info, smiles_list):
        self.molecule_analyzer = BasicMolecularMetrics(dataset_info, smiles_list)

    def training_step(self, batch, batch_idx):
        loss, _ = self.criterion(self.model, batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch['batch'].max()+1)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.criterion(self.model, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch['batch'].max()+1)
        return loss
    
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 20 == 0:
            results_dict = self.validate(num_molecules=10000, batch_size=self.hparams.batch_size, rdkit_metrics=True)
            for key, value in results_dict.items():
                self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            results_dict = self.validate(num_molecules=self.hparams.batch_size, batch_size=self.hparams.batch_size, rdkit_metrics=False)
            for key, value in results_dict.items():
                self.log(key+" (estimate)", value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        return None
    
    def on_test_epoch_end(self):
        results_dict = self.validate(num_molecules=10000, batch_size=self.hparams.batch_size, rdkit_metrics=True)
        for key, value in results_dict.items():
            self.log(key+" (final)", value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return super().on_test_epoch_end()
  
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def sample(self, num_molecules=100):
        self.eval()
        with torch.no_grad():
            num_atoms = self.num_atoms_sampler(num_molecules).to(self.device)
            batch_indices = torch.arange(len(num_atoms), device=self.device)
            batch_idx = torch.repeat_interleave(batch_indices, num_atoms)
            pos_0 = torch.randn([len(batch_idx), 3], device=self.device)
            pos_0 = subtract_mean(pos_0, batch_idx)
            x_0 = torch.randn([len(batch_idx), 5 + 1 * self.hparams.use_charges], device=self.device)
            edge_index = fully_connected_edge_index(batch_idx)
            samples = edm_sampler(self.model, pos_0, x_0, edge_index, batch_idx, S_churn=self.hparams.S_churn, num_steps=self.hparams.num_steps, sigma_max=self.hparams.sigma_max)
        # Convert to list of molecules (!!!! AND WE SWAP THE ORDER TO POS, FEATURE, CHARGES !!!!)
        sample_list = []
        for i in range(batch_idx.max()+1):
            positions = samples[1][batch_idx==i]
            if self.hparams.use_charges:
                atom_types = samples[0][batch_idx==i,:-1].argmax(dim=-1)
                charges = (samples[0][batch_idx==i,-1] * self.hparams.normalize_charge_factor).round().long()
                sample_list.append((positions, atom_types, charges))
            else:
                atom_types = samples[0][batch_idx==i].argmax(dim=-1)
                sample_list.append((positions, atom_types))
        return sample_list
    
    def validate(self, num_molecules=10000, batch_size=100, rdkit_metrics=True):
        results_dict = {}

        # Generate molecules
        steps = num_molecules // batch_size
        molecules = []
        for _ in trange(steps):
            molecules += self.sample(batch_size)

        # Check how many of the molecules are stable
        count_mol_stable = 0
        count_atm_stable = 0
        count_mol_total = 0
        count_atm_total = 0
        for mol in molecules:  # mol is a tuple (positions, atom_types, charges), with charges=None if not used
            is_stable, nr_stable, total = check_stability(*mol)
            count_atm_stable += nr_stable
            count_atm_total += total
            count_mol_stable += int(is_stable)
            count_mol_total += 1
        atom_stability = 100. * count_atm_stable/count_atm_total
        molecule_stability =  100. * count_mol_stable/count_mol_total
        results_dict["atom_stability"] = atom_stability
        results_dict["molecule_stability"] = molecule_stability

        # Check RDKit metrics
        if rdkit_metrics:
            [validity, uniqueness, novelty], _ = self.molecule_analyzer.evaluate(molecules)
            discovery = validity * uniqueness * novelty
            results_dict["validity"] = validity
            results_dict["uniqueness"] = uniqueness
            results_dict["novelty"] = novelty
            results_dict["discovery"] = discovery

        # Return the metrics
        return results_dict
    
def load_data(args):
    train_set = QM9Dataset(split='train', root=args.root, use_charges=args.use_charges)
    val_set = QM9Dataset(split='val', root=args.root, use_charges=args.use_charges)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    num_atoms_sampler = train_set.NumAtomsSampler()
    smiles_list = []
    for i in range(len(train_set)):
        smiles_list.append(train_set[i]['smiles'])
    dataset_info = train_set.dataset_info
    return train_loader, val_loader, num_atoms_sampler, smiles_list, dataset_info


def main(args):
    # Seed everything
    pl.seed_everything(42)

    # Load the data
    train_loader, val_loader, num_atoms_sampler, smiles_list, dataset_info = load_data(args)

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
        logger = pl.loggers.WandbLogger(project="PONITA-QM9-EDM", name=args.model+"-EDM", config=args, save_dir='logs')
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
        model.set_num_atoms_sampler(num_atoms_sampler)
        model.init_molecule_analyzer(dataset_info, smiles_list)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        trainer.test(model, val_loader, ckpt_path = checkpoint_callback.best_model_path)
    else:   
        model = DiffusionModel.load_from_checkpoint(args.test_ckpt)
        model.set_num_atoms_sampler(num_atoms_sampler)
        model.init_molecule_analyzer(dataset_info, smiles_list)
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
    parser.add_argument('--batch_size', type=int, default=96)
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
    parser.add_argument('--root', type=str, default="./datasets/qm9_dataset")
    parser.add_argument('--use_charges', type=eval, default=True)
    
    # Graph connectivity settings (currently not in use, as it always uses fully connected graphs with self-connectiosn)
    parser.add_argument('--radius', type=eval, default=None)
    parser.add_argument('--loop', type=eval, default=True)
    
    # Model class
    parser.add_argument('--model', type=str, default="rapidash")  # ponita or egnn or rapidash (see shapenet for multi-scale specification)

    # PONTA model settings
    parser.add_argument('--num_ori', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--basis_dim', type=int, default=256)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--layers', type=eval, default=9)
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
