
import torch

NPOINTS = 2048


from main_shapenet_edm import DiffusionModel

model = DiffusionModel.load_from_checkpoint("~/halfwaytraining.ckpt")
# Overwrite sampling settings if you wanted to
# model.hparams.S_churn = args.S_churn
# model.hparams.sigma_max = args.sigma_max
# model.hparams.num_steps = args.num_steps
# model.hparams.batch_size = args.batch_size

num_samples = 10
# Returns a list of samples (pos, x) where x is just ones
samples = model.sample(num_samples)
samples = torch.stack([pos for pos, _ in samples])

print(samples.shape)