# preamble
# make sure all of the packages are installed in your conda environment, so that you don't get import errors
import os
import numpy as np
import torch
import argparse
import time
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torchvision
import torchvision.transforms as T
import random
import matplotlib.pyplot as plt
from scipy.special import ellipj


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# setting up random seeds
rng = np.random.default_rng()
set_deterministic(42)


# optimization
class LRScheduler(object):
    """
    Learning rate scheduler for the optimizer.
    """

    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def negative_cosine_similarity(p, z):
    """
    Negative cosine similarity.
    Contrastive learning loss with only *positive* terms.
    :param p: the first vector. p stands for prediction, as in BYOL and SimSiam
    :param z: the second vector. z stands for representation
    :return: -cosine_similarity(p, z)
    """
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


def info_nce(z1, z2, temperature=0.1):
    """
    Noise contrastive estimation loss.
    Contrastive learning loss with *both* positive and negative terms.
    :param z1: first vector
    :param z2: second vector
    :param temperature: how sharp the prediction task is
    :return: infoNCE(z1, z2)
    """
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    logits = z1 @ z2.T
    logits /= temperature
    n = z1.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


# pendulum dataset
# provided by Peter: ask him for issues with the dataset generation
def pendulum_train_gen(batch_size, traj_samples=100, noise=0., shuffle=True, check_energy=False, k2=None):
    t = np.expand_dims(np.linspace(0, 10 * traj_samples, num=traj_samples), axis=0).repeat(batch_size, axis=0)
    k2 = rng.uniform(size=(batch_size, 1)) if k2 is None else k2 * np.ones((batch_size, 1))  # energies (conserved)

    # finding what q (angle) and p (angular momentum) correspond to the time
    # derivation is a bit involved and optional to study
    # if interested, see https://en.wikipedia.org/wiki/Pendulum_(mathematics)# at section (Arbitrary-amplitude period)
    sn, cn, dn, _ = ellipj(t, k2)
    q = 2 * np.arcsin(np.sqrt(k2) * sn)
    p = 2 * np.sqrt(k2) * cn * dn / np.sqrt(1 - k2 * sn ** 2)
    data = np.stack((q, p), axis=-1)

    if shuffle:
        for x in data:
            rng.shuffle(x, axis=0)

    if check_energy:
        H = 0.5 * p ** 2 - np.cos(q) + 1
        diffH = H - 2 * k2
        print("max diffH = ", np.max(np.abs(diffH)))
        assert np.allclose(diffH, np.zeros_like(diffH))

    if noise > 0:
        data += noise * rng.standard_normal(size=data.shape)
    return k2, data


k2, data = pendulum_train_gen(2, noise=0)
for traj in data:
    plt.scatter(traj[:, 0][0:10], traj[:, 1][0:10], s=5.)
    plt.scatter(traj[:, 0][50:60], traj[:, 1][50:60], s=5.)

plt.xlabel(r"angle $\theta$")
plt.ylabel(r"angular momentum $L$")
plt.savefig('/home/darumen/Desktop/prototyping/dataset.png', dpi=300)
print('hello')
exit()


# transforms
class DatasetDefinitionTransform:
    def __init__(self, crop_scale_lower_bound, jitter_magnitude):
        def f(x):
            output = torch.zeros(3, 28, 28)
            output[random.randint(0, 2)] = x
            return output

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Lambda(f),  # RGB
                T.ToPILImage(),
                T.RandomResizedCrop(size=28, scale=(crop_scale_lower_bound, 1.0)),  # Crop
                T.ColorJitter(0.4 * jitter_magnitude, 0.4 * jitter_magnitude, 0.4 * jitter_magnitude,
                              0.1 * jitter_magnitude),
                T.ToTensor()
            ]
        )

    def __call__(self, x):
        return self.transform(x)


class ContrastiveLearningTransform:
    def __init__(self, crop_dat, crop_cl, jitter_dat, jitter_cl, type_tr):
        self.dataset_transform = DatasetDefinitionTransform(crop_dat, jitter_dat)
        self.crop_transform = T.RandomResizedCrop(size=28, scale=(crop_cl, 1.0))
        self.jitter_transform = T.ColorJitter(0.4 * jitter_cl, 0.4 * jitter_cl, 0.4 * jitter_cl, 0.1 * jitter_cl)
        self.tensor_transform = T.ToTensor()
        self.pil_transform = T.ToPILImage()
        self.composition = T.Compose(
            [
                self.pil_transform,
                self.crop_transform,
                self.jitter_transform,
                self.tensor_transform
            ]
        )
        self.crop = T.Compose(
            [
                self.pil_transform,
                self.crop_transform,
                self.tensor_transform
            ]
        )
        self.jitter = T.Compose(
            [
                self.pil_transform,
                self.jitter_transform,
                self.tensor_transform
            ]
        )
        self.type = type_tr

    def __call__(self, x):
        x = self.dataset_transform(x)
        if self.type == 'composition':
            return self.composition(x), self.composition(x), x
        elif self.type == 'decoupled':
            return self.crop(x), self.crop(x), self.jitter(x), self.jitter(x), x
        else:
            raise


# models
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, deeper=False, affine=False):
        super().__init__()
        list_layers = [nn.Linear(in_dim, hidden_dim),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        if deeper:
            list_layers += [nn.Linear(hidden_dim, hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(inplace=True)]
        if affine:
            last_bn = nn.BatchNorm1d(out_dim, eps=0, affine=False)
        else:
            last_bn = nn.BatchNorm1d(out_dim)
        list_layers += [nn.Linear(hidden_dim, out_dim),
                        last_bn]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Branch(nn.Module):
    def __init__(self, proj_dim, proj_hidden, deeper, affine, encoder=None):
        super().__init__()
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = torchvision.models.resnet18(zero_init_residual=True)
            self.encoder.fc = nn.Identity()  # replace the classification head with identity
        self.projector = ProjectionMLP(512, proj_hidden, proj_dim, affine=affine, deeper=deeper)
        self.net = nn.Sequential(
            self.encoder,
            self.projector
        )
        self.alpha_crop = nn.Parameter(0.5 * torch.ones([1]), requires_grad=True)
        self.alpha_jitter = nn.Parameter(0.5 * torch.ones([1]), requires_grad=True)

    def forward(self, x):
        return self.net(x)


# loops

def plotting_loop():
    dataset = torchvision.datasets.MNIST(
        '../Data', train=True, transform=DatasetDefinitionTransform(
            0.2, 1.0), download=True
    )
    cl_dataset = torchvision.datasets.MNIST(
        '../Data', train=True, transform=ContrastiveLearningTransform(
            0.2, 1.0, 1., 0.5, 'composition'), download=True
    )
    a = dataset.__getitem__(0)[0]
    b1 = cl_dataset.__getitem__(0)[0][0]
    b2 = cl_dataset.__getitem__(0)[0][1]

    plt.imshow(a.permute(1, 2, 0))
    plt.savefig('/home/darumen/Desktop/a.png', dpi=300)
    plt.imshow(b1.permute(1, 2, 0))
    plt.savefig('/home/darumen/Desktop/b1.png', dpi=300)
    plt.imshow(b2.permute(1, 2, 0))
    plt.savefig('/home/darumen/Desktop/b2.png', dpi=300)
    print("saved plots")


def training_loop(args, encoder=None):
    dataloader_kwargs = dict(drop_last=True, pin_memory=True, num_workers=16)

    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            '../Data', train=True, transform=ContrastiveLearningTransform(
                args.crop_dat, args.crop_cl, args.jitter_dat, args.jitter_cl, args.cl_aug_type), download=True
        ),
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            '../Data', train=True, transform=DatasetDefinitionTransform(
                args.crop_dat, args.jitter_dat), download=True
        ),
        shuffle=False,
        batch_size=args.bsz,
        **dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            '../Data', train=False, transform=DatasetDefinitionTransform(
                args.crop_dat, args.jitter_dat), download=True
        ),
        shuffle=False,
        batch_size=args.bsz,
        **dataloader_kwargs
    )

    dim_proj = [int(x) for x in args.dim_proj.split(',')]
    main_branch = Branch(dim_proj[1], dim_proj[0], args.deeper, args.affine, encoder=encoder).cuda()
    if args.dim_pred:
        h = PredictionMLP(dim_proj[0], args.dim_pred, dim_proj[0])

    # optimization
    optimizer = get_optimizer(
        name='sgd',
        momentum=0.9,
        lr=args.lr,
        model=main_branch,
        weight_decay=args.wd
    )
    lr_scheduler = LR_Scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=0,
        num_epochs=args.epochs,
        base_lr=args.lr * args.bsz / 256,
        final_lr=0,
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=True
    )
    if args.dim_pred:
        pred_optimizer = get_optimizer(
            name='sgd',
            momentum=0.9,
            lr=args.lr,
            model=h,
            weight_decay=args.wd
        )

    # macros
    b = main_branch.encoder
    proj = main_branch.projector

    # helpers
    def get_z(x):
        return proj(b(x))

    def apply_loss(z1, z2):
        if args.loss == 'square':
            loss = (z1 - z2).pow(2).sum()
        elif args.loss == 'mse':
            loss = F.mse_loss(z1, z2)
        elif args.loss == 'cosine':
            loss = F.cosine_similarity(z1, z2, dim=-1).mean()
        elif args.loss == 'infonce':
            loss = 0.5 * info_nce(z1, z2) + info_nce(z2, z1)
        elif args.loss == 'cosine_predictor':
            p1 = h(z1)
            p2 = h(z2)
            loss = negative_cosine_similarity(p1, z2) / 2 + negative_cosine_similarity(p2, z1) / 2
        return loss

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'knn_and_loss.log'), 'w')
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))

    knn_acc = knn_loop(b, memory_loader, test_loader)
    line_to_print = f'epoch: {0} | knn_acc: {knn_acc:.3f}  | time_elapsed: {time.time() - start:.3f}'
    file_to_update.write(line_to_print + '\n')
    file_to_update.flush()
    print(line_to_print)

    # training
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()
        if args.dim_pred:
            h.train()

        # epoch
        for it, (inputs, y) in enumerate(train_loader):
            # zero grad
            main_branch.zero_grad()
            if args.dim_pred:
                h.zero_grad()

            # forward pass
            if args.cl_train_type == 'standard':
                assert args.cl_aug_type == 'composition'
                z1 = get_z(inputs[0].cuda())
                z2 = get_z(inputs[1].cuda())
                loss = apply_loss(z1, z2)
            elif args.cl_train_type == 'stochastic':
                assert args.cl_aug_type == 'decoupled'
                choice = random.randint(0, 1)
                z1 = get_z(inputs[choice * 2 + 0].cuda())
                z2 = get_z(inputs[choice * 2 + 1].cuda())
                loss = apply_loss(z1, z2)
            elif args.cl_train_type in ['joint', 'trainable_joint']:
                assert args.cl_aug_type == 'decoupled'
                z1_crop = get_z(inputs[0].cuda())
                z2_crop = get_z(inputs[1].cuda())
                z1_jitter = get_z(inputs[2].cuda())
                z2_jitter = get_z(inputs[3].cuda())
                loss_crop = apply_loss(z1_crop, z2_crop)
                loss_jitter = apply_loss(z1_jitter, z2_jitter)
                if args.cl_train_type == 'joint':
                    loss = 0.5 * loss_crop + 0.5 * loss_jitter
                else:
                    assert args.loss == 'infonce'  # otherwise, it will learn zero alphas
                    loss = main_branch.alpha_crop * loss_crop + main_branch.alpha_jitter * loss_jitter
            else:
                pass

            # optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if args.dim_pred:
                pred_optimizer.step()

        if e % args.save_every == 0:
            knn_acc = knn_loop(b, memory_loader, test_loader)
            torch.save(dict(epoch=0, state_dict=main_branch.state_dict()),
                       os.path.join(args.path_dir, f'{e}.pth'))
            line_to_print = f'epoch: {e} | knn_acc: {knn_acc:.3f} | loss: {loss.item():.3f} | time_elapsed: {time.time() - start:.3f}'
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
            print(line_to_print)

    file_to_update.close()
    return main_branch.encoder


def main(args):
    if args.mode == 'training':
        training_loop(args)
    elif args.mode == 'analysis':
        analysis_loop(args)
    elif args.mode == 'plotting':
        plotting_loop(args)
    else:
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_proj', default='1024,128', type=str)
    parser.add_argument('--dim_pred', default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--loss', default='infonce', type=str)
    parser.add_argument('--affine', action='store_false')
    parser.add_argument('--deeper', action='store_false')
    parser.add_argument('--save_every', default=10, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--mode', default='plot', type=str,
                        choices=['plotting', 'training', 'analysis'])
    parser.add_argument('--path_dir', default='../output/pendulum', type=str)

    args = parser.parse_args()
    main(args)
