# preamble
# make sure all of the packages are installed in your conda environment, so that you don't get import errors
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import torch
import argparse
import time
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import random
import matplotlib.pyplot as plt
from scipy.special import ellipj


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print("Deterministic with seed = " + str(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def most_recent_file(folder):
    max_time = 0
    max_file = ""
    for dirname, subdirs, files in os.walk(folder):
        for fname in files:
            full_path = os.path.join(dirname, fname)
            time = os.stat(full_path).st_mtime
            if time > max_time:
                max_time = time
                max_file = full_path

    return max_file

# optimization
class LRScheduler(object):
    """
    Learning rate scheduler for the optimizer.

    Warmup increases to base linearly, while base decays to final using cosine
    (quick immediate dropoff that smoothly decreases.)
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
            lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


# loss functions for contrastive learning
def negative_cosine_similarity(p, z):
    """
    Negative cosine similarity. (Cosine similarity is the cosine of the angle
    between two vectors of arbitrary length.)

    Contrastive learning loss with only *positive* terms.
    :param p: the first vector. p stands for prediction, as in BYOL and SimSiam
    :param z: the second vector. z stands for representation
    :return: -cosine_similarity(p, z)
    """
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean() # detach removes gradient tracking


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
    labels = torch.arange(0, n, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


# dataset
def pendulum_train_gen(batch_size, traj_samples=100, noise=0., shuffle=True, check_energy=False, k2=None):
    """
    pendulum dataset generation
    provided by Peter: ask him for issues with the dataset generation
    """
    # setting up random seeds
    rng = np.random.default_rng()

    t = rng.uniform(0, 10. * traj_samples, size=(batch_size, traj_samples))
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


class PendulumDataset(torch.utils.data.Dataset):
    def __init__(self, size=10240, trajectory_length=100, noise=0.00):
        self.size = size
        self.k2, self.data = pendulum_train_gen(size, noise=noise)
        self.trajectory_length = trajectory_length

    def __getitem__(self, idx):
        i = random.randint(0, self.trajectory_length - 1)
        j = random.randint(0, self.trajectory_length - 1)
        return [
            torch.FloatTensor(self.data[idx][i]),
            torch.FloatTensor(self.data[idx][j]),
            self.k2[idx]
        ]  # [first_view, second_view, energy]

    def __len__(self):
        return self.size


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
            self.encoder = nn.Sequential(
                nn.Linear(2, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )  # simple encoder to start with
            # self.encoder = torchvision.models.resnet18(zero_init_residual=True)
            # TODO: replace the encoder with CNN once we have 2D dataset
            # self.encoder.fc = nn.Identity()  # replace the classification head with identity
        # self.projector = ProjectionMLP(32, 64, 32, affine=affine, deeper=deeper)
        self.projector = nn.Identity()  # TODO: to keep it simple, for now we will not use projector
        self.net = nn.Sequential(
            self.encoder,
            self.projector
        )

    def forward(self, x):
        return self.net(x)


# loops

def plotting_loop(args):
    # TODO: can be used to plot the data in 2D.
    k2, data = pendulum_train_gen(100, noise=0.05)
    for traj in data:
        plt.scatter(traj[:, 0], traj[:, 1], s=5.)
    plt.xlabel(r"angle $\theta$")
    plt.ylabel(r"angular momentum $L$")
    plt.savefig(os.path.join(args.path_dir, 'dataset.png'), dpi=300)

def supervised_loop(args):
    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=4)
    train_loader = torch.utils.data.DataLoader(
        dataset=PendulumDataset(),
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    ) # check if the data is actually different?
    print("Completed data loading")

    # model
    dim_proj = [int(x) for x in args.dim_proj.split(',')]
    main_branch = Branch(dim_proj[1], dim_proj[0], args.deeper, args.affine, encoder=encoder)
    if args.dim_pred:
        h = PredictionMLP(dim_proj[0], args.dim_pred, dim_proj[0])

    # optimization
    dim_proj = [int(x) for x in args.dim_proj.split(',')]
    main_branch = Branch(dim_proj[1], dim_proj[0], args.deeper, args.affine, encoder=encoder)
    if args.dim_pred:
        h = PredictionMLP(dim_proj[0], args.dim_pred, dim_proj[0])

    # optimization
    optimizer = torch.optim.SGD(
        main_branch.parameters(),
        momentum=0.9,
        lr=args.lr,
        weight_decay=args.wd
    )
    lr_scheduler = LRScheduler(
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
        pred_optimizer = torch.optim.SGD(
            h.parameters(),
            momentum=0.9,
            lr=args.lr,
            weight_decay=args.wd
        )

    # macros
    b = main_branch.encoder

    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'training_loss.log'), 'w')
    torch.save(dict(epoch=0, state_dict=b.state_dict()), os.path.join(args.path_dir, '0.pth'))

    for e in range(1, args.epochs + 1):
        # declaring train
        b.train()

        # epoch
        for it, (x1, x2, energy) in enumerate(train_loader):
            # zero grad
            b.zero_grad()

            # forward pass
            loss = torch.nn.MSELoss(b(x1), energy)

            # optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if args.dim_pred:
                pred_optimizer.step()

        if e % args.save_every == 0:
            torch.save(dict(epoch=0, state_dict=b.state_dict()), os.path.join(args.path_dir, f'{e}.pth'))
            line_to_print = f'epoch: {e} | loss: {loss.item():.3f} | time_elapsed: {time.time() - start:.3f}'
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
            print(line_to_print)

        file_to_update.close()
        return b


def training_loop(args, encoder=None):
    # dataset
    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=4)
    train_loader = torch.utils.data.DataLoader(
        dataset=PendulumDataset(),
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    ) # check if the data is actually different?
    print("Completed data loading")

    # model
    dim_proj = [int(x) for x in args.dim_proj.split(',')]
    main_branch = Branch(dim_proj[1], dim_proj[0], args.deeper, args.affine, encoder=encoder)
    if args.dim_pred:
        h = PredictionMLP(dim_proj[0], args.dim_pred, dim_proj[0])

    # optimization
    optimizer = torch.optim.SGD(
        main_branch.parameters(),
        momentum=0.9,
        lr=args.lr,
        weight_decay=args.wd
    )
    lr_scheduler = LRScheduler(
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
        pred_optimizer = torch.optim.SGD(
            h.parameters(),
            momentum=0.9,
            lr=args.lr,
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
        elif args.loss == 'infonce':
            loss = 0.5 * info_nce(z1, z2) + 0.5 * info_nce(z2, z1)
        elif args.loss == 'cosine_predictor':
            p1 = h(z1)
            p2 = h(z2)
            loss = negative_cosine_similarity(p1, z2) / 2 + negative_cosine_similarity(p2, z1) / 2
        return loss
    print("Setup complete")

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'training_loss.log'), 'w')
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))

    # training
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()
        if args.dim_pred:
            h.train()
        print("train")

        # epoch
        for it, (x1, x2, energy) in enumerate(train_loader):
            print("data")
            # zero grad
            main_branch.zero_grad()
            if args.dim_pred:
                h.zero_grad()

            # forward pass
            z1 = get_z(x1)
            z2 = get_z(x2)
            loss = apply_loss(z1, z2)

            # optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if args.dim_pred:
                pred_optimizer.step()

        if e % args.save_every == 0:
            torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, f'{e}.pth'))
            line_to_print = f'epoch: {e} | loss: {loss.item():.3f} | time_elapsed: {time.time() - start:.3f}'
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
            print(line_to_print)

    file_to_update.close()
    return main_branch.encoder


def analysis_loop(args):
    # TODO: can be used to study if the neural network has learned the conserved quantity.
    dim_proj = [int(x) for x in args.dim_proj.split(',')]
    branch = Branch(dim_proj[1], dim_proj[0], args.deeper, args.affine, encoder=encoder)
    if args.dim_pred:
        h = PredictionMLP(dim_proj[0], args.dim_pred, dim_proj[0])

    load_file = args.load_file
    if load_file == "recent":
        load_file = most_recent_file(args.path_dir)
    branch.load_state_dict(torch.load(load_file))
    branch.eval()
    b = branch.encoder
    print("Completed model loading")

    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        dataset=PendulumDataset(size=args.test_size),
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    ) # check if the data is actually different?
    print("Completed data loading")

    coded = np.array((test_size))
    energies = np.array((test_size))
    for it, (x1, x2, energy) in enumerate(train_loader):
        coded[it] = b(x1)
        energies[it] = energy

    return coded, energies


def main(args):
    set_deterministic(42)
    if args.mode == 'training':
        training_loop(args)
    elif args.mode == 'analysis':
        analysis_loop(args)
    elif args.mode == 'plotting':
        plotting_loop(args)
    elif args.mode == 'supervised':
        supervised_loop(args)
    else:
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_proj', default='1024,128', type=str)
    parser.add_argument('--dim_pred', default=None, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--loss', default='infonce', type=str)
    parser.add_argument('--affine', action='store_false')
    parser.add_argument('--deeper', action='store_false')
    parser.add_argument('--save_every', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--mode', default='training', type=str,
                        choices=['plotting', 'training', 'analysis', 'supervised'])
    parser.add_argument('--path_dir', default='../output/pendulum', type=str)
    parser.add_argument('--load_file', default='recent', type=str)
    parser.add_argument('--test_size', default=1000, type=int)

    args = parser.parse_args()
    main(args)
