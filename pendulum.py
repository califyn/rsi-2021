# python imports
import os
import subprocess
import argparse
import time
import random
from PIL import Image
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# sci suite
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj

# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as T

verbose = False

def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print("Deterministic with seed = " + str(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def most_recent_file(folder, ext=""):
    max_time = 0
    max_file = ""
    for dirname, subdirs, files in os.walk(folder):
        for fname in files:
            full_path = os.path.join(dirname, fname)
            time = os.stat(full_path).st_mtime
            if time > max_time and full_path.endswith(ext):
                max_time = time
                max_file = full_path

    return max_file

class LRScheduler(object):
    """
    Learning rate scheduler for the optimizer.

    Warmup increases to base linearly, while base decays to final using cosine.
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

def euclidean_dist(z1, z2):
    n = z1.shape[0]
    return 2 * z1 @ z2.T - torch.diagonal(z1 @ z1.T).repeat(n).reshape((n,n)).T - torch.diagonal(z2 @ z2.T).repeat(n).reshape((n,n))

def simsiam_loss(p, z, distance="cosine"):
    """
    Negative cosine similarity. (Cosine similarity is the cosine of the angle
    between two vectors of arbitrary length.)

    Contrastive learning loss with only *positive* terms.
    :param p: the first vector. p stands for prediction, as in BYOL and SimSiam
    :param z: the second vector. z stands for representation
    :return: -cosine_similarity(p, z)
    """
    if distance == "euclidean":
        return - torch.diagonal(euclidean_dist(p, z.detach())).mean()
    elif distance == "cosine":
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


def info_nce(z1, z2, temperature=0.1, distance="cosine"):
    """
    Noise contrastive estimation loss.
    Contrastive learning loss with *both* positive and negative terms.
    :param z1: first vector
    :param z2: second vector
    :param temperature: how sharp the prediction task is
    :return: infoNCE(z1, z2)
    """
    if z1.size()[1] <= 1 and distance == "cosine":
        raise UserWarning('InfoNCE loss has only one dimension, add more dimensions')

    if distance == "cosine":
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        logits = z1 @ z2.T
    elif distance == "euclidean":
        logits = euclidean_dist(z1, z2)

    logits /= temperature
    if torch.cuda.is_available():
        logits = logits.cuda()

    n = z1.shape[0]
    labels = torch.arange(0, n, dtype=torch.long)
    if torch.cuda.is_available():
        labels = labels.cuda()

    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def pendulum_train_gen(data_size, traj_samples=10, noise=0., uniform=False,
        shuffle=True, check_energy=False, k2=None, image=True,
        blur=False, img_size=32, diff_time=0.5, bob_size=1, continuous=False,
        gaps=[-1,-1], crop=1.0, crop_c=[-1,-1], t_window=-1, t_range=-1, mink=0, maxk=1):
    """
    pendulum dataset generation
    :param data_size: number of pendulums
    :param traj_samples: number of samples per pendulum
    :param noise: Gaussian noise
    :param shuffle: if shuffle data
    :param check_energy: check the energy (numerical mode only)
    :param k2: specify energy
    :param image: use image/graphical mode
    :param blur: whether to use motion blur mode (otherwise, use two-frame mode) [not implemented]
    :param img_size: size of (square) image (graphical mode only)
    :param diff_time: time difference between two images (graphical mode only)
    :param bob_size: bob = square of length bob_size * 2 + 1
    :param continuous: whether to use continuous generation [not implemented]
    :param gaps: generate gaps in data (graphical mode only)
                    (-1 if not used; otherwise [# gaps, total gap len as proportion])
    :param crop: proportion of image cutout returned
    :param crop_c: center of the image cutout
    :param time_window: specify a certain window of time
    :param time_range: specify max window of time per energy
            (-1 if not used)
    :param mink: minimum energy
    :param maxk: maximum energy
    :return: energy, data
    """
    # setting up random seeds
    rng = np.random.default_rng()

    if not image:
        t = rng.uniform(0, 10. * traj_samples, size=(data_size, traj_samples))
        k2 = rng.uniform(size=(data_size, 1)) if k2 is None else k2 * np.ones((data_size, 1))  # energies (conserved)

        sn, cn, dn, _ = ellipj(t, k2)
        q = 2 * np.arcsin(np.sqrt(k2) * sn) # angle
        p = 2 * np.sqrt(k2) * cn * dn / np.sqrt(1 - k2 * sn ** 2) # anglular momentum
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

    elif image and not blur:
        if t_window != [-1, -1] and t_range != -1:
            raise UserWarning("Cannot use both time windows & ranges at the same time")
        elif t_window != [-1, -1]:
            t = rng.uniform(t_window[0], t_window[1], size=(data_size, traj_samples))
            t = np.stack((t, t + diff_time), axis=-1) # time steps
        elif t_range != -1:
            t_base = rng.uniform(0, 10. * traj_samples, size=(data_size, 1))
            t_rng = rng.uniform(0, t_range, size=(data_size, traj_samples))
            t = t_base + t_rng
            t = np.stack((t, t + diff_time), axis=-1) # time steps
        else:
            t = rng.uniform(0, 10. * traj_samples, size=(data_size, traj_samples))
            t = np.stack((t, t + diff_time), axis=-1) # time steps

        if gaps == [-1,-1]:
            if not uniform:
                k2 = rng.uniform(mink, maxk, size=(data_size, 1, 1)) if k2 is None else k2 * np.ones((data_size, 1, 1))  # energies (conserved)
            else:
                if k2 == None:
                    k2 = np.linspace(mink, maxk, num=data_size)
                    k2 = np.reshape(k2, (data_size, 1, 1))
                else:
                    k2 = k2 * np.ones((data_size, 1, 1))
        else:
            if k2 != None:
                if not uniform:
                    k2 = rng.uniform(0, 1 - gaps[1], size=(data_size, 1, 1))
                    prefix = np.floor(k2 / (1 - gaps[1]) * (gaps[0] + 1)) * ((1 - gaps[1]) / (gaps[0] + 1) + gaps[1] / gaps[0])
                    frac = k2 - np.floor(k2 / (1 - gaps[1]) * (gaps[0] + 1)) * (1 - gaps[1]) / (gaps[0] + 1)
                    k2 = prefix + frac
                    k2 = k2 * (maxk - mink) + mink
                else:
                    k2 = np.linspace(0, 1 - gaps[1], num=data_size)
                    k2 = np.reshape(k2, (data_size, 1, 1))
                    prefix = np.floor(k2 / (1 - gaps[1]) * (gaps[0] + 1)) * ((1 - gaps[1]) / (gaps[0] + 1) + gaps[1] / gaps[0])
                    frac = k2 - np.floor(k2 / (1 - gaps[1]) * (gaps[0] + 1)) * (1 - gaps[1]) / (gaps[0] + 1)
                    k2 = prefix + frac
                    k2 = k2 * (maxk - mink) + mink
            else:
                k2 = k2 * np.ones((data_size, 1, 1))

        sn, cn, dn, _ = ellipj(t, k2)
        q = 2 * np.arcsin(np.sqrt(k2) * sn)

        if verbose:
            print("[Dataset] Numerical generation complete")

        if shuffle:
            for x in q:
                rng.shuffle(x, axis=0) # TODO: check if the shapes work out

        if noise > 0:
            q += noise * rng.standard_normal(size=q.shape)

        # Image generation begins here
        if crop != 1.0:
            if crop_c == [-1, -1]:
                crop_c = [1 - crop / 2, 1 - crop / 2]
            big_img = np.floor(img_size / crop + 4).astype('int32')
            left = np.floor(crop_c[0] * big_img - img_size / 2)
            top = np.floor(crop_c[1] * big_img - img_size / 2)
        else:
            big_img = img_size

        center_x = big_img // 2
        center_y = big_img // 2
        str_len = big_img - 4 - big_img // 2 - bob_size
        bob_area = (2 * bob_size + 1)**2

        pxls = np.ones((data_size, traj_samples, img_size + 2, img_size + 2, 3))
        if verbose:
            print("[Dataset] Blank images created")

        x = center_x + np.round(np.cos(q) * str_len)
        y = center_y + np.round(np.sin(q) * str_len)

        idx = np.indices((data_size, traj_samples))
        idx = np.expand_dims(idx, [0, 1, 5])

        bob_idx = np.indices((2 * bob_size + 1, 2 * bob_size + 1)) - bob_size
        bob_idx = np.swapaxes(bob_idx, 0, 2)
        bob_idx = np.expand_dims(bob_idx, [3, 4, 5])

        pos = np.expand_dims(np.stack((x, y), axis=0), [0, 1])
        pos = pos + bob_idx
        pos = np.reshape(pos, (bob_area, 2, data_size, traj_samples, 2))
        pos = np.expand_dims(pos, 0)

        c = np.expand_dims(np.array([[1, 1], [0, 2]]), [1, 2, 3, 4])

        idx, pos, c = np.broadcast_arrays(idx, pos, c)
        c = np.expand_dims(c[:, :, 0, :, :, :], 2)
        idx_final = np.concatenate((idx, pos, c), axis=2)

        if noise > 0:
            translation_noise = rng.uniform(0, 1, size=(2, data_size, traj_samples))
            translation_noise = np.minimum(np.ones(translation_noise.shape),
                                    np.floor(np.abs(translation_noise) / (1 - 2 * noise))) * translation_noise / np.abs(translation_noise)
            translation_noise = np.concatenate((np.zeros(translation_noise.shape), translation_noise, np.zeros(translation_noise.shape)), axis=0)
            translation_noise = translation_noise[:5]
            translation_noise = np.expand_dims(translation_noise, [0, 1, 5])
            idx_final = idx_final + translation_noise

        idx_final = np.swapaxes(idx_final, 0, 2)
        idx_final = np.reshape(idx_final, (5, 4 * data_size * traj_samples * bob_area))
        idx_final = idx_final.astype('int32')

        if verbose:
            print("[Dataset] Color indices computed")

        if crop == 1.0:
            pxls[idx_final[0], idx_final[1], idx_final[2] + 1, idx_final[3] + 1, idx_final[4]] = 0
        else:
            #bigpxls = np.ones((data_size, traj_samples, big_img + 2, big_img + 2, 3))
            #bigpxls[idx_final[0], idx_final[1], idx_final[2] + 1, idx_final[3] + 1, idx_final[4]] = 0
            idx_final[2] = idx_final[2] - left.astype('int32') + 1
            idx_final[3] = idx_final[3] - top.astype('int32') + 1
            idx_final[2] = np.maximum(idx_final[2], np.array(0))
            idx_final[3] = np.maximum(idx_final[3], np.array(0))
            idx_final[2] = np.minimum(idx_final[2], np.array(img_size + 1))
            idx_final[3] = np.minimum(idx_final[3], np.array(img_size + 1))

            pxls[idx_final[0], idx_final[1], idx_final[2], idx_final[3], idx_final[4]] = 0
        pxls = pxls[:, :, 1:img_size + 1, 1:img_size + 1, :]

        pxls = pxls + noise / 4 * rng.standard_normal(size=pxls.shape)
        tint_noise = rng.uniform(- noise / 8, noise / 8, size=(data_size, traj_samples, 1, 1, 3))
        pxls = pxls + tint_noise
        pxls = np.minimum(np.ones(pxls.shape), np.maximum(np.zeros(pxls.shape), pxls))

        if verbose:
            print("[Dataset] Images computed")

        """pxls = pxls * 255
        pxls = pxls.astype(np.uint8)
        #bigpxls = bigpxls * 255
        #bigpxls = bigpxls.astype(np.uint8)
        for j in range(0, traj_samples):
            for i in range(0, data_size):
                img = Image.fromarray(pxls[i,j,:,:,:], 'RGB')
                img.show()
                #fig, axs = plt.subplots(2)
                #axs[0].imshow(pxls[i,j,:,:,:])
                #axs[1].imshow(bigpxls[i,j,:,:,:])
                #plt.show()
                input("continue...")
                #plt.clf()"""

        pxls = np.swapaxes(pxls, 4, 2)
        return np.broadcast(k2, q)[0], pxls, q


class PendulumNumericalDataset(torch.utils.data.Dataset):
    def __init__(self, size=10240, trajectory_length=100, noise=0.00):
        self.size = size
        self.k2, self.data = pendulum_train_gen(size, noise=noise, image=False)
        self.trajectory_length = trajectory_length

    def __getitem__(self, idx):
        i = random.randint(0, self.trajectory_length - 1)
        j = random.randint(0, self.trajectory_length - 1)
        if torch.cuda.is_available():
            return [
                torch.cuda.FloatTensor(self.data[idx][i]),
                torch.cuda.FloatTensor(self.data[idx][j]),
                self.k2[idx]
            ]  # [first_view, second_view, energy]
        else:
            return [
                torch.FloatTensor(self.data[idx][i]),
                torch.FloatTensor(self.data[idx][j]),
                self.k2[idx]
            ]

    def __len__(self):
        return self.size

class PendulumImageDataset(torch.utils.data.Dataset):
    def __init__(self, size=5120, trajectory_length=20, noise=0.00,
                    img_size=32, diff_time=0.5, gaps=-1, crop=1.0, crop_c=[.75,.5],
                    t_window=[-1,-1], t_range=-1, mink=0, maxk=1, full_out=False):
        self.k2, self.data, self.q = pendulum_train_gen(size, noise=noise, traj_samples=trajectory_length,
                                                img_size=img_size, diff_time=diff_time, gaps=gaps, crop=crop, crop_c=crop_c,
                                                t_window=t_window, t_range=t_range, mink=mink, maxk=maxk)
        self.trajectory_length = trajectory_length
        self.full = full_out

    def __getitem__(self, idx):
        if not self.full:
            i = random.randint(0, self.trajectory_length - 1)
            j = random.randint(0, self.trajectory_length - 1)
            if torch.cuda.is_available():
                return [
                    torch.cuda.FloatTensor(self.data[idx][i]),
                    torch.cuda.FloatTensor(self.data[idx][j]),
                    self.k2[idx],
                    self.q[idx][i],
                    self.q[idx][j]
                ]  # [first_view, second_view, energy]
            else:
                return [
                    torch.FloatTensor(self.data[idx][i]),
                    torch.FloatTensor(self.data[idx][j]),
                    self.k2[idx],
                    self.q[idx][i],
                    self.q[idx][j]
                ]
        else:
            if torch.cuda.is_available():
                return [
                    torch.cuda.FloatTensor(self.data[idx]),
                    self.k2[idx],
                    self.q[idx]
                ]  # [first_view, second_view, energy]
            else:
                return [
                    torch.FloatTensor(self.data[idx]),
                    self.k2[idx],
                    self.q[idx]
                ]

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

class EncodingMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


class Branch(nn.Module):
    #def __init__(self, proj_dim, proj_hidden, deeper, affine, encoder=None, resnet=True):
    def __init__(self, repr_dim, proj_hidden=16, proj_out=-1, deeper=True, affine=True, encoder=None, resnet=True):
        super().__init__()

        if encoder:
            self.encoder = encoder
        elif resnet:
            # self.encoder = torchvision.models.resnet18(zero_init_residual=True)
            self.encoder = torchvision.models.resnet18(pretrained=False)
            # self.encoder.fc = nn.Identity()  # replace the classification head with identity
            self.encoder.fc = nn.Sequential(
                nn.Linear(512, repr_dim)
            )
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
                nn.Linear(64, repr_dim)
            )  # simple encoder to start with

        if proj_out == -1:
            self.projector = nn.Identity()  # TODO: to keep it simple, for now we will not use projector
        else:
            # self.projector = ProjectionMLP(32, 64, 32, affine=affine, deeper=deeper)
            self.projector = ProjectionMLP(repr_dim, proj_hidden, proj_out, affine=affine, deeper=deeper)

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

def supervised_loop(args, encoder=None):
    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=4)
    train_loader = torch.utils.data.DataLoader(
        dataset=PendulumImageDataset(size=args.data_size, gaps=args.gaps),
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )
    if args.validation:
        test_loader = torch.utils.data.DataLoader(
            dataset=PendulumImageDataset(size=512, gaps=args.gaps),
            shuffle=False,
            batch_size=512,
            **dataloader_kwargs
        )
    if verbose:
        print("[Supervised] Completed data loading")

    # optimization
    main_branch = Branch(args.repr_dim, deeper=args.deeper, affine=args.affine, encoder=encoder)
    if torch.cuda.is_available():
        main_branch.cuda()

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

    # macros
    b = main_branch.encoder

    if verbose:
        print("[Supervised] Model generation complete, training begins")

    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'training_loss.log'), 'w')
    torch.save(dict(epoch=0, state_dict=b.state_dict()), os.path.join(args.path_dir, '0.pth'))

    loss = torch.nn.MSELoss()

    for e in range(1, args.epochs + 1):
        b.train()

        # epoch
        for it, (x1, x2, energy, q1, q2) in enumerate(train_loader):
            # zero grad
            b.zero_grad()

            # forward pass
            out = b(x1)
            out_loss = loss(out, energy.float())

            # optimization step
            out_loss.backward()
            torch.nn.utils.clip_grad_norm_(b.parameters(), 3)
            optimizer.step()

            lr_scheduler.step()
            if args.dim_pred:
                pred_optimizer.step()

        if e % args.progress_every == 0 or e % args.save_every == 0:
            if args.validation:
                b.eval()
                val_loss = -1
                for it, (x1, x2, energy, q1, q2) in enumerate(test_loader):
                    val_loss = loss(b(x1), energy.float())
                    break
                line_to_print = f'epoch: {e} | loss: {out_loss.item()} | val loss: {val_loss.item()} | time_elapsed: {time.time() - start:.3f}'
            else:
                line_to_print = f'epoch: {e} | loss: {out_loss.item()} | time_elapsed: {time.time() - start:.3f}'

            print(line_to_print)

            if e % args.save_every == 0:
                torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, f'{e}.pth'))
                file_to_update.write(line_to_print + '\n')
                file_to_update.flush()

    file_to_update.close()
    if verbose:
        print("[Supervised] Training complete")

    return b


def training_loop(args, encoder=None):
    # dataset
    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(
        dataset=PendulumImageDataset(size=args.data_size, trajectory_length=args.traj_len, noise=args.noise,
                                        img_size=args.img_size, diff_time=args.diff_time, gaps=args.gaps, crop=args.crop, crop_c = args.crop_c,
                                        t_window=args.t_window, t_range=args.t_range, mink=args.mink, maxk=args.maxk),
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )
    if args.validation:
        test_loader = torch.utils.data.DataLoader(
            dataset=PendulumImageDataset(size=512, gaps=args.gaps),
            shuffle=False,
            batch_size=512,
            **dataloader_kwargs
        )
    if verbose:
        print("[Self-supervised] Completed data loading")

    # model
    main_branch = Branch(args.repr_dim, deeper=args.deeper, affine=args.affine, encoder=encoder)
    if torch.cuda.is_available():
        main_branch.cuda()

    if args.method == "simsiam":
        h = PredictionMLP(args.repr_dim, args.dim_pred, args.repr_dim)
        if torch.cuda.is_available():
            h.cuda()

    # optimization
    optimizer = torch.optim.SGD(
        main_branch.parameters(),
        momentum=0.9,
        lr=args.lr,
        weight_decay=args.wd
    ) # extremely sensitive to learning rate
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
    if args.method == "simsiam":
        pred_optimizer = torch.optim.SGD(
            h.parameters(),
            momentum=0.9,
            lr=args.pred_lr,
            weight_decay=args.wd
        )

    # macros
    b = main_branch.encoder
    proj = main_branch.projector

    # helpers
    def get_z(x):
        return proj(b(x))

    def apply_loss(z1, z2, distance):
        #if args.loss == 'square':
        #    loss = (z1 - z2).pow(2).sum()
        if args.method == 'infonce':
            loss = 0.5 * info_nce(z1, z2, temperature=args.temp, distance=distance) + 0.5 * info_nce(z2, z1, temperature=args.temp, distance=distance)
        elif args.method == 'simsiam':
            p1 = h(z1)
            p2 = h(z2)
            loss = simsiam_loss(p1, z2, distance=distance) / 2 + simsiam_loss(p2, z1, distance=distance) / 2
        return loss

    if verbose:
        print("[Self-supervised] Model generation complete, training begins")

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'training_loss.log'), 'w')
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))

    # training
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()
        if args.method == "simsiam":
            h.train()

        # epoch
        losses = []
        zmin = torch.tensor(10000)
        zmax = torch.tensor(-10000)
        for it, (x1, x2, energy, q1, q2) in enumerate(train_loader):
            # zero grad
            main_branch.zero_grad()
            if args.method == "simsiam":
                h.zero_grad()

            # forward pass
            z1 = get_z(x1)
            z2 = get_z(x2)

            zmin = torch.min(torch.min(z1, zmin))
            zmax = torch.max(torch.max(z1, zmax))
            zmin = torch.min(torch.min(z2, zmin))
            zmax = torch.max(torch.max(z2, zmax))

            if args.cosine:
                loss = apply_loss(z1, z2, distance="cosine")
            else:
                loss = apply_loss(z1, z2, distance="euclidean")
            losses.append(loss.item())

            # optimization step
            loss.backward()
            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(main_branch.parameters(), args.clip)
                if args.method == "simsiam":
                    torch.nn.utils.clip_grad_norm_(h.parameters(), args.clip * 2)
            optimizer.step()
            lr_scheduler.step()
            if args.method == "simsiam":
                pred_optimizer.step()

        if e % args.progress_every == 0 or e % args.save_every == 0:
            if args.validation:
                main_branch.eval()
                val_loss = -1
                for it, (x1, x2, energy, q1, q2) in enumerate(test_loader):
                    val_loss = loss(get_z(x1), get_z(x2)).item()
                    break
                line_to_print = f'epoch: {e} | loss: {loss.item()} | val loss: {val_loss.item()} | time_elapsed: {time.time() - start:.3f}'
            else:
                losses = torch.tensor(losses)
                losses = torch.std(losses)
                zrange = zmax - zmin
                line_to_print = f'epoch: {e} | loss: {loss.item()} | std: {losses.item()} | range: {zrange} | time_elapsed: {time.time() - start:.3f}'

            print(line_to_print)

            if e % args.save_every == 0:
                torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, f'{e}.pth'))
                file_to_update.write(line_to_print + '\n')
                file_to_update.flush()

                data_args = {
                                "img_size": args.img_size,
                                "diff_time": args.diff_time,
                                "gaps": args.gaps,
                                "crop": args.crop,
                                "crop_c": args.crop_c,
                                "t_window": args.t_window,
                                "t_range": args.t_range,
                                "mink": args.mink,
                                "maxk": args.maxk,
                                "repr_dim": args.repr_dim
                            }
                with open(os.path.join(args.path_dir, f'{e}.args'), 'wb') as fp:
                    pickle.dump(itemlist, fp)
                print("[saved]")

    file_to_update.close()

    if verbose:
        print("[Self-supervised] Training complete")

    return main_branch.encoder


def testing_loop(args, encoder=None):
    load_files = args.load_file
    if args.load_every != -1:
        load_files = []
        idx = 0
        while 1 + 1 == 2 and idx * args.load_every <= args.load_max:
            file_to_add = os.path.join(args.path_dir, str(idx * args.load_every) + ".pth")
            if os.path.isfile(file_to_add):
                load_files.append(file_to_add)
                idx = idx + 1
            else:
                break
    elif load_files == "recent":
        load_files = [most_recent_file(args.path_dir, ext=".pth")]
    else:
        load_files = os.path.join(args.path_dir, load_files)
        load_files = [load_files]

    b = []
    data_args = {}

    for load_file in load_files:
        with open(os.path.join(args.path_dir, load_file[:-3] + ".args"), 'rb') as fp:
            new_data_args = pickle.load(fp)
            if data_args == {}:
                data_args = new_data_args
            else:
                assert(data_args == new_data_args)

        branch = Branch(data_args['repr_dim'], deeper=args.deeper, affine=args.affine, encoder=encoder)
        branch.load_state_dict(torch.load(load_file)["state_dict"])

        if torch.cuda.is_available():
            branch.cuda()

        branch.eval()
        b.append(branch.encoder)

    if verbose:
        print("[testing] Completed model loading")

    #test_k2, test_data, test_q = pendulum_train_gen(data_size=args.data_size,
        #traj_samples=(args.traj_len if args.sparse_testing else 1),
        #noise=args.noise, full_out=args.sparse_testing,
        #uniform=(not args.sparse_testing),img_size=args.img_size,
        #diff_time=args.diff_time, gaps=args.gaps,crop=args.crop,
        #crop_c = args.crop_c, t_window=args.t_window,t_range=args.t_range,
        #mink=args.mink, maxk=args.max)
    test_k2, test_data, test_q = pendulum_train_gen(data_size=args.data_size,
        traj_samples=(args.traj_len if args.sparse_testing else 1),
        noise=args.noise, uniform=(not args.sparse_testing), img_size=data_args['img_size'],
        diff_time=data_args['diff_time'], crop=data_args['crop'], crop_c=data_args['crop_c'])

    if verbose:
        print("[testing] Completed data loading")

    coded = np.zeros((len(b), args.data_size, args.traj_len if args.sparse_testing else 1, args.repr_dim))

    for i in range(len(b)):
        for j in range(0, args.data_size):
            coded[i, j, :, :] = b[i](test_data[j, :, :, :, :]).cpu().detach().numpy()
    energies = test_energy.cpu().detach().numpy()
    qs = test_q[:, 0].cpu().detach().numpy()

    os.makedirs(os.path.join(args.path_dir, "testing"), exist_ok=True)
    for idx, load_file in enumerate(load_files):
        save_file = "-" + load_file.rpartition("/")[2].rpartition(".")[0]
        np.save(os.path.join(args.path_dir, "testing/coded" + save_file + ".npy"), coded[idx])
    np.save(os.path.join(args.path_dir, "testing/energies.npy"), energies)
    np.save(os.path.join(args.path_dir, "testing/qs.npy"), qs)

    if verbose:
        print("[testing] Testing data saved... beginning analysis")

    # Analysis: NN modeling, density, spearman, line optimality

    prt_image = data_args['crop'] != 1.0
    prt_time = data_args['t_window'] != [-1, -1] or data_args['t_range'] != -1
    prt_energy = data_args['gaps'] != [-1, -1] or data_args['mink'] != 0 or data_args['maxk'] != 0

    datashape = list(test_data.shape)
    datashape[0] = datashape[0] * datashape[1]
    datashape.pop(1)

    slim_data = np.reshape(test_data, tuple(datashape))

    k2qshape = list(test_k2.shape)
    k2qshape[0] = k2qshape[0] * k2qshape[1]
    k2qshape.pop(1)

    slim_k2 = np.reshape(test_k2, tuple(k2qshape))
    slim_q = np.reshape(test_q, tuple(k2qshape))

    cshape = list(coded.shape)
    cshape[1] = cshape[1] * cshape[2]
    cshape.pop(2)

    slim_coded = np.reshape(coded, cshape)

    ## NN modeling
    if prt_image and not prt_time:
        red_visible = slim_data[:, :, :, 0]
        red_visible = np.reshape(red_visible, (np.size(red_visible, axis=0), -1))
        red_visible = np.sum(1 - np.floor(red_visible * 2), axis=2)
        red_visible = np.where(red_visible < 9, 0, 1)

        blue_visible = slim_data[:, :, :, 2]
        blue_visible = np.reshape(blue_visible, (np.size(blue_visible, axis=0), -1))
        blue_visible = np.sum(1 - np.floor(blue_visible * 2), axis=2)
        blue_visible = np.where(blue_visible < 9, 0, 1)

        visible = red_visible * blue_visible

        vis_data = slim_data[visible == 1, ...]
        vis_k2 = slim_k2[visible == 1, ...]
        vis_q = slim_q[visible == 1, ...]
        vis_coded = slim_coded[:, visible == 1, ...]
    else:
        vis_data = slim_data
        vis_k2 = slim_k2
        vis_q = slim_q
        vis_coded = slim_coded

    if prt_time:
        mimic_input = np.stack((vis_k2.flat, vis_q.flat), axis=0)
    else:
        mimic_input = vis_k2.flat

    mimics = []
    for i in range(len(b)):
        if prt_time:
            mimic = EncodingMLP(2)
        else:
            mimic = EncodingMLP(1)

        if torch.cuda.is_available():
            mimic.cuda()

        optimizer = torch.optim.Adam(
            mimic.parameters(),
            weight_decay=args.wd
        )
        loss = torch.nn.MSELoss()
        for e in range(1, args.epochs + 1):
            mimic.train()
            mimic.zero_grad()

            z = mimic(mimic_input)
            out_loss = loss(z, b[i](vis_data))
            print(out_loss.item())

            out_loss.backward()
            optimizer.step()
        torch.save(dict(state_dict=mimic.state_dict()), os.path.join(args.path_dir, f'{e}_map.pth'))
        mimics.append(mimic)

    ## Density computation
    densities = []

    for i in range(len(b)):
        den_size = data_size / 16
        if prt_time:
            uniform_q = torch.linspace(torch.min(test_q), torch.max(test_q), den_size)
            uniform_k2 = torch.linspace(torch.min(test_k2), torch.max(test_k2), den_size)

            uniform_q = torch.reshape(uniform_q, (den_size, 1))
            uniform_k2 = torch.reshape(uniform_k2, (1, den_size))
            uniform_q = uniform_q.repeat(1, den_size)
            uniform_k2 = uniform_k2.repeat(den_size, 1)

            den_in = torch.stack((uniform_q, uniform_k2), dim=2)
            den_in = torch.reshape(den_in, (den_size**2, 2))
        else:
            uniform_k2 = torch.linspace(torch.min(test_k2), torch.max(test_k2), den_size)
            den_in = torch.reshape(uniform_k2, (den_size, 1))

        mimic.eval()
        x = mimic(den_in)

        if prt_time:
            x = torch.reshape(x, (den_size, den_size))

        if prt_time:
            den = (1 / torch.abs(x[2:,1:-1] - x[1:-1,1:-1]) +
                1 / torch.abs(x[:-2,1:-1] - x[1:-1,1:-1]) +
                1 / torch.abs(x[1:-1,2:] - x[1:-1,1:-1]) +
                1 / torch.abs(x[1:-1,:-2] - x[1:-1,1:-1]))
            den = 0.25 * den
        else:
            den = 1 / torch.abs(x[2:] - x[1:-1]) + 1 / torch.abs(x[:-2] - x[1:-1])
            den = 0.5 * den

        torch.save(den, os.path.join(args.path_dir, f'{e}_den.pth'))
        densities.append(den)
    densities = torch.tensor(densities).cuda()

    # Spearman
    print("Raw spearman.")
    for i in range(0, len(b)):
        print(load_files[i] + ": " + str(stats.spearmanr(coded[i], energies).correlation))

    if prt_image:
        print("Visible spearman.")
        for i in range(0, len(b)):
            print(load_files[i] + ": " + str(stats.spearmanr(vis_coded[i], vis_k2).correlation))

    if prt_time or prt_energy:
        print("Same testing set spearman.")
        exact_k2, exact_data, exact_q = pendulum_train_gen(data_size=args.data_size,traj_samples=1,
            noise=args.noise,uniform=True,img_size=data_args['img_size'],
            diff_time=data_args['diff_time'], gaps=data_args['gaps'],
            crop=data_args['crop'],crop_c = data_args['crop_c'],
            t_window=data_args['t_window'],t_range=data_args['t_range'],
            mink=data_args['mink'], maxk=data_args['maxk'])
        for i in range(0, len(b)):
            b[i].eval()
            out = b[i](exact_data)
            print(load_files[i] + ": " + str(stats.spearmanr(out, exact_k2).correlation))

    if prt_energy:
        den_size = data_size / 16
        uniform_k2 = torch.linspace(torch.min(test_k2), torch.max(test_k2), den_size).tolist()
        uniform_k2 = uniform_k2[1:-1]

        num_gaps = data_args['gaps'][0]
        if num_gaps == -1:
            num_gaps = 0
        mink = data_args['mink']
        maxk = data_args['maxk']

        gap_width = data_args['gaps'][1] / num_gaps
        btwn_width = (1 - data_args['gaps'][1]) / (num_gaps + 1)
        gap_width = gap_width * (maxk - mink)
        btwn_width = btwn_width * (maxk - mink)

        borders = [mink]

        for i in range(0, num_gaps):
            borders.append(borders[-1] + btwn_width)
            borders.append(borders[-1] + gap_width)
        borders.append(borders[-1] + btwn_width)
        borders.append(maxk)

        combined = []

        print("Segmented spearman.")
        for i in range(0, len(borders)):
            store = []
            for it, k in reversed(list(enumerate(uniform_k2))):
                if k < borders[i]:
                    if prt_time:
                        to_append = torch.median(densities[:, it, :])
                    else:
                        to_append = densities[:, it]

                    store.append(combined)
                    uniform_k2.pop(it)
            store = torch.tensor(store).transpose(0, 1)
            store = torch.median(store)
            combined.append(store)

            s_energy, s_data, s_q = pendulum_train_gen(data_size=args.data_size, noise=args.noise,
                mink=0 if i == 0 else borders[i - 1], maxk=borders[i], uniform=True, img_size=data_args['img_size'],
                diff_time=data_args['diff_time'], crop=data_args['crop'], crop_c=data_args['crop_c'])
            for j in range(0, len(b)):
                b[j].eval()
                out = b[j](s_data)
                print(load_files[j] + ": " + str(stats.spearmanr(out, s_data).correlation))

        combined = torch.tensor(store)

        print("Density differences.")
        for i in range(0, len(b)):
            if verbose:
                print(combined[i].item())

            with open(os.path.join(args.path_dir, f'{e}_avgden'), 'wb') as fp:
                pickle.dump(combined[i], fp)


def main(args):
    global verbose

    if args.verbose:
        verbose = True

    args.gaps = args.gaps.split(",")
    assert(len(args.gaps) == 2)
    args.gaps[0] = float(args.gaps[0])
    args.gaps[1] = float(args.gaps[1])

    args.crop_c = args.crop_c.split(",")
    assert(len(args.crop_c) == 2)
    args.crop_c[0] = float(args.crop_c[0])
    args.crop_c[1] = float(args.crop_c[1])

    args.t_window = args.t_window.split(",")
    assert(len(args.t_window) == 2)
    args.t_window[0] = float(args.t_window[0])
    args.t_window[1] = float(args.t_window[1])

    set_deterministic(42)
    if args.mode == 'training':
        training_loop(args)
    elif args.mode == 'testing':
        testing_loop(args)
    elif args.mode == 'plotting':
        plotting_loop(args)
    elif args.mode == 'supervised':
        supervised_loop(args)
    else:
        raise

    if not args.silent:
        for i in range(0,15):
            subprocess.run("echo $\'\a\'", shell=True)
            time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='training', type=str,
                        choices=['plotting', 'training', 'testing', 'supervised'])
    parser.add_argument('--method', default='infonce', type=str,
                        choices=['infonce', 'simsiam'])

    # Data generation options
    parser.add_argument('--data_size', default=5120, type=int)
    parser.add_argument('--traj_len', default=20, type=int)
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--diff_time', default=0.5, type=float)

    parser.add_argument('--gaps', default="-1,-1", type=str)
    parser.add_argument('--crop', default=1.0, type=float)
    parser.add_argument('--crop_c', default="-1,-1", type=str)
    parser.add_argument('--t_window', default="-1,-1", type=str)
    parser.add_argument('--t_range', default=-1,type=float)
    parser.add_argument('--mink', default=0.0, type=float)
    parser.add_argument('--maxk', default=1.0, type=float)

    parser.add_argument('--noise', default=0., type=float)

    # File I/O
    parser.add_argument('--path_dir', default='../output/pendulum', type=str)

    parser.add_argument('--load_file', default='recent', type=str)
    parser.add_argument('--load_every', default='-1', type=int)
    parser.add_argument('--load_max', default=1000000, type=int)
    parser.add_argument('--sparse_testing', default=False, action='store_true')

    # Training reporting options
    parser.add_argument('--progress_every', default=5, type=int)
    parser.add_argument('--save_every', default=20, type=int)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--validation', default=False, action='store_true')
    parser.add_argument('--silent', default=False, action='store_true')

    # Optimizer options
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)

    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--pred_lr', default=0.02, type=float)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--cosine', default=False, action='store_true')

    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--clip', default=3.0, type=float)

    # NN size options
    parser.add_argument('--dim_pred', default=1, type=int)
    parser.add_argument('--repr_dim', default=2, type=int)
    parser.add_argument('--affine', action='store_false')
    parser.add_argument('--deeper', action='store_false')

    args = parser.parse_args()
    main(args)
