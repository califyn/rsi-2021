def pendulum_train_gen(batch_size, traj_samples=10, noise=0.,
        shuffle=True, check_energy=False, k2=None, gaps=False,
        image=True, img_size=32, diff_time=0.5, bob_size=1):
    """
    pendulum dataset generation
    :param batch_size: number of pendulums
    :param traj_samples: number of samples per pendulum
    :param noise: Gaussian noise
    :param shuffle: if shuffle data
    :param check_energy: check the energy (numerical mode only)
    :param k2: specify energy
    :param gaps: generate gaps in data to test interpolation (graphical mode only)
    :param image: use image/graphical mode
    :param img_size: size of (square) image (graphical mode only)
    :param diff_time: time difference between two images (graphical mode only)
    :param bob_size: bob = square of length bob_size * 2 + 1
    :return: energy, data
    """
    # setting up random seeds
    rng = np.random.default_rng()

    if not image:
        t = rng.uniform(0, 10. * traj_samples, size=(batch_size, traj_samples))
        k2 = rng.uniform(size=(batch_size, 1)) if k2 is None else k2 * np.ones((batch_size, 1))  # energies (conserved)

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

    elif image:
        t = rng.uniform(0, 10. * traj_samples, size=(batch_size, traj_samples))
        t = np.stack((t, t + diff_time), axis=-1) # time steps

        k2 = rng.uniform(size=(batch_size, 1, 1)) if k2 is None else k2 * np.ones((batch_size, 1, 1))  # energies (conserved)
        if gaps:
            for i in range(0, batch_size):
                if np.floor(k2[i, 0, 0] * 3) % 2 == 1:
                    k2[i, 0, 0] = k2[i, 0, 0] - 1/3

        center_x = img_size // 2
        center_y = img_size // 2
        str_len = img_size - 4 - img_size // 2 - bob_size
        bob_area = (2 * bob_size + 1)**2

        sn, cn, dn, _ = ellipj(t, k2)
        q = 2 * np.arcsin(np.sqrt(k2) * sn)

        if shuffle:
            for x in q:
                rng.shuffle(x, axis=0) # TODO: check if the shapes work out

        if noise > 0:
            q += noise * rng.standard_normal(size=q.shape)

        # Image generation begins here
        pxls = np.ones((batch_size, traj_samples, img_size, img_size, 3))

        x = center_x + np.round(np.cos(q) * str_len)
        y = center_y + np.round(np.sin(q) * str_len)

        idx = np.indices((batch_size, traj_samples))
        idx = np.expand_dims(idx, [0, 1, 5])

        bob_idx = np.indices((2 * bob_size + 1, 2 * bob_size + 1)) - bob_size
        bob_idx = np.swapaxes(bob_idx, 0, 2)
        bob_idx = np.expand_dims(bob_idx, [3, 4, 5])

        pos = np.expand_dims(np.stack((x, y), axis=0), [0, 1])
        pos = pos + bob_idx
        pos = np.reshape(pos, (bob_area, 2, batch_size, traj_samples, 2))
        pos = np.expand_dims(pos, 0)

        c = np.expand_dims(np.array([[1, 1], [0, 2]]), [1, 2, 3, 4])

        idx, pos, c = np.broadcast_arrays(idx, pos, c)
        c = np.expand_dims(c[:, :, 0, :, :, :], 2)
        idx_final = np.concatenate((idx, pos, c), axis=2)

        idx_final = np.swapaxes(idx_final, 0, 2)
        idx_final = np.reshape(idx_final, (5, 4 * batch_size * traj_samples * bob_area))
        idx_final = idx_final.astype('int32')

        pxls[idx_final[0], idx_final[1], idx_final[2], idx_final[3], idx_final[4]] = 0

        """pxls = pxls * 255
        pxls = pxls.astype(np.uint8)
        for i in range(0, batch_size):
            for j in range(0, traj_samples):
                img = Image.fromarray(pxls[i,j,:,:,:], 'RGB')
                img.show()
                input("continue...")""" # use this to display your images for debugging

        pxls = np.swapaxes(pxls, 4, 2)
        return np.reshape(k2, (batch_size, 1)), pxls
