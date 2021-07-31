import os
import multiprocessing
import subprocess
import numpy as np
try:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception:
    import matplotlib
    matplotlib.use('ps')
    pass
import matplotlib.pyplot as plt
import json

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def run_cmd(cmds, gpu):
    for cmd in cmds:
        print("running " + cmd)
        subprocess.run(cmd  + " --silent --gpu=" + str(gpu), shell=True)

if __name__ == "__main__":
    #names = ["vnladim_1", "dataset_5120", "inoise_inf", "nnoise_inf", "nnoise_1"]
    names = ["nnoise"]
    exps = [f.name for f in os.scandir("../../output/pendulum") if f.is_dir()]
    myexps = []
    short_cmd = []
    for k in exps:
        goahead = False
        for name in names:
            if name in k:
                goahead = True
        if not goahead:
            continue

        if k.endswith("_sup"):
            continue
        myexps.append(k)
        if not os.path.exists("../../output/pendulum/" + k + "/testing/coded-100.npy"):
            if "noise_inf" in k or "vnladim" in k or "dataset" in k:
                short_cmd.append("cd ..; python pendulum.py --verbose --mode=testing --path_dir=" + k)
            else:
                num = k[k.rfind("_")+1:]
                num = int(num)
                num = 1.0/num
                if "inoise" in k:
                    short_cmd.append("cd ..; python pendulum.py --verbose --mode=testing --path_dir=" + k + " --gnoise=" + str(num))
                elif "nnoise" in k:
                    short_cmd.append("cd ..; python pendulum.py --verbose --mode=testing --path_dir=" + k + " --nnoise=" + str(num))

    """for cmds in [long_cmd]:
        jobs = []
        for i in range(3, 8):
            print(str(i))
            this_cmds = []
            j = i % 5
            while j < len(cmds):
                this_cmds.append(cmds[j])
                j += 5
            p = multiprocessing.Process(target=run_cmd, args=(this_cmds,i))
            jobs.append(p)
            print("starting " + str(i))
            p.start()
        for job in jobs:
            job.join()"""

    for cmds in [short_cmd]:
        run_cmd(cmds, 3)

    with open("../data/master_experiments.json") as f:
        data = json.load(f)
    
    borders = np.array([1/7,2/7,3/7,4/7,5/7,6/7,1]).ravel()
    stdev = []
    dens = []
    for i in range(0,7):
        stdev.append([])
        dens.append([])
    for k in myexps:
        num = k[k.rfind("_")+1:]
        if num != "inf":
            num = int(num)
            if num < 20:
                continue

        coded = np.load("../../output/pendulum/" +k+ "/testing/coded-100.npy")
        coded = coded[:,0,0].ravel()
        energies = np.load("../../output/pendulum/" +k+ "/testing/energies.npy")
        energies = energies[:,0].ravel()
        idx = np.searchsorted(borders, energies) + 1
        odd = np.array(range(len(coded.tolist()))) % 2
        good = idx * odd - 1

        test_coded = coded[good == -1]
        test_energy = energies[good == -1]

        mlp = RandomForestRegressor()
        mlp.fit(test_coded.reshape(-1,1),test_energy.reshape(-1,))

        thisdens = []

        for key in data.keys():
            if data[key]["experiment_name"] == k:# and data[key]["new_params"]["nnoise"] == 0 and data[key]["new_params"]["gnoise"] == 0:
                thisdens = data[key]["k_density"][1:-1]

        for i in range(0,7):
            p = mlp.predict(coded[good ==i].reshape(-1,1))
            p = p.ravel() - energies[good == i]
            p = np.sqrt(np.mean(np.square(p)))
            stdev[i].append(p)
            dens[i].append(thisdens[i])
            if thisdens[i] > 1:
                print(k)
                print(i)

    linear = LinearRegression()
    stdev_full = []
    dens_full = []
    for i in range(0,7):
        stdev_full = stdev_full + stdev[i]
        dens_full = dens_full + dens[i]
    stdev_full = np.array(stdev_full).ravel().reshape(-1,1)
    dens_full = np.array(dens_full).ravel().reshape(-1,1)
    linear.fit(stdev_full, dens_full)
    m = linear.coef_.ravel()
    b = linear.intercept_.ravel()
    print(linear.score(stdev_full, dens_full))
    xx = np.linspace(np.min(stdev_full), np.max(stdev_full), 500)
    yy = xx * m + b

    for i in range(0,7):
        plt.scatter(stdev[i], dens[i], s=0.5, label=str(i))
    plt.plot(xx, yy, '--', color='k')
    plt.show()
