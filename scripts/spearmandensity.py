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
import seaborn as sns

sns.set_theme(context="paper", style="ticks")
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


names = ["vnladim_1", "dataset_5120", "inoise_inf", "nnoise_inf"]

with open("../data/master_experiments.json") as f:
    data = json.load(f)

splitsize=7
borders = (np.array(range(splitsize)).ravel() + 1) * 1/splitsize
sp = []
dens = []
for i in range(0,splitsize):
    sp.append([])
    dens.append([])

for d in data.keys():
    if data[d]["experiment_name"].endswith("_sup"):
        continue
    goahead = False
    for name in names:
        if name in data[d]["experiment_name"]:
            goahead = True
    if not goahead:
        continue

    for i in range(0,splitsize):
        dens[i].append(data[d]["k_density"][i + 1])
        sp[i].append(abs(data[d]["k_spearman"][i + 1]))
print(sp[:50])
#linear = LinearRegression()
#stdev_full = []
#dens_full = []
#for i in range(0,splitsize):
#    stdev_full = stdev_full + stdev[i]
#    dens_full = dens_full + dens[i]
#stdev_full = np.array(stdev_full).ravel().reshape(-1,1)
#dens_full = np.array(dens_full).ravel().reshape(-1,1)
#linear.fit(stdev_full, dens_full)
#m = linear.coef_.ravel()
#b = linear.intercept_.ravel()
#print(linear.score(stdev_full, dens_full))
#xx = np.linspace(np.min(stdev_full), np.max(stdev_full), 500)
#yy = xx * m + b

fig, ax = plt.subplots()
palette = sns.color_palette("flare_r", n_colors=splitsize)
for i in range(0,splitsize):
    plt.scatter(sp[i], dens[i], s=3, c=palette[i])
plt.plot([1,0],[0,1],transform=ax.transAxes, ls="--", c='k')
#plt.plot(xx, yy, '--', color='k')
plt.colorbar()
sns.despine()
plt.xlabel("Local Spearman")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
