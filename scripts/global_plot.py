import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception:
    import matplotlib
    matplotlib.use('ps')
    pass
import matplotlib.pyplot as plt

expname = "nnoise"

with open("../data/master_experiments.json", "r") as f:
    data = json.load(f)

keys = list(data.keys())
for key in keys:
    if expname not in data[key]["experiment_name"] or int(data[key]["experiment_name"][0]) > 4:
        data.pop(key)

noises = [0,0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.125,0.2,0.25]
orig_noises = noises
noises = np.array(noises)
noises[0] =1
noises = np.reciprocal(noises).astype("int32").tolist()
noises[0] = "inf"
for i in range(0, len(noises)):
    noises[i] = str(noises[i])

res = np.zeros((11,5))
resp = np.zeros((11,5,7))
sres= np.zeros((11,5))
sresp = np.zeros((11,5,7))

for key in data.keys():
    exp = data[key]["experiment_name"]
    print(exp)
    trial = int(exp[0])
    sup = False
    if exp.endswith("_sup"):
        sup = True
        exp = exp[:-4]
    num = exp[exp.rfind("_") + 1:]

    if sup:
        sres[noises.index(num), trial] = data[key]["full_speaman"]
        sresp[noises.index(num), trial] = np.array(data[key]["spearman"])[1:-1]
    else:
        res[noises.index(num), trial] = data[key]["full_speaman"]
        resp[noises.index(num), trial] = np.array(data[key]["spearman"])[1:-1]

w = plt.get_cmap('plasma')
res = np.abs(res)
resp = np.abs(resp)
sres = np.abs(sres)
sresp = np.abs(sresp)

#res = np.log(1 - res)
#resp = np.log(1 - resp)
#sres = np.log(1 - sres)
#sresp = np.log(1 - sresp)

res = np.mean(res, axis=1)
resp = np.mean(resp, axis=1)
sres = np.mean(sres, axis=1)
sresp = np.mean(sresp, axis=1)
print(res)
print(sres)

def OneMinusLog(arr):
    return -1 * (np.log(1 - arr))

def OneMinusLogInv(arr):
    return 1 - np.exp(-1 * arr)


fig, ax1 = plt.subplots()

ax1.set_xscale('log')
ax1.set_yscale('function', functions=(OneMinusLog, OneMinusLogInv))
ax1.set_xlabel("Noise")
ax1.set_ylabel("Global Spearman", color=[1,0,0])
ax1.tick_params(axis='y', labelcolor=[1,0,0])

ax1.plot(orig_noises[1:], res[1:], lw=0.75, color=[1,0,0])
ax1.plot(orig_noises[1:], sres[1:], lw=0.75, color=[1,0,0], linestyle="--")

ax1.scatter(0.1,0.99383, color="white")
ax1.axhline(y=0.9938278137526757, lw=0.75, linestyle="-", c=[1,0.5,0.5])
ax1.axhline(y=0.993669710607535, lw=0.75, linestyle="--", c=[1,0.5,0.5])
ax2=ax1.twinx()

ax2.set_yscale('function', functions=(OneMinusLog, OneMinusLogInv))
ax2.set_ylabel("Local Spearman", color=[0,1,0])
ax2.tick_params(axis='y', labelcolor=[0,1,0])

ax2.plot(orig_noises[1:], np.mean(resp[1:],axis=1), lw=0.75, color=[0,1,0])
ax2.plot(orig_noises[1:], np.mean(sresp[1:],axis=1), lw=0.75, color=[0,1,0], linestyle="--")

ax2.axhline(y=0.8092634019325426, lw=0.75, linestyle="-", c=[0.5,1,0.5])
ax2.axhline(y=0.8043783294835989, lw=0.75, linestyle="--", c=[0.5,1,0.5])

fig.tight_layout()
plt.show()

plt.clf()

"""plt.xscale('linear')
plt.yscale('linear')

x=[1,2,3,4,5,6,7]

subrange = [1,4,7]
colors = {}
for it, i in enumerate(subrange):
    colors[i] = np.array(w((it)/(len(subrange) - 1)))
for it, i in enumerate(subrange):
    plt.plot(x, resp[i], lw=0.75, linestyle="-", c=colors[i], label=orig_noises[i])
    plt.plot(x, sresp[i], lw=0.5, linestyle="--", c=colors[i])
plt.plot(x, resp[0], lw=1, linestyle="-", c="k")
plt.plot(x, sresp[0], lw=1, linestyle="--", c="k")
plt.legend()
plt.xlabel("Segment")
plt.ylabel("Local Spearman")
plt.show()"""
"""
fig, axs=plt.subplots()
axs.set_visible(False)
ax1  = fig.add_axes([0.10,0.10,0.70,0.85])

ax1.set_yscale("log")
ax1.set_xscale("linear")

ax1.set_ylabel("Noise")
ax1.set_xlabel("Encoding")

seen_exps = []
ax1.axvline(x=0, lw=1, linestyle="--", c="k")
for key in data.keys():
    exp = data[key]["experiment_name"]

    if exp[0] != "0":
        continue
    if exp.endswith("_sup"):
        continue
    if exp.endswith("_inf"):
        continue
    if exp in seen_exps:
        continue
    else:
        seen_exps.append(exp)
    
    coded = np.load("../../output/pendulum/" + exp + "/testing/coded-100.npy")
    energies = np.load("../../output/pendulum/" + exp + "/testing/energies.npy")

    noise = exp[exp.rfind("_") + 1:]
    noise = float(1 / int(noise))

    ax1.axhline(y=noise, lw=0.35, linestyle="-", c="k")

    dsize = 102400
    coded = np.reshape(coded, (dsize))
    energies = np.tile(energies, (1, 10))
    energies = np.reshape(energies, (dsize))
    idx = np.random.permutation(dsize)
    coded = coded[idx]
    if expname == "nnoise":
        if orig_noises.index(noise) in [4, 5, 6, 9, 10]:
            coded = -1 * coded
    elif expname == "inoise":
        if orig_noises.index(noise) in [1,2,4,5, 10]:
            coded = -1 * coded

    coded = coded - np.quantile(coded, 0.5)
    energies = energies[idx]
    coded = coded[::1600]
    energies = energies[::1600]

    cols = []
    for i in range(0,np.size(energies)):
        cols.append(w(energies[i]))

    ax1.scatter(coded, np.repeat(noise, np.size(coded)), s=1.5, c=cols)
norm = matplotlib.colors.Normalize(vmin=0,vmax=1)
ax2  = fig.add_axes([0.85,0.10,0.05,0.85])
cb1  = matplotlib.colorbar.ColorbarBase(ax2,cmap=plt.get_cmap("plasma"),norm=norm,orientation='vertical')
plt.show()"""
    
