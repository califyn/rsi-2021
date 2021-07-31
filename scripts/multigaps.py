import json
import traceback
import numpy as np
import copy
try:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception:
    import matplotlib
    matplotlib.use('ps')
    pass
import matplotlib.pyplot as plt

expname = "spreadgapm"

gap_borders = False

with open("../data/master_experiments.json", "r") as f:
    data = json.load(f)

keys = list(data.keys())
for key in keys:
    if expname not in data[key]["experiment_name"]:
        data.pop(key)

gapsizes = [1,2,3,4,6,8,10]
orig_gaps= copy.deepcopy(gapsizes)
for i in range(0, len(gapsizes)):
    gapsizes[i] = str(int(gapsizes[i]))
print(gapsizes)

if gap_borders:
    res = np.zeros((7,5,11))
    sres= np.zeros((7,5,11))
    resp = np.zeros((7,5,10))
    sresp = np.zeros((7,5,10))
else:
    res = np.zeros((7,5))
    sres = np.zeros((7,5))
    resp = np.zeros((7,5,7))
    sresp = np.zeros((7,5,7))

for key in data.keys():
    exp = data[key]["experiment_name"]
    #print(exp)
    #print(data[key]["full_spearman"])
    trial = int(exp[0])
    sup = False
    if exp.endswith("_sup"):
        sup = True
        exp = exp[:-4]
    num = exp[exp.rfind("_") + 1:]

    #try:
    if gap_borders and data[key]["new_params"]["test_mode"] == "same":
        print(num)
        if sup:
            #print(data[key]["k_spearman"])
            sres[gapsizes.index(num), trial, :int(num)+1] = data[key]["k_spearman"][1:-1:2]
            sresp[gapsizes.index(num), trial, :int(num)] = np.array(data[key]["k_spearman"])[2:-1:2]
        else:
            #print(data[key]["k_spearman"])
            res[gapsizes.index(num), trial, :int(num)+1] = data[key]["k_spearman"][1:-1:2]
            resp[gapsizes.index(num), trial, :int(num)] = np.array(data[key]["k_spearman"])[2:-1:2]
    elif not gap_borders and data[key]["new_params"]["test_mode"] == "energy":
        if sup:
            sres[gapsizes.index(num), trial] = data[key]["full_spearman"]
            sresp[gapsizes.index(num), trial] = np.array(data[key]["k_spearman"])[1:-1]
        else:
            res[gapsizes.index(num), trial] = data[key]["full_spearman"]
            resp[gapsizes.index(num), trial] = np.array(data[key]["k_spearman"])[1:-1]
    #except Exception as err:traceback.print_stack()
        #traceback.print_tb(err.__traceback__)
        pass

ok = np.ones((7,5))
for x in range(0, 7):
    for y in range(0,5):
        if gap_borders:
            l = res[x,y,:].tolist()
        else:
            l = resp[x,y,:].tolist()
        l = [z for z in l if z != 0]
        if not all(z > 0 for z in l) and not all(z < 0 for z in l):
            ok[x,y] = 0
print(ok)
print(np.argwhere(ok[2]==1))
#print(res)
#print(sres)
#print(resp)
#print(sresp)
w = plt.get_cmap('plasma')
res = np.abs(res)
resp = np.abs(resp)
sres = np.abs(sres)
sresp = np.abs(sresp)

#res = np.log(1 - res)
#resp = np.log(1 - resp)
#sres = np.log(1 - sres)
#sresp = np.log(1 - sresp)



if gap_borders:
    plot_res = []
    plot_resp = []
    splot_res = []
    splot_resp= []
    plot_res_dark = []
    plot_resp_dark = []
    for it, i in enumerate(orig_gaps):
        plot_res.append(np.mean(res[it,:,:i+1]))
        plot_resp.append(np.mean(resp[it,:,:i]))
        splot_res.append(np.mean(sres[it,:,:i+1]))
        splot_resp.append(np.mean(sresp[it,:,:i]))
        plot_res_dark.append(np.mean(res[it,np.argwhere(ok[it]==1),:i+1]))
        plot_resp_dark.append(np.mean(resp[it,np.argwhere(ok[it]==1),:i]))
else:
    plot_res = []
    plot_resp = []
    splot_res = []
    splot_resp= []
    plot_res_dark = []
    plot_resp_dark = []
    for it, i in enumerate(orig_gaps):
        plot_res.append(np.mean(res[it]))
        splot_res.append(np.mean(sres[it]))
        plot_resp.append(np.mean(resp[it]))
        splot_resp.append(np.mean(sresp[it]))
        plot_res_dark.append(np.mean(res[it,np.argwhere(ok[it]==1)]))
        plot_resp_dark.append(np.mean(resp[it,np.argwhere(ok[it]==1)]))

def OneMinusLog(arr):
    return -1 * (np.log(1 - arr))

def OneMinusLogInv(arr):
    return 1 - np.exp(-1 * arr)

fig, ax1 = plt.subplots()

ax1.set_xlabel("Gap Size")
ax1.set_ylabel("Normal Spearman", color=[1,0,0])
ax1.tick_params(axis='y', labelcolor=[1,0,0])

ax1.plot(orig_gaps, plot_res_dark, lw=0.5, color=[0.5,0,0])
ax1.plot(orig_gaps, plot_res, lw=0.75, color=[1,0,0])
ax1.plot(orig_gaps, splot_res, lw=0.75, color=[1,0,0], linestyle="--")

ax2=ax1.twinx()

ax2.set_yscale('function', functions=(OneMinusLog, OneMinusLogInv))
ax2.set_ylabel("Interpolation Spearman", color=[0,1,0])
ax2.tick_params(axis='y', labelcolor=[0,1,0])

ax2.plot(orig_gaps, plot_resp_dark, lw=0.5, color=[0,0.5,0])
ax2.plot(orig_gaps, plot_resp, lw=0.75, color=[0,1,0])
ax2.plot(orig_gaps, splot_resp, lw=0.75, color=[0,1,0], linestyle="--")


fig.tight_layout()
plt.show()

#plt.clf()

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
    
