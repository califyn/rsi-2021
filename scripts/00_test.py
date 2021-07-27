import json
import numpy as np

data = {}
with open("../data/master_experiments.json", "r") as f:
    data = json.load(f)

keys = list(data.keys())
seen_exp=[]
for key in keys:
    if "noise" not in data[key]["experiment_name"] and "vnladim_1" not in data[key]["experiment_name"]:
        data.pop(key)
        continue
    if data[key]["experiment_name"] in seen_exp:
        data.pop(key)
        continue
    else:
        seen_exp.append(data[key]["experiment_name"])
    print("0inoise_inf" in seen_exp)
    try:
        if data[key]["old_params"]["gnoise"] != 0 or data[key]["old_params"]["nnoise"] != 0 or data[key]["new_params"]["gnoise"] != 0 or data[key]["new_params"]["nnoise"] != 0:
            data.pop(key)
    except:
        if data[key]["old_params"]["noise"] != 0 or data[key]["new_params"]["noise"] != 0:
            data.pop(key)
        pass

sp = []
sps = []
for key in data.keys():
    print(data[key]["experiment_name"])
    if data[key]["experiment_name"].endswith("_sup"):
        sps.append(data[key]["full_speaman"])
    else:
        sp.append(data[key]["full_speaman"])
#sp = np.abs(np.array(sp))[:,1:-1]
#sps = np.abs(np.array(sps))[:,1:-1]
#sp = np.mean(sp, axis=1)
#sps = np.mean(sps, axis=1)
print(np.mean(np.abs(np.array(sp))))
print(np.mean(np.abs(np.array(sps))))
