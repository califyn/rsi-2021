import json
import numpy as np

data = {}
with open("../data/master_experiments.json", "r") as f:
    data = json.load(f)

keys = list(data.keys())
for key in keys:
    if not "vnladim" in data[key]["experiment_name"]:
        del data[key]
    else:
        print(data[key]["experiment_name"])
final_sp = [[],[],[]]
final_sps = [[],[],[]]
final_sg = [[],[],[]]
final_sgs = [[],[],[]]
for key in data.keys():
    dim = data[key]["experiment_name"]
    sup = False
    if dim.endswith("_sup"):
        sup = True
        dim = dim[:-4]
    dim = dim[dim.rfind("_") + 1:]
    dim = int(dim) - 1
    print(dim)
    sp = data[key]["full_speaman"]
    sg = data[key]["spearman"]
    if sup:
        final_sps[dim].append(sp)
        final_sgs[dim].append(sg)
    else:
        final_sp[dim].append(sp)
        final_sg[dim].append(sg)
print(final_sg)
final_sp = np.abs(np.array(final_sp))
final_sps = np.abs(np.array(final_sps))
final_sg = np.abs(np.array(final_sg))[:,:,1:-1]
final_sgs = np.abs(np.array(final_sgs))[:,:,1:-1]
print(final_sgs.shape)
print(final_sp[1])
print(final_sp[2])

for i in range(0, 3):
    if i == 1:
        print("sp self")
        print(np.mean(final_sp[i,[0,2,4]]))
        print("sg self")
        print(np.mean(final_sg[i,[0,2,4]]))
    else:
        print("sp self")
        print(np.mean(final_sp[i]))
        print("sg self")
        print(np.mean(final_sg[i]))

    print("sp sup")
    print(np.mean(final_sps[i]))
    print("sg sup")
    print(np.mean(final_sgs[i]))
    


