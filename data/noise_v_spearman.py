try:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception:
    import matplotlib
    matplotlib.use('ps')
    pass
import matplotlib.pyplot as plt

import json

with open("master_experiments.json") as f:
    allexp = json.load(f)

exp_type = "nnoise"
test_noise = 0.0
res = {}
supres = {}
for key in allexp.keys():
    exp = allexp[key]

    if not exp_type in exp["experiment_name"]:
        continue

    if not exp["new_params"]["noise"] == test_noise:
        continue

    if not "sup" in exp["experiment_name"]:
        res[exp["old_params"]["noise"]] = exp["full_speaman"]
    else:
        supres[exp["old_params"]["noise"]] = exp["full_speaman"]
print(res)
plt.loglog()
x = []
y1 = []
y2 = []
for key in res.keys():
    if key != 0:
        x.append(key)
        y1.append(1 - abs(res[key]))
        y2.append(1 - abs(supres[key]))
plt.plot(x, y1, label="self")
plt.plot(x, y2, label="sup")
plt.legend()
plt.show()
