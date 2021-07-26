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
        res[exp["old_params"]["noise"]] = exp["spearman"]
    else:
        supres[exp["old_params"]["noise"]] = exp["spearman"]
print(res)
for key in res.keys():
    if key in [0, 0.05, 0.125, 0.25]:
        a = [abs(x) for x in res[key][1:-1]]
        b = [abs(x) for x in supres[key][1:-1]]
        plt.plot([0, 1, 2, 3, 4, 5, 6], a, label=key)
        plt.plot([0, 1, 2, 3, 4,5,6], b, label=str(key) + "_s")
plt.legend()
plt.show()
