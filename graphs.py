#%%
import pickle

import numpy as np
from matplotlib import pyplot as plt

#%%
results = pickle.load(
    open("/home/joschnei/batch-active-preference-based-learning/out.pkl", "rb")
)

#%%
by_n = dict()
for epsilon, n in results.keys():
    tmp = by_n.get(n, dict())
    tmp[epsilon] = results[(epsilon, n)]
    by_n[n] = tmp

ns = list(by_n.keys())

#%%
for n in ns:
    epsilons = list(by_n[n].keys())
    fps = [by_n[n][e][0][1] for e in epsilons]
    plt.plot(epsilons, fps, label=str(n))
plt.xlabel("Epsilon")
plt.ylabel("False Positives")
plt.ylim((0, 10000))
plt.legend()
plt.show()

#%%
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 22})
for n in ns:
    epsilons = list(by_n[n].keys())
    fprs = [by_n[n][e][0][1] / np.sum(by_n[n][e][0]) for e in epsilons]
    plt.plot(epsilons, fprs, label=str(n))
plt.xlabel(r"$\epsilon$")
plt.ylabel("False Positive Rate")
plt.title(r"$\epsilon$-relaxation's Effect on FPR")
plt.ylim((0, 1.01))
plt.legend()
plt.show()

#%%
for n in ns:
    epsilons = list(by_n[n].keys())
    precisions = [
        by_n[n][e][0][1] / (by_n[n][e][0][1] + by_n[n][e][1][1]) for e in epsilons
    ]
    plt.plot(epsilons, precisions, label=str(n))
plt.xlabel("Epsilon")
plt.ylabel("False Discovery Rate (FP / (FP + TN))")
plt.ylim((0, 1.01))
plt.legend()
plt.show()

#%%
pairs = [(e, tmp[1][1]) for e, tmp in by_n[1000].items()]
epsilons = [e for e, _ in pairs]
tps = [tp for _, tp in pairs]
plt.plot(epsilons, tps)
plt.xlabel(r"$\epsilon$")
plt.ylabel("Aligned rewards (of 10,000)")
plt.ylim((0, 10000))
plt.title(r"$\epsilon$-Aligned Rewards")
plt.show()


# %%
