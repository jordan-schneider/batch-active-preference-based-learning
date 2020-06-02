#%%
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

#%%
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 22})

replication = 4

results = pickle.load(open(Path("questions") / str(replication) / "out.pkl", "rb"))


#%%
by_n = dict()
for epsilon, n in results.keys():
    tmp = by_n.get(n, dict())
    tmp[epsilon] = results[(epsilon, n)]
    by_n[n] = tmp

ns = list(by_n.keys())


#%%
for n in ns:
    epsilons = list(by_n[n].keys())[1:]
    if len(epsilons) > 1:
        fprs = [by_n[n][e][0][1] / np.sum(by_n[n][e][0]) for e in epsilons]
        plt.plot([0.0] + epsilons, [0.0] + fprs, label=str(n))
plt.xlabel(r"$\epsilon$")
plt.ylabel("False Positive Rate")
plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
plt.xticks(
    ticks=np.linspace(0, 1, 11), labels=[0.0, "", "", "", "", 0.5, "", "", "", "", 1.0]
)
plt.ylim((0, 1.01))
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig(Path("plots") / str(replication) / "fpr", bbox_inches="tight")
plt.show()


#%%
pairs = [(e, tmp[1][1]) for e, tmp in by_n[1000].items() if e != 0.0]
epsilons = [e for e, _ in pairs]
tps = [tp / 10000 for _, tp in pairs]
plt.plot([0.0] + epsilons, [0.0] + tps)
plt.xlabel(r"$\epsilon$")
plt.xticks(
    ticks=np.linspace(0, 1, 11), labels=[0.0, "", "", "", "", 0.5, "", "", "", "", 1.0]
)
plt.ylabel("\% Aligned")
plt.ylim((0, 1.01))
plt.title(r"$\epsilon$-Aligned Rewards")
plt.tight_layout()
plt.savefig(Path("plots") / str(replication) / "tps")
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
for n in ns:
    epsilons = list(by_n[n].keys())
    fps = [by_n[n][e][0][1] for e in epsilons]
    plt.plot(epsilons, fps, label=str(n))
plt.xlabel("Epsilon")
plt.ylabel("False Positives")
plt.ylim((0, 10000))
plt.legend()
plt.show()
