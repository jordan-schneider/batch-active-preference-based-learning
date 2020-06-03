#%%
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

#%%
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 22})

result = pickle.load(open(Path("questions") / str(1) / ("out.skip_noise.pkl"), "rb"))
ns = set()
for _, n in result.keys():
    ns.add(n)

palette = sns.color_palette("muted", len(ns))

palette_map = {str(n): palette[i] for i, n in enumerate(sorted(ns))}


def nest_by_second(
    dict_of_pairs: Dict[Tuple[Any, Any], Any]
) -> Dict[float, Dict[float, Any]]:
    by_second: Dict[float, Dict[float, Any]] = dict()
    for first, second in dict_of_pairs.keys():
        by_first = by_second.get(second, dict())
        by_first[first] = dict_of_pairs[(float(first), int(second))]
        by_second[second] = by_first
    return by_second


def get_df(rootdir: Path, ablation: str):
    df = pd.DataFrame(columns=["epsilon", "n", "tn", "fp", "tp"])
    for replication in range(1, 6):
        out_dict = nest_by_second(
            pickle.load(
                open(rootdir / str(replication) / ("out" + ablation + ".pkl"), "rb")
            )
        )
        tmp = pd.DataFrame(
            pd.DataFrame.from_dict(out_dict).stack(), columns=["confusion"]
        )
        tmp.index.names = ["epsilon", "n"]
        tmp = tmp.apply(
            lambda x: [
                int(x.confusion[0][0]),
                int(x.confusion[0][1]),
                int(x.confusion[1][1]),
            ]
            if x.confusion.shape == (2, 2)
            else [10000, 0, 0],
            result_type="expand",
            axis="columns",
        )
        tmp.columns = ["tn", "fp", "tp"]
        tmp = tmp.reset_index()
        df = df.append(tmp)

    df = df.convert_dtypes()

    # Seaborn tries to convert integer hues into rgb values. So we make them strings.
    df["n"] = df["n"].astype(str)
    df["fpr"] = df.fp / (df.fp + df.tn)

    return df


def plot_fpr(df: pd.DataFrame, rootdir: Path, ablation: str):
    g = sns.relplot(
        x="epsilon",
        y="fpr",
        hue="n",
        kind="line",
        palette=palette_map,
        data=df,
        ci=80,
        hue_order=["100", "200", "300", "400", "500", "1000"],
        legend="brief",
    )

    g._legend.texts[0].set_text("")

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("False Postive Rate")
    plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
    plt.xticks(
        ticks=np.linspace(0, 1, 11),
        labels=[0.0, "", "", "", "", 0.5, "", "", "", "", 1.0],
    )
    plt.savefig(rootdir / ("fpr" + ablation + ".png"))
    plt.show()


######################
# No simulated noise #
######################
#%%
skip_noise = get_df(Path("questions"), ".skip_noise")
skip_lp = get_df(Path("questions"), ".skip_noise.skip_lp")

rows_per_replication = skip_noise.epsilon.unique().size * skip_noise.n.unique().size
first_experiment = skip_noise.iloc[:rows_per_replication]

#%%
plot_fpr(skip_noise, Path("questions"), ".skip_noise")
plot_fpr(skip_lp, Path("questions"), ".skip_noise.skip_lp")

#%%
g = sns.relplot(
    x="epsilon",
    y="fpr",
    kind="line",
    data=first_experiment,
    hue="n",
    hue_order=["100", "200", "300", "400", "500", "1000"],
    palette=palette_map,
)
g._legend.texts[0].set_text("")

###################
# Simulated Noise #
###################

#%%

rootdir = Path("noisy-questions")

noise_with_filtering = get_df(rootdir=rootdir, ablation=".skip_noise")
noise_without_filtering = get_df(rootdir=rootdir, ablation="")
#%%
plot_fpr(df=noise_with_filtering, rootdir=rootdir, ablation=".skip_noise")
plot_fpr(df=noise_without_filtering, rootdir=rootdir, ablation="")

### SINGLE PLOTS/DEPRECATED

#%%

# rootdir = Path("questions")
# ablation = ".skip_noise"

# results = pickle.load(open(rootdir / str(1) / ("out" + ablation + ".pkl"), "rb"))
# by_n = nest_by_second(results)


# #%%
# for n in ns:
#     epsilons = list(by_n[n].keys())[1:]
#     if len(epsilons) > 1:
#         fprs = [by_n[n][e][0][1] / np.sum(by_n[n][e][0]) for e in epsilons]
#         plt.plot([0.0] + epsilons, [0.0] + fprs, label=str(n))
# plt.xlabel(r"$\epsilon$")
# plt.ylabel("False Positive Rate")
# plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
# plt.xticks(
#     ticks=np.linspace(0, 1, 11), labels=[0.0, "", "", "", "", 0.5, "", "", "", "", 1.0]
# )
# plt.ylim((0, 1.01))
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# # plt.savefig(Path("plots") / str(replication) / "fpr", bbox_inches="tight")
# plt.show()


# #%%
# pairs = [(e, tmp[1][1]) for e, tmp in by_n[1000].items() if e != 0.0]
# epsilons = [e for e, _ in pairs]
# tps = [tp / 10000 for _, tp in pairs]
# plt.plot([0.0] + epsilons, [0.0] + tps)
# plt.xlabel(r"$\epsilon$")
# plt.xticks(
#     ticks=np.linspace(0, 1, 11), labels=[0.0, "", "", "", "", 0.5, "", "", "", "", 1.0]
# )
# plt.ylabel("\% Aligned")
# plt.ylim((0, 1.01))
# plt.title(r"$\epsilon$-Aligned Rewards")
# plt.tight_layout()
# # plt.savefig(Path("plots") / str(replication) / "tps")
# plt.show()

# #%%
# for n in ns:
#     epsilons = list(by_n[n].keys())
#     precisions = [
#         by_n[n][e][0][1] / (by_n[n][e][0][1] + by_n[n][e][1][1]) for e in epsilons
#     ]
#     plt.plot(epsilons, precisions, label=str(n))
# plt.xlabel("Epsilon")
# plt.ylabel("False Discovery Rate (FP / (FP + TN))")
# plt.ylim((0, 1.01))
# plt.legend()
# plt.show()
# #%%
# for n in ns:
#     epsilons = list(by_n[n].keys())
#     fps = [by_n[n][e][0][1] for e in epsilons]
#     plt.plot(epsilons, fps, label=str(n))
# plt.xlabel("Epsilon")
# plt.ylabel("False Positives")
# plt.ylim((0, 10000))
# plt.legend()
# plt.show()
