#%%
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

#%%
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 33})

interactive = getattr(sys, "ps1", None) is not None


def closefig():
    if interactive:
        plt.show()
    else:
        plt.close()


result = pickle.load(open(Path("questions") / str(1) / ("out.skip_noise.pkl"), "rb"))
ns = set()
for _, n in result.keys():
    ns.add(n)

palette = sns.color_palette("muted", len(ns))

palette_map = {str(n): palette[i] for i, n in enumerate(sorted(ns))}


n_replications = 10

xticks = np.linspace(0, 1.5, 16)
xlabels = [0.0, "", "", "", "", 0.5, "", "", "", "", 1.0, "", "", "", "", 1.5]


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
    for replication in range(1, n_replications + 1):
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
                int(x.confusion[1][0]),
                int(x.confusion[1][1]),
            ],
            result_type="expand",
            axis="columns",
        )
        tmp.columns = ["tn", "fp", "fn", "tp"]
        tmp = tmp.reset_index()
        df = df.append(tmp)

    df = df.convert_dtypes()

    # Seaborn tries to convert integer hues into rgb values. So we make them strings.
    df["n"] = df["n"].astype(str)
    df["fpr"] = df.fp / (df.fp + df.tn)
    df["tpf"] = df.tp / (df.tp + df.fp + df.tn)
    df["fnr"] = df.fn / (df.fn + df.tp)

    return df


def plot_fpr(df: pd.DataFrame, rootdir: Path, ablation: str):

    plt.figure(figsize=(10, 10))

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
        aspect=2,
    )

    g._legend.texts[0].set_text("")

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("False Postive Rate")
    plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
    plt.xticks(
        ticks=xticks, labels=xlabels,
    )
    plt.ylim((0, 1.01))
    plt.savefig(rootdir / ("fpr" + ablation + ".png"))
    closefig()


def plot_fnr(df: pd.DataFrame, rootdir: Path, ablation: str):

    plt.figure(figsize=(10, 10))

    g = sns.relplot(
        x="epsilon",
        y="fnr",
        hue="n",
        kind="line",
        palette=palette_map,
        data=df,
        ci=80,
        hue_order=["100", "200", "300", "400", "500", "1000"],
        legend="brief",
        aspect=2,
    )

    g._legend.texts[0].set_text("")

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("False Negative Rate")
    plt.title(r"$\epsilon$-Relaxation's Effect on FNR")
    plt.xticks(
        ticks=xticks, labels=xlabels,
    )
    plt.ylim((0, 1.01))
    plt.savefig(rootdir / ("fnr" + ablation + ".png"))
    closefig()


def plot_individual_fpr(df: pd.DataFrame, rootdir: Path, ablation: str):
    for i in range(1, n_replications + 1):
        g = sns.relplot(
            x="epsilon",
            y="fpr",
            kind="line",
            data=df[rows_per_replication * (i - 1) : rows_per_replication * i],
            hue="n",
            hue_order=["100", "200", "300", "400", "500", "1000"],
            palette=palette_map,
            aspect=2,
        )
        g._legend.texts[0].set_text("")
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("False Positive Rate")
        plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
        plt.xticks(
            ticks=xticks, labels=xlabels,
        )
        plt.ylim((0, 1.01))
        plt.savefig(rootdir / str(i) / ("fpr" + ablation + ".png"))
        closefig()


def plot_tp(df: pd.DataFrame, rootdir: Path, ablation: str):
    g = sns.relplot(
        x="epsilon",
        y="tpf",
        hue="n",
        kind="line",
        palette=palette_map,
        data=df,
        ci=80,
        hue_order=["100", "200", "300", "400", "500", "1000"],
        legend="brief",
        aspect=2,
    )
    g._legend.texts[0].set_text("")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("\% True Positives")
    plt.title(r"$\epsilon$-Relaxation's Effect on TP \%")
    plt.xticks(
        ticks=xticks, labels=xlabels,
    )
    plt.ylim((0, 1.01))
    plt.savefig(rootdir / ("tp" + ablation + ".png"))


def plot_individual_tp(df: pd.DataFrame, rootdir: Path, ablation: str):
    for i in range(1, n_replications + 1):
        g = sns.relplot(
            x="epsilon",
            y="tpf",
            hue="n",
            kind="line",
            palette=palette_map,
            data=df[rows_per_replication * (i - 1) : rows_per_replication * i],
            ci=80,
            hue_order=["100", "200", "300", "400", "500", "1000"],
            legend="brief",
            aspect=2,
        )
        g._legend.texts[0].set_text("")
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("\% True Positives")
        plt.title(r"$\epsilon$-Relaxation's Effect on TP \%")
        plt.xticks(
            ticks=xticks, labels=xlabels,
        )
        plt.ylim((0, 1.01))
        plt.savefig(rootdir / str(i) / ("tp" + ablation + ".png"))
        closefig()


######################
# No simulated noise #
######################
#%%

# rootdir = Path("questions")

# skip_noise = get_df(rootdir, ".skip_noise")
# # skip_lp = get_df(rootdir, ".skip_noise.skip_lp")

# rows_per_replication = skip_noise.epsilon.unique().size * skip_noise.n.unique().size
# first_experiment = skip_noise.iloc[:rows_per_replication]

# #%%
# plot_fpr(skip_noise, rootdir, ".skip_noise")
# # plot_individual_fpr(df=skip_noise, rootdir=rootdir, ablation=".skip_noise")

# plot_tp(df=skip_noise, rootdir=rootdir, ablation=".skip_noise")
# # plot_individual_tp(df=skip_noise, rootdir=rootdir, ablation=".skip_noise")

#%%
###################
# Simulated Noise #
###################

rootdir = Path("noisy-questions")

noise_with_filtering = get_df(rootdir=rootdir, ablation=".skip_noise")
noise_without_filtering = get_df(rootdir=rootdir, ablation="")
#%%
plot_fpr(df=noise_with_filtering, rootdir=rootdir, ablation=".skip_noise")
plot_fnr(df=noise_with_filtering, rootdir=rootdir, ablation=".skip_noise")

plot_fpr(df=noise_without_filtering, rootdir=rootdir, ablation="")
plot_fnr(df=noise_without_filtering, rootdir=rootdir, ablation="")
