#%%
import pickle
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


result = pickle.load(open(Path("human/subject-1/confusion.pkl"), "rb"))
ns = set()
deltas = set()
for _, delta, n in result.keys():
    ns.add(n)
    deltas.add(delta)

palette = sns.color_palette("muted", max(len(ns), len(deltas)))

ns_palette_map = {str(n): palette[i] for i, n in enumerate(sorted(ns))}
deltas_pallete_map = {delta: palette[i] for i, delta in enumerate(sorted(deltas))}

n_replications = 10

xticks = np.linspace(0, 1.5, 16)
xlabels = [0.0, "", "", "", "", 0.5, "", "", "", "", 1.0, "", "", "", "", 1.5]


def read_confusion(dir: Path, ablation: str = ""):
    """ Read dict of confusion matrices. """
    out_dict = pickle.load(open(dir / f"confusion{ablation}.pkl", "rb"))

    out = pd.Series(out_dict).reset_index()
    out.columns = ["epsilon", "delta", "n", "confusion"]
    out = out.join(
        out.apply(
            lambda x: [
                int(x.confusion[0][0]),
                int(x.confusion[0][1]),
                int(x.confusion[1][0]),
                int(x.confusion[1][1]),
            ],
            result_type="expand",
            axis="columns",
        )
    )
    del out["confusion"]
    out.columns = ["epsilon", "delta", "n", "tn", "fp", "fn", "tp"]
    return out


def get_df(rootdir: Path, ablation: str, replications: Optional[int] = None):
    df = pd.DataFrame(columns=["epsilon", "n", "tn", "fp", "tp"])
    if replications is not None:
        for replication in range(1, replications + 1):
            df = df.append(
                read_confusion(rootdir / str(replication), ablation=ablation)
            )
    else:
        df = read_confusion(rootdir, ablation=ablation)

    df = df.convert_dtypes()

    # Seaborn tries to convert integer hues into rgb values. So we make them strings.
    df["n"] = df["n"].astype(str)
    df["fpr"] = df.fp / (df.fp + df.tn)
    df["tpf"] = df.tp / (df.tp + df.fp + df.tn)
    df["fnr"] = df.fn / (df.fn + df.tp)

    return df


def plot_fpr(df: pd.DataFrame, rootdir: Path, ablation: str, hue: str = "n"):

    plt.figure(figsize=(10, 10))

    pallete, hue_order = get_hue(hue, df)

    g = sns.relplot(
        x="epsilon",
        y="fpr",
        hue=hue,
        kind="line",
        palette=pallete,
        data=df,
        ci=80,
        hue_order=hue_order,
        legend="brief",
        aspect=2,
    )

    g._legend.texts[0].set_text("")

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("False Postive Rate")
    plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
    # plt.xticks(
    #     ticks=xticks, labels=xlabels,
    # )
    plt.ylim((0, 1.01))
    plt.savefig(rootdir / ("fpr" + ablation + ".png"))
    closefig()


def plot_fnr(df: pd.DataFrame, rootdir: Path, ablation: str, hue: str = "n"):

    plt.figure(figsize=(10, 10))

    pallete, hue_order = get_hue(hue, df)

    g = sns.relplot(
        x="epsilon",
        y="fnr",
        hue=hue,
        kind="line",
        palette=pallete,
        data=df,
        ci=80,
        hue_order=hue_order,
        legend="brief",
        aspect=2,
    )

    g._legend.texts[0].set_text("")

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("False Negative Rate")
    plt.title(r"$\epsilon$-Relaxation's Effect on FNR")
    # plt.xticks(
    #     ticks=xticks, labels=xlabels,
    # )
    plt.ylim((0, 1.01))
    plt.savefig(rootdir / ("fnr" + ablation + ".png"))
    closefig()


def get_hue(hue: str, df):
    if hue == "n":
        pallete = ns_palette_map
        hue_order = df.ns.unique()
    elif hue == "delta":
        pallete = deltas_pallete_map
        hue_order = df.delta.unique()
    else:
        raise ValueError("Hue must be n or delta")

    return pallete, hue_order


def plot_individual_fpr(df: pd.DataFrame, rootdir: Path, ablation: str, hue: str = "n"):

    pallete, hue_order = get_hue(hue, df)

    for i in range(1, n_replications + 1):

        g = sns.relplot(
            x="epsilon",
            y="fpr",
            hue=hue,
            kind="line",
            palette=pallete,
            data=df[rows_per_replication * (i - 1) : rows_per_replication * i],
            hue_order=hue_order,
            legend="brief",
            aspect=2,
        )
        g._legend.texts[0].set_text("")
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("False Positive Rate")
        plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
        # plt.xticks(
        #     ticks=xticks, labels=xlabels,
        # )
        plt.ylim((0, 1.01))
        plt.savefig(rootdir / str(i) / ("fpr" + ablation + ".png"))
        closefig()


def plot_largest_fpr(df: pd.DataFrame, rootdir: Path, ablation: str, n):
    df = df[df.n == n]
    plt.figure(figsize=(10, 10))

    g = sns.relplot(
        x="epsilon", y="fpr", kind="line", data=df, ci=80, legend="brief", aspect=2,
    )

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("False Postive Rate")
    plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
    # plt.xticks(
    #     ticks=xticks, labels=xlabels,
    # )
    plt.savefig(rootdir / ("fpr.largest" + ablation + ".png"))
    closefig()


def plot_tp(df: pd.DataFrame, rootdir: Path, ablation: str, hue: str = "n"):
    pallete, hue_order = get_hue(hue, df)
    g = sns.relplot(
        x="epsilon",
        y="tpf",
        hue=hue,
        kind="line",
        palette=pallete,
        data=df,
        ci=80,
        hue_order=hue_order,
        legend="brief",
        aspect=2,
    )
    g._legend.texts[0].set_text("")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("\% True Positives")
    plt.title(r"$\epsilon$-Relaxation's Effect on TP \%")
    # plt.xticks(
    #     ticks=xticks, labels=xlabels,
    # )
    plt.ylim((0, 1.01))
    plt.savefig(rootdir / ("tp" + ablation + ".png"))


def plot_individual_tp(df: pd.DataFrame, rootdir: Path, ablation: str, hue: str = "n"):
    pallete, hue_order = get_hue(hue, df)
    for i in range(1, n_replications + 1):
        g = sns.relplot(
            x="epsilon",
            y="tpf",
            hue=hue,
            kind="line",
            palette=palette,
            data=df[rows_per_replication * (i - 1) : rows_per_replication * i],
            ci=80,
            hue_order=hue_order,
            legend="brief",
            aspect=2,
        )
        g._legend.texts[0].set_text("")
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("\% True Positives")
        plt.title(r"$\epsilon$-Relaxation's Effect on TP \%")
        # plt.xticks(
        #     ticks=xticks, labels=xlabels,
        # )
        plt.ylim((0, 1.01))
        plt.savefig(rootdir / str(i) / ("tp" + ablation + ".png"))
        closefig()


def fill_na(df):
    df.fpr.fillna(1.0, inplace=True)
    df.fnr.fillna(0.0, inplace=True)


# %%
# Human results
rootdir = Path("speed")

#%%
# Reconstruct agreements from indices output.
# They agree now, ignore this.
rewards = np.load(rootdir / "fake_rewards.npy")
normals = np.load(rootdir / "normals.npy")
prefs = np.load(rootdir / "preferences.npy")
indices = pickle.load(open(rootdir / "indices.pkl", "rb"))

normals = (normals.T * prefs).T

agreements = pd.DataFrame(columns=["epsilon", "delta", "n", "aligned", "value"])
j = 0

validation_test = normals.T
for (epsilon, delta, n), i in indices.items():
    test = normals[i]
    aligned = np.all(np.dot(rewards, test.T) > 0, axis=1)

    aligned_rewards = rewards[aligned]
    misaligned_rewards = rewards[np.logical_not(aligned)]

    for agreement in np.mean(np.dot(aligned_rewards, validation_test) > 0, axis=1):
        agreements.loc[j] = [epsilon, delta, n, True, agreement]
        j += 1

    for agreement in np.mean(np.dot(misaligned_rewards, validation_test) > 0, axis=1):
        agreements.loc[j] = [epsilon, delta, n, False, agreement]
        j += 1

#%%
# Load from agreements.pkl
agreements = pd.Series(pickle.load(open(rootdir / "agreement.pkl", "rb"))).reset_index()
agreements = agreements.join(
    agreements.apply(lambda x: list(x[0]), result_type="expand", axis="columns"),
    rsuffix="_",
)
del agreements["0"]
agreements.columns = ["epsilon", "delta", "n", "aligned", "misaligned"]
agreements = agreements.set_index(["epsilon", "delta", "n"]).stack().reset_index()
agreements.columns = ["epsilon", "delta", "n", "aligned", "value"]
agreements = agreements.explode("value")
agreements["aligned"] = agreements.aligned == "aligned"

agreements.value = agreements.value.apply(lambda x: float(x))
agreements = agreements.dropna()


#%%
# Plot agreements

epsilon = 0.3
delta = 0.05
n = 900

tmp = agreements[
    np.logical_and(
        np.logical_and(agreements.epsilon == epsilon, agreements.delta == delta),
        agreements.n == n,
    )
]

tmp[tmp.aligned].value.hist(label="aligned", alpha=0.3)

tmp[tmp.aligned == False].value.hist(label="misaligned", alpha=0.3)
plt.legend()
plt.show()
#%%

# mean_agreement = (
# agreements.groupby(["epsilon", "delta", "n", "aligned"]).mean().reset_index()
# )
# plt.hist(mean_agreement[mean_agreement.aligned].value, label="aligned", alpha=0.3)
# plt.hist(
#     mean_agreement[np.logical_not(mean_agreement.aligned)].value,
#     label="unaligned",
#     alpha=0.3,
# )

# plt.xlabel("\% holdout agreement")
# plt.legend()
# plt.show()

#%%
# Graphs when you have ground truth alignment of agents
# human = get_df(rootdir=rootdir, ablation="")
# fill_na(human)

# strict_indices = (human.tp + human.tn) != 0

# strict_indices = strict_indices.index[strict_indices]
# start_index = max(strict_indices[0] - 1, 0)
# empty_indices = (human.tn + human.fn) == 0
# empty_indices = empty_indices.index[empty_indices]
# stop_index = min(empty_indices[0], len(human) - 1)

# human = human.iloc[start_index:stop_index]

# plot_fpr(df=human, rootdir=rootdir, ablation="", hue="delta")
# plot_fnr(df=human, rootdir=rootdir, ablation="", hue="delta")
# plot_tp(df=human, rootdir=rootdir, ablation="", hue="delta")

#%%

rootdir = Path("questions")

skip_noise = get_df(rootdir, ".skip_noise", replications=n_replications)
# skip_lp = get_df(rootdir, ".skip_noise.skip_lp", replications=n_replications)

rows_per_replication = skip_noise.epsilon.unique().size * skip_noise.n.unique().size
# first_experiment = skip_noise.iloc[:rows_per_replication]

# %%
plot_fpr(skip_noise, rootdir, ".skip_noise")
plot_fnr(skip_noise, rootdir, ".skip_noise")
plot_largest_fpr(skip_noise, rootdir, ".skip_noise", n="1000")
plot_individual_fpr(df=skip_noise, rootdir=rootdir, ablation=".skip_noise")

plot_tp(df=skip_noise, rootdir=rootdir, ablation=".skip_noise")
plot_individual_tp(df=skip_noise, rootdir=rootdir, ablation=".skip_noise")

#%%
###################
# Simulated Noise #
###################

rootdir = Path("noisy-questions")

noise_with_filtering = get_df(rootdir=rootdir, ablation="", replications=n_replications)
fill_na(noise_with_filtering)

noise_without_filtering = get_df(
    rootdir=rootdir, ablation=".skip_noise", replications=n_replications
)
fill_na(noise_without_filtering)
#%%
plot_fpr(df=noise_without_filtering, rootdir=rootdir, ablation=".skip_noise")
plot_fnr(df=noise_without_filtering, rootdir=rootdir, ablation=".skip_noise")
plot_largest_fpr(
    df=noise_without_filtering, rootdir=rootdir, ablation=".skip_noise", n="1000"
)


plot_fpr(df=noise_with_filtering, rootdir=rootdir, ablation="")
plot_fnr(df=noise_with_filtering, rootdir=rootdir, ablation="")
plot_largest_fpr(df=noise_with_filtering, rootdir=rootdir, ablation="", n="1000")


# %%
