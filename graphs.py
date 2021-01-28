import pickle
import sys
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, cast

import fire  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from typing_extensions import Literal  # type: ignore

from run_tests import Experiment

Style = Union[Literal["POSTER"], Literal["PAPER"], Literal["ICML"]]


def get_interactive():
    """ Cursed magic for determining if the code is being run in an interactive environment. """
    return getattr(sys, "ps1", None) is not None


def closefig(out: Optional[Path] = None, transparent: bool = False):
    if get_interactive() and out is None:
        plt.show()
    else:
        if out is not None:
            plt.savefig(out, transparent=transparent)
        plt.close()


def make_xaxis():
    # TODO(joschnei): In theory I should be able to specify n_ticks, n_labels, lower, upper and do
    # the math, but it's not worth it.
    xticks = np.linspace(0, 1.5, 16)
    xlabels = [0.0, "", "", "", "", 0.5, "", "", "", "", 1.0, "", "", "", "", 1.5]
    return xticks, xlabels


def make_palette_maps(experiments: Sequence[Experiment]):
    """Given a sequence of experimental parameters, generate palette maps for representing
    parameter values."""
    ns = set()
    deltas = set()
    for _, delta, n in experiments:
        ns.add(n)
        deltas.add(delta)

    palette = sns.color_palette("muted", max(len(ns), len(deltas)))

    ns_palette_map = {str(n): palette[i] for i, n in enumerate(sorted(ns))}
    deltas_palette_map = {delta: palette[i] for i, delta in enumerate(sorted(deltas))}
    return ns_palette_map, deltas_palette_map


def get_hue(hue: str, df):
    ns_palette_map, deltas_palette_map = make_palette_maps(
        df.loc[:, ["epsilon", "delta", "n"]].drop_duplicates().to_numpy()
    )
    if hue == "n":
        palette = ns_palette_map
        hue_order = df["n"].unique()
    elif hue == "delta":
        palette = deltas_palette_map
        hue_order = df["delta"].unique()
    else:
        raise ValueError("Hue must be n or delta")

    return palette, hue_order


# HUMAN EXPERIMENTS


def check(
    normals: np.ndarray,
    indices: np.ndarray,
    rewards: np.ndarray,
    saved_agreements: Dict[Experiment, np.ndarray],
):
    """Reconstruct alignment decisions and check that they agree with cached values."""
    j = 0
    agreements = pd.DataFrame(columns=["epsilon", "delta", "n", "aligned", "value"])
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

    assert agreements.keys() == saved_agreements.keys()
    for key in agreements.keys():
        assert np.all(agreements[key] == saved_agreements[key])


def make_agreements(file) -> pd.DataFrame:
    """In some of the human conditions, we hold out questions. Each randomly generated agent is
    given our test and then asked it's opinion on every hold out question.

    agreements.pkl is a Dict[Experiment, Tuple(ndarray, ndarray)] where each array element
    contains the fraction of holdout questions a single agent answered correctly. The first array
    contains agents that passed our test, and the second contains agents that didn't pass our test.

    This method massages that data into a DataFrame with experiments as they keys, a column
    for predicted alignment, and a column for the fraction of holdout questions answered correctly.
    """
    agreements = pd.Series(pickle.load(file)).reset_index()
    agreements = agreements.join(
        agreements.apply(lambda x: list(x[0]), result_type="expand", axis="columns"), rsuffix="_",
    )
    del agreements["0"]
    agreements.columns = ["epsilon", "delta", "n", "aligned", "misaligned"]
    agreements = agreements.set_index(["epsilon", "delta", "n"]).stack().reset_index()
    agreements.columns = ["epsilon", "delta", "n", "aligned", "value"]
    agreements = agreements.explode("value")
    agreements["aligned"] = agreements.aligned == "aligned"

    agreements.value = agreements.value.apply(lambda x: float(x))
    agreements = agreements.dropna()
    return agreements


def plot_agreements(
    agreements: pd.DataFrame, epsilon: float, delta: float, n: int, out: Optional[Path] = None,
) -> None:
    """Plots histograms of how many agents had different amounts of holdout agreement for agents
    prediced tobe aligned and misaligned."""
    tmp = agreements[
        np.logical_and(
            np.logical_and(agreements.epsilon == epsilon, agreements.delta == delta),
            agreements.n == n,
        )
    ]

    tmp[tmp.aligned].value.hist(label="aligned", alpha=0.3)
    tmp[tmp.aligned == False].value.hist(label="misaligned", alpha=0.3)

    plt.xlabel("Hold out agreement")
    plt.legend()
    closefig(out)


def plot_mean_agreement(agreements: pd.DataFrame, out: Optional[Path] = None) -> None:
    mean_agreement = agreements.groupby(["epsilon", "delta", "n", "aligned"]).mean().reset_index()
    plt.hist(mean_agreement[mean_agreement.aligned].value, label="aligned", alpha=0.3)
    plt.hist(
        mean_agreement[np.logical_not(mean_agreement.aligned)].value, label="unaligned", alpha=0.3,
    )

    plt.xlabel("\% holdout agreement")
    plt.legend()
    closefig(out)


def make_human_confusion(
    label_path: Path = Path("questions/gt_rewards/alignment.npy"),
    prediction_path: Path = Path("questions/test_results.skip_noise.pkl"),
) -> pd.DataFrame:
    label = np.load(label_path)
    predictions: Dict[Experiment, np.ndarray] = pickle.load(open(prediction_path, "rb"))

    confusions = []
    for experiment, prediction in predictions.items():
        epsilon, delta, n = experiment
        if n <= 0:
            continue
        confusion = confusion_matrix(y_true=label, y_pred=prediction, labels=[False, True])
        confusions.append(
            (*experiment, confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1],)
        )

    df = pd.DataFrame(confusions, columns=["epsilon", "delta", "n", "tn", "fp", "fn", "tp"],)

    # idx = pd.MultiIndex.from_tuples(
    #     confusion_dict.keys(), names=["epsilon", "delta", "n"]
    # )

    # df = pd.Series(confusion_dict.values(), index=idx).unstack(-1)
    # df.columns = ["tp", "fp", "fn", "tn"]

    # TODO(joschnei): Factor common confusion stuff out.
    df = df.convert_dtypes()

    # Seaborn tries to convert integer hues into rgb values. So we make them strings.
    df["fpr"] = df.fp / (df.fp + df.tn)
    df["tpf"] = df.tp / (df.tp + df.fp + df.tn)
    df["fnr"] = df.fn / (df.fn + df.tp)
    df["acc"] = (df.tn + df.tp) / (df.tp + df.fp + df.tn + df.fn)

    df = df.sort_values(by="n")
    df["n"] = df["n"].astype(str)

    return df


# SIMULATIONS


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
    out["n"] = out["n"].astype(str)
    out["fpr"] = out.fp / (out.fp + out.tn)
    out["tpf"] = out.tp / (out.tp + out.fp + out.tn)
    out["fnr"] = out.fn / (out.fn + out.tp)
    out["acc"] = (out.tp + out.tn) / (out.tp + out.tn + out.fp + out.fn)
    return out


def read_replications(rootdir: Path, ablation: str, replications: Optional[int] = None):
    df = pd.DataFrame(columns=["epsilon", "n", "tn", "fp", "tp"])
    if replications is not None:
        for replication in range(1, replications + 1):
            df = df.append(read_confusion(rootdir / str(replication), ablation=ablation))
    else:
        df = read_confusion(rootdir, ablation=ablation)

    df = df.convert_dtypes()

    # Seaborn tries to convert integer hues into rgb values. So we make them strings.
    df["n"] = df["n"].astype(str)
    df["fpr"] = df.fp / (df.fp + df.tn)
    df["tpf"] = df.tp / (df.tp + df.fp + df.tn)
    df["fnr"] = df.fn / (df.fn + df.tp)

    return df


def plot_fpr(df: pd.DataFrame, rootdir: Path, ablation: str, style: Style, hue: str = "n"):
    plt.figure(figsize=(10, 10))

    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis()

    # TODO(joschnei): The way I was doing this before meant that the different axes had different
    # fonts. Using sns for everything would probalby fix that, but sns is undocumented garbage.
    # Switch to pure matplotlib if you can, and figure out how sns works if you can't.

    g = sns.relplot(
        x="epsilon",
        y="fpr",
        hue=hue,
        kind="line",
        palette=palette,
        data=df,
        ci=80,
        hue_order=hue_order,
        legend=False,
        # aspect=2,
        facet_kws={"legend_out": True},
    )

    # g._legend.texts[0].set_text("")

    if style == "ICML":
        g.set_axis_labels(r"$\epsilon$", "False Positive Rate")
        g.add_legend(title="")
        # plt.ylabel("False Postive Rate")
        # plt.xlabel(r"$\epsilon$")
        # plt.xticks(ticks=xticks, labels=xlabels, size="small")
        # plt.ylim((0, 1.01))
    elif style == "POSTER":
        raise NotImplementedError()
    elif style == "NEURIPS":
        raise NotImplementedError()
    else:
        raise ValueError(f"Style {style} not defined.")

    plt.savefig(rootdir / ("fpr" + ablation + ".pdf"))
    closefig()


def plot_fnr(df: pd.DataFrame, rootdir: Path, ablation: str, style: Style, hue: str = "n"):
    plt.figure(figsize=(10, 10))

    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis()

    g = sns.relplot(
        x="epsilon",
        y="fnr",
        hue=hue,
        kind="line",
        palette=palette,
        data=df,
        ci=80,
        hue_order=hue_order,
        legend="brief",
        aspect=2,
    )

    g._legend.texts[0].set_text("")

    if style == "ICML":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("False Negative Rate")
        plt.xticks(
            ticks=xticks, labels=xlabels,
        )
        plt.ylim((0, 1.01))
    elif style == "POSTER":
        raise NotImplementedError()
    elif style == "NEURIPS":
        raise NotImplementedError()
    else:
        raise ValueError(f"Style {style} not defined.")

    plt.savefig(rootdir / ("fnr" + ablation + ".pdf"))
    closefig()


def plot_accuracy(
    df: pd.DataFrame, rootdir: Path, ablation: str, style: Style, hue: str = "n",
):
    plt.figure(figsize=(10, 10))

    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis()

    g = sns.relplot(
        x="epsilon",
        y="acc",
        hue=hue,
        kind="line",
        palette=palette,
        data=df,
        ci=80,
        hue_order=hue_order,
        legend="brief",
        aspect=2,
    )

    g._legend.texts[0].set_text("")

    if style == "POSTER":
        plt.xlabel("Value Slack")
        plt.ylabel("Accuracy")
        plt.xticks(
            ticks=xticks, labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = True
    elif style == "NEURIPS":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("Accuracy")
        plt.title(r"$\epsilon$-Relaxation's Effect on Accuracy")
        plt.xticks(
            ticks=xticks, labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = False
    elif style == "ICML":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("Accuracy")
        plt.xticks(
            ticks=xticks, labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = False
    else:
        raise ValueError(f"Style {style} not defined.")

    closefig(out=rootdir / ("acc" + ablation + ".pdf"), transparent=transparent)


def get_rows_per_replication(df: pd.DataFrame) -> int:
    return df.epsilon.unique().size * df.n.unique().size * df.delta.unique().size


def plot_individual_fpr(
    df: pd.DataFrame, rootdir: Path, ablation: str, hue: str = "n", n_replications: int = 10,
):
    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis()
    rows_per_replication = get_rows_per_replication(df)
    for i in range(1, n_replications + 1):

        g = sns.relplot(
            x="epsilon",
            y="fpr",
            hue=hue,
            kind="line",
            palette=palette,
            data=df[rows_per_replication * (i - 1) : rows_per_replication * i],
            hue_order=hue_order,
            legend="brief",
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
        plt.savefig(rootdir / str(i) / ("fpr" + ablation + ".pdf"))
        closefig()


def plot_largest_fpr(df: pd.DataFrame, rootdir: Path, ablation: str, n):
    xticks, xlabels = make_xaxis()
    df = df[df.n == n]
    plt.figure(figsize=(10, 10))

    g = sns.relplot(x="epsilon", y="fpr", kind="line", data=df, ci=80, legend="brief", aspect=2,)

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("False Postive Rate")
    plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
    plt.xticks(
        ticks=xticks, labels=xlabels,
    )
    plt.savefig(rootdir / ("fpr.largest" + ablation + ".pdf"))
    closefig()


def plot_tp(df: pd.DataFrame, rootdir: Path, ablation: str, hue: str = "n"):
    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis()
    g = sns.relplot(
        x="epsilon",
        y="tpf",
        hue=hue,
        kind="line",
        palette=palette,
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
    plt.xticks(
        ticks=xticks, labels=xlabels,
    )
    plt.ylim((0, 1.01))
    plt.savefig(rootdir / ("tp" + ablation + ".pdf"))


def plot_individual_tp(
    df: pd.DataFrame, rootdir: Path, ablation: str, hue: str = "n", n_replications: int = 10,
):
    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis()
    rows_per_replication = get_rows_per_replication(df)
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
        plt.xticks(
            ticks=xticks, labels=xlabels,
        )
        plt.ylim((0, 1.01))
        plt.savefig(rootdir / str(i) / ("tp" + ablation + ".pdf"))
        closefig()


def fill_na(df):
    df.fpr.fillna(1.0, inplace=True)
    df.fnr.fillna(0.0, inplace=True)


def setup_plt(font_size: int, use_dark_background: bool) -> None:
    plt.rc("text", usetex=True)
    plt.rcParams.update({"font.size": font_size})
    if use_dark_background:
        plt.style.use("dark_background")


def assert_style(style: str) -> Style:
    style = style.upper()
    assert style in ("ICML", "NEURIPS", "POSTER")
    return cast(Style, style)


def gt(
    outdir: Path,
    style: Union[str, Style],
    confusion_path: Path,
    ablation: str = ".skip_noise",
    font_size: int = 33,
    use_dark_background: bool = False,
) -> None:
    setup_plt(font_size, use_dark_background)

    outdir = Path(outdir)
    confusion_path = Path(outdir)
    style = assert_style(style)

    confusion = read_confusion(dir=confusion_path, ablation=ablation)

    plot_fpr(confusion, outdir, ablation, style, hue="n")
    plot_fnr(confusion, outdir, ablation, style, hue="n")
    plot_accuracy(confusion, outdir, ablation, hue="n", style=style)

    print("Best accuracy:")
    print(confusion[confusion.acc == confusion.acc.max()])


def human(
    label_path: Path,
    prediction_path: Path,
    outdir: Path,
    style: Style,
    ablation: str = ".skip_noise",
    font_size: int = 33,
    use_dark_background: bool = False,
) -> None:
    setup_plt(font_size, use_dark_background)

    confusion = make_human_confusion(label_path=label_path, prediction_path=prediction_path,)
    plot_fpr(confusion, outdir, ablation, style, hue="n")
    plot_fnr(confusion, outdir, ablation, style, hue="n")
    plot_accuracy(confusion, outdir, ablation, hue="n", style=style)

    print("Best accuracy:")
    print(confusion[confusion.acc == confusion.acc.max()])


if __name__ == "__main__":
    fire.Fire()
