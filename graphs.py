import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import fire  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from typing_extensions import Literal  # type: ignore

from run_tests import Experiment
from utils import setup_logging

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


def make_xaxis(
    n_labels: int = 5, ticks_per_label: int = 5, lower: float = 0.0, upper: float = 1.5
) -> Tuple[np.ndarray, List[str]]:
    n_ticks = (n_labels - 1) * ticks_per_label + 1
    xticks = np.linspace(lower, upper, n_ticks)
    xlabels = [""] * n_ticks
    for i, val in enumerate(xticks[::ticks_per_label]):
        xlabels[i * ticks_per_label] = str(val)
    return xticks, xlabels


def make_palette_map(key_values: np.ndarray) -> dict:
    """Given a sequence of experimental parameters, generate palette maps for representing
    parameter values."""
    logging.debug(f"key_values shape={key_values.shape}")
    logging.debug(f"key_values={key_values}")
    palette = sns.color_palette("muted", len(key_values))
    palette_map = {val: palette[j] for j, val in enumerate(sorted(key_values))}

    return palette_map


def get_hue(hue: str, df: pd.DataFrame):
    palette = make_palette_map(df[hue].drop_duplicates().to_numpy())
    if hue == "n":
        assert "n" in df.columns
        hue_order = np.sort(df.n.astype(int).unique()).astype(str)
    elif hue == "delta":
        hue_order = np.sort(df.delta.astype(float).unique()).astype(str)
    else:
        raise ValueError("Hue must be n or delta")

    return palette, hue_order


def check(
    normals: np.ndarray,
    indices: Dict[Experiment, np.ndarray],
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

        aligned_rewards: np.ndarray = rewards[aligned]
        misaligned_rewards = rewards[np.logical_not(aligned)]

        for agreement in cast(np.ndarray, np.mean((aligned_rewards @ validation_test) > 0, axis=1)):
            agreements.loc[j] = [epsilon, delta, n, True, agreement]
            j += 1

        for agreement in cast(
            np.ndarray, np.mean((misaligned_rewards @ validation_test) > 0, axis=1)
        ):
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
    return agreements


def make_human_confusion(
    label_path: Path = Path("questions/gt_rewards/alignment.npy"),
    prediction_path: Path = Path("questions/test_results.skip_noise.pkl"),
) -> pd.DataFrame:
    label = np.load(label_path)
    predictions: Dict[Experiment, np.ndarray] = pickle.load(open(prediction_path, "rb"))

    confusions = []
    for experiment, prediction in predictions.items():
        n = experiment[2]
        if n <= 0:
            continue
        confusion = confusion_matrix(y_true=label, y_pred=prediction, labels=[False, True])
        confusions.append(
            (
                *experiment,
                confusion[0][0],
                confusion[0][1],
                confusion[1][0],
                confusion[1][1],
            )
        )

    df = pd.DataFrame(
        confusions,
        columns=["epsilon", "delta", "n", "tn", "fp", "fn", "tp"],
    )

    df = df.convert_dtypes()

    df = df.sort_values(by="n")
    df["n"] = df["n"].astype(str)

    df = compute_targets(df)

    return df


def read_confusion(dir: Path, ablation: str = "", max_n: int = -1) -> pd.DataFrame:
    """ Read dict of confusion matrices. """
    out_dict = pickle.load(open(dir / f"confusion.{ablation}.pkl", "rb"))

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

    if max_n > 0:
        out = out[out.n.astype(int) <= max_n]

    out = compute_targets(out)

    logging.debug(f"Reading inputs with shape={out.shape}, columns={out.columns}")

    return out


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("fp", "tp", "fn", "tn"):
        df[col] = df[col].astype(int)

    for col in ("epsilon", "delta"):
        df[col] = df[col].astype(float)

    df["n"] = df["n"].astype(str)
    df["fpr"] = df.fp / (df.fp + df.tn)
    df["tpf"] = df.tp / (df.tp + df.fp + df.tn)
    df["fnr"] = df.fn / (df.fn + df.tp)
    df["acc"] = (df.tp + df.tn) / (df.tp + df.tn + df.fp + df.fn)
    return df


def read_replications(
    rootdir: Path, ablation: str, replications: Optional[int] = None, max_n: int = -1
) -> pd.DataFrame:
    df = pd.DataFrame(columns=["epsilon", "delta", "n", "tn", "fp", "fn", "tp"])
    if replications is not None:
        for replication in range(1, int(replications) + 1):
            if (rootdir / str(replication)).exists():
                df = df.append(
                    read_confusion(rootdir / str(replication), ablation=ablation, max_n=max_n)
                )
            else:
                logging.warning(f"rootdir={rootdir / str(replication)} does not exist")
    else:
        df = read_confusion(rootdir, ablation=ablation)

    df = df.convert_dtypes()

    # Seaborn tries to convert integer hues into rgb values. So we make them strings.
    df = compute_targets(df)

    return df


def get_max_delta(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()

    df["means"] = (
        df[["n", "epsilon", "delta", target]].groupby(["n", "epsilon", "delta"]).transform("mean")
    ).astype(float)
    df["max_mean_delta"] = df.groupby(["n", "epsilon"]).means.transform("max")

    df = df[df.means == df.max_mean_delta]
    df = df.drop(columns=["means", "max_mean_delta"])

    return df


def get_rows_per_replication(df: pd.DataFrame) -> int:
    return df.epsilon.unique().size * df.n.unique().size * df.delta.unique().size


def fill_na(df) -> None:
    df.fpr.fillna(1.0, inplace=True)
    df.fnr.fillna(0.0, inplace=True)


def setup_plt(font_size: int, use_dark_background: bool) -> None:
    plt.rc("text", usetex=True)
    plt.rcParams.update({"font.size": font_size})
    if use_dark_background:
        plt.style.use("dark_background")

    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

    logging.getLogger("PIL").setLevel("WARNING")


def assert_style(style: str) -> Style:
    style = style.upper()
    assert style in ("ICML", "NEURIPS", "POSTER")
    return cast(Style, style)


### Plots


def plot_agreements(
    agreements: pd.DataFrame,
    epsilon: float,
    delta: float,
    n: int,
    out: Optional[Path] = None,
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
        mean_agreement[np.logical_not(mean_agreement.aligned)].value,
        label="unaligned",
        alpha=0.3,
    )

    plt.xlabel("\% holdout agreement")
    plt.legend()
    closefig(out)


def plot_fpr(
    df: pd.DataFrame,
    rootdir: Path,
    ablation: str,
    style: Style,
    hue: str = "n",
    best_delta: bool = True,
    n_labels: int = 5,
    ticks_per_label: int = 5,
) -> None:
    plt.figure(figsize=(10, 10))

    df = df.dropna(subset=["fpr"])
    if best_delta:
        df = get_max_delta(df, "fpr")

    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis(
        lower=df.epsilon.min(),
        upper=df.epsilon.max(),
        n_labels=n_labels,
        ticks_per_label=ticks_per_label,
    )

    g = sns.relplot(
        x="epsilon",
        y="fpr",
        hue=hue,
        kind="line",
        palette=palette,
        data=df,
        ci=80,
        hue_order=hue_order,
        legend="brief",
        aspect=2,
    )

    if style == "ICML":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("False Positive Rate")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = False
    elif style == "POSTER":
        plt.xlabel("Value Slack")
        plt.ylabel("False Positive Rate")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = True
    elif style == "NEURIPS":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("False Positive Rate")
        plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = False
    else:
        raise ValueError(f"Style {style} not defined.")

    plt.savefig(rootdir / ("fpr." + ablation + ".pdf"), transparent=transparent)
    plt.savefig(rootdir / ("fpr." + ablation + ".png"), transparent=transparent)
    closefig()

    # TODO(joschnei): The way I was doing this before meant that the different axes had different
    # fonts. Using sns for everything would probalby fix that, but sns is undocumented garbage.
    # Switch to pure matplotlib if you can, and figure out how sns works if you can't.


def plot_fnr(
    df: pd.DataFrame,
    rootdir: Path,
    ablation: str,
    style: Style,
    hue: str = "n",
    best_delta: bool = True,
) -> None:
    plt.figure(figsize=(10, 10))

    df = df[np.isfinite(df.fnr)]
    if best_delta:
        df = get_max_delta(df, "fnr")

    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis(lower=df.epsilon.min(), upper=df.epsilon.max())

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

    if style == "ICML":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("False Negative Rate")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
    elif style == "POSTER":
        raise NotImplementedError()
    elif style == "NEURIPS":
        raise NotImplementedError()
    else:
        raise ValueError(f"Style {style} not defined.")

    plt.savefig(rootdir / ("fnr." + ablation + ".pdf"))
    plt.savefig(rootdir / ("fnr." + ablation + ".png"))
    closefig()


def plot_accuracy(
    df: pd.DataFrame,
    rootdir: Path,
    ablation: str,
    style: Style,
    hue: str = "n",
    best_delta: bool = True,
    n_labels: int = 5,
    ticks_per_label: int = 5,
) -> None:
    plt.figure(figsize=(10, 10))

    df = df.dropna(subset=["acc"])
    if best_delta:
        df = get_max_delta(df, "acc")

    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis(
        lower=df.epsilon.min(),
        upper=df.epsilon.max(),
        n_labels=n_labels,
        ticks_per_label=ticks_per_label,
    )

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

    if style == "POSTER":
        plt.xlabel("Value Slack")
        plt.ylabel("Accuracy")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = True
    elif style == "NEURIPS":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("Accuracy")
        plt.title(r"$\epsilon$-Relaxation's Effect on Accuracy")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = False
    elif style == "ICML":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("Accuracy")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = False
    else:
        raise ValueError(f"Style {style} not defined.")

    plt.savefig(rootdir / ("acc." + ablation + ".pdf"), transparent=transparent)
    plt.savefig(rootdir / ("acc." + ablation + ".png"), transparent=transparent)
    closefig()


def plot_delta_accuracy(
    confusion_a: pd.DataFrame,
    confusion_b: pd.DataFrame,
    outdir: Path,
    ablation: str,
    style: Style,
    hue: str = "n",
    best_delta: bool = False,
    n_labels=6,
    ticks_per_label=5,
) -> None:
    # confusion = pd.merge(confusion_a, confusion_b, how="inner", on="")
    # Clean individual dataframes
    accs = list()
    for confusion in (confusion_a, confusion_b):
        confusion = confusion.dropna(subset=["acc"])
        if best_delta:
            confusion = get_max_delta(confusion, "acc")

        accs.append(pd.DataFrame(confusion.groupby(["epsilon", "n"]).mean())[["acc"]])

    delta_acc = pd.DataFrame(accs[0] - accs[1]).reset_index()

    palette, hue_order = get_hue(hue, delta_acc)
    xticks, xlabels = make_xaxis(
        lower=delta_acc.epsilon.min(),
        upper=delta_acc.epsilon.max(),
        n_labels=n_labels,
        ticks_per_label=ticks_per_label,
    )

    sns.relplot(
        kind="line",
        x="epsilon",
        y="acc",
        hue=hue,
        data=delta_acc,
        palette=palette,
        hue_order=hue_order,
        legend="brief",
        aspect=2,
    )

    if style == "POSTER":
        plt.xlabel("Value Slack")
        plt.ylabel(r"$\Delta$Accuracy")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = True
    elif style == "NEURIPS":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$\Delta$Accuracy")
        plt.title(r"Accuracy improvement of active queries over random baseline")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = False
    elif style == "ICML":
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$\Delta$Accuracy")
        plt.xticks(
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        transparent = False
    else:
        raise ValueError(f"Style {style} not defined.")

    plt.savefig(outdir / ("delta_acc." + ablation + ".pdf"), transparent=transparent)
    plt.savefig(outdir / ("delta_acc." + ablation + ".png"), transparent=transparent)
    closefig()


def plot_individual_fpr(
    df: pd.DataFrame,
    rootdir: Path,
    ablation: str,
    hue: str = "n",
    n_replications: int = 10,
) -> None:
    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis(lower=df.epsilon.min(), upper=df.epsilon.max())
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
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        plt.savefig(rootdir / str(i) / ("fpr." + ablation + ".pdf"))
        plt.savefig(rootdir / str(i) / ("fpr." + ablation + ".png"))
        closefig()


def plot_largest_fpr(df: pd.DataFrame, rootdir: Path, ablation: str, n) -> None:
    xticks, xlabels = make_xaxis(lower=df.epsilon.min(), upper=df.epsilon.max())
    df = df[df.n == n]
    plt.figure(figsize=(10, 10))

    g = sns.relplot(
        x="epsilon",
        y="fpr",
        kind="line",
        data=df,
        ci=80,
        legend="brief",
        aspect=2,
    )

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("False Postive Rate")
    plt.title(r"$\epsilon$-Relaxation's Effect on FPR")
    plt.xticks(
        ticks=xticks,
        labels=xlabels,
    )
    plt.savefig(rootdir / ("fpr.largest." + ablation + ".pdf"))
    plt.savefig(rootdir / ("fpr.largest." + ablation + ".png"))
    closefig()


def plot_tp(df: pd.DataFrame, rootdir: Path, ablation: str, hue: str = "n") -> None:
    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis(lower=df.epsilon.min(), upper=df.epsilon.max())
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
        ticks=xticks,
        labels=xlabels,
    )
    plt.ylim((0, 1.01))
    plt.savefig(rootdir / ("tp." + ablation + ".pdf"))
    plt.savefig(rootdir / ("tp." + ablation + ".png"))


def plot_individual_tp(
    df: pd.DataFrame,
    rootdir: Path,
    ablation: str,
    hue: str = "n",
    n_replications: int = 10,
) -> None:
    palette, hue_order = get_hue(hue, df)
    xticks, xlabels = make_xaxis(lower=df.epsilon.min(), upper=df.epsilon.max())
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
            ticks=xticks,
            labels=xlabels,
        )
        plt.ylim((0, 1.01))
        plt.savefig(rootdir / str(i) / ("tp." + ablation + ".pdf"))
        plt.savefig(rootdir / str(i) / ("tp." + ablation + ".png"))
        closefig()


def comparison(
    outdir: Path,
    style: Union[str, Style],
    rootdir: Path = Path("data/simulated"),
    methods: List[str] = ["active", "random"],
    test_name: str = "comparison_point_reward_test",
    ablation: str = "skip_noise",
    replications: Optional[int] = None,
    max_n: int = -1,
    use_best_delta: bool = False,
    font_size: int = 33,
    dark: bool = False,
    skip_easy: bool = True,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    logging.basicConfig(level=verbosity)
    setup_plt(font_size, dark)
    max_n = int(max_n)
    rootdir = Path(rootdir)
    outdir = Path(outdir)

    assert len(methods) == 2, "Exactly two methods required"
    confusions = list()
    for method in methods:
        datadir = Path(rootdir) / method / test_name
        confusion = read_replications(datadir, ablation, replications, max_n)
        if skip_easy:
            confusion = confusion[confusion.tp + confusion.fn != 100]
            confusion = confusion[confusion.tn + confusion.fp != 100]
        confusions.append(confusion)

    plot_delta_accuracy(
        confusions[0],
        confusions[1],
        ablation=ablation,
        style=cast(Style, style),
        hue="n",
        best_delta=use_best_delta,
        outdir=outdir,
    )


def gt(
    outdir: Path,
    style: Union[str, Style],
    confusion_path: Path,
    ablation: str = "skip_noise",
    replications: Optional[int] = None,
    max_n: int = -1,
    font_size: int = 33,
    use_dark_background: bool = False,
    skip_easy: bool = True,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(verbosity=verbosity)
    setup_plt(font_size, use_dark_background)
    max_n = int(max_n)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    confusion_path = Path(confusion_path)
    style = assert_style(style)

    confusion = read_replications(
        rootdir=confusion_path, ablation=ablation, replications=replications, max_n=max_n
    )

    if skip_easy:
        confusion = confusion[confusion.tp + confusion.fn != 100]
        confusion = confusion[confusion.tn + confusion.fp != 100]

    best_delta = not confusion.delta.isna().all()

    plot_fpr(confusion, outdir, ablation, style, hue="n", best_delta=best_delta)
    plot_fnr(confusion, outdir, ablation, style, hue="n", best_delta=best_delta)
    plot_accuracy(confusion, outdir, ablation, hue="n", best_delta=best_delta, style=style)

    print("Best accuracy:")
    print(confusion[confusion.acc == confusion.acc.max()])


def human(
    label_path: Path,
    prediction_path: Path,
    outdir: Path,
    style: Style,
    ablation: str = "skip_noise",
    font_size: int = 33,
    use_dark_background: bool = False,
) -> None:
    outdir = Path(outdir)
    setup_plt(font_size, use_dark_background)

    confusion = make_human_confusion(
        label_path=label_path,
        prediction_path=prediction_path,
    )
    plot_fpr(confusion, outdir, ablation, style, hue="n")
    plot_fnr(confusion, outdir, ablation, style, hue="n")
    plot_accuracy(confusion, outdir, ablation, hue="n", style=style, n_labels=6, ticks_per_label=5)

    print("Best accuracy:")
    best_experiments = confusion[confusion.acc == confusion.acc.max()]
    print(best_experiments)
    print(best_experiments.epsilon.min())
    print(best_experiments.epsilon.max())


if __name__ == "__main__":
    fire.Fire({"comparison": comparison, "gt": gt, "human": human})
