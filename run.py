from pathlib import Path

import fire  # type: ignore

import demos


def run(
    task: str,
    criterion: str,
    query_type: str,
    M: int,
    epsilon: float = 0.0,
    delta: float = 1.1,
    outdir: str = "questions",
    overwrite: bool = False,
):
    assert delta > 1.0, "Delta must be strcitly greater than 1"
    task = task.lower()
    criterion = criterion.lower()
    query_type = query_type.lower()
    outpath = Path(outdir)
    assert (
        criterion == "information" or criterion == "volume" or criterion == "random"
    ), ("There is no criterion called " + criterion)

    demos.nonbatch(
        task=task,
        criterion=criterion,
        query_type=query_type,
        epsilon=epsilon,
        M=M,
        delta=delta,
        outdir=outpath,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    fire.Fire(run)
