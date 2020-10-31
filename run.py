from pathlib import Path

import fire  # type: ignore

import demos


def run(
    task: str,
    criterion: str,
    query_type: str,
    epsilon: float,
    M: int,
    outdir: str = "questions",
    overwrite: bool = False,
):
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
        outdir=outpath,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    fire.Fire(run)
