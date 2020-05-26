from pathlib import Path

import numpy as np

from post import filter_halfplanes
from run_tests import run_test, run_tests


def main():
    reward = np.load(Path("preferences/reward.npy"))
    psi = np.load(Path("preferences/psi.npy"))
    s = np.load(Path("preferences/s.npy"))

    print("Doing filtering with LP")
    filtered_psi, filtered_s, _ = filter_halfplanes(psi, s, 100, skip_lp=False)
    print(f"There are {len(filtered_s)} halfplanes after filtering.")
    frac_pass = run_test(reward, filtered_psi, filtered_s)
    print(
        f"With {len(filtered_s)} questions, "
        f"{frac_pass * 100}% of the fake rewards passed the test."
    )

    print("Doing filtering without LP")
    filtered_psi, filtered_s, _ = filter_halfplanes(psi, s, 100, skip_lp=True)
    print(f"There are {len(filtered_s)} halfplanes after filtering.")
    frac_pass = run_test(reward, filtered_psi, filtered_s)
    print(
        f"With {len(filtered_s)} questions, "
        f"{frac_pass * 100}% of the fake rewards passed the test."
    )


if __name__ == "__main__":
    main()
