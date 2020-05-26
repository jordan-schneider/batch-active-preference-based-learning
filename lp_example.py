from pathlib import Path

import numpy as np

from post import filter_halfplanes
from run_tests import run_test


def main():
    reward = np.load(Path("preferences/reward.npy"))
    normals = np.load(Path("preferences/psi.npy"))
    preferences = np.load(Path("preferences/s.npy"))

    print("Doing filtering with LP")
    filtered_normals, filtered_preferences, _ = filter_halfplanes(
        normals, preferences, 100, skip_lp=False
    )
    print(f"There are {len(filtered_preferences)} halfplanes after filtering.")
    frac_pass = run_test(reward, filtered_normals, filtered_preferences)
    print(
        f"With {len(filtered_preferences)} questions, "
        f"{frac_pass * 100}% of the fake rewards passed the test."
    )

    print("Doing filtering without LP")
    filtered_normals, filtered_preferences, _ = filter_halfplanes(
        normals, preferences, 100, skip_lp=True
    )
    print(f"There are {len(filtered_preferences)} halfplanes after filtering.")
    frac_pass = run_test(reward, filtered_normals, filtered_preferences)
    print(
        f"With {len(filtered_preferences)} questions, "
        f"{frac_pass * 100}% of the fake rewards passed the test."
    )


if __name__ == "__main__":
    main()
