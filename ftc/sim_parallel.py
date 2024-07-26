import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fym
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def sim_parallel(sim, N, initials, Env, faults=None, workers=None):
    cpu_workers = os.cpu_count()
    workers = int(workers or cpu_workers)
    assert workers <= os.cpu_count(), f"workers should be less than {cpu_workers}"
    print(f"Sample with {workers} workers ...")
    with ProcessPoolExecutor(workers) as p:
        if faults is not None:
            list(
                tqdm.tqdm(
                    p.map(sim, range(N), initials, itertools.repeat(Env), faults),
                    total=N,
                )
            )
        else:
            list(
                tqdm.tqdm(
                    p.map(sim, range(N), initials, itertools.repeat(Env)),
                    total=N,
                )
            )


def get_errors(data, time_from=5):
    time_index = data["env"]["t"] > max(data["env"]["t"]) - time_from
    pos_errors = (
        data["env"]["posd"][time_index, :, 0]
        - data["env"]["plant"]["pos"][time_index, :, 0]
    ).squeeze()
    ang_errors = (
        data["env"]["angd"][time_index, :, 0] - data["env"]["ang"][time_index, :, 0]
    ).squeeze()
    errors = np.hstack((pos_errors, ang_errors))
    return errors


def calculate_mse(data, time_from=5, weight=np.ones(6)):
    errors = get_errors(data, time_from)
    if bool(list(errors)):
        mse = np.mean(np.abs(errors), axis=0)
    else:
        mse = []
    return mse @ np.diag(weight) @ mse


def calculate_recovery_rate(errors, threshold=0.5):
    assert threshold > 0
    if bool(list(errors)):
        recovery_rate = np.mean(np.abs(errors) <= threshold)
    else:
        recovery_rate = 0
    return recovery_rate


def evaluate_recovery_rate(
    N, time_from=5, threshold=0.5, weight=np.array([0, 0, 1, 0, 0, 0]), dirpath="data"
):
    mses = []
    for i in range(N):
        data = fym.load(Path(dirpath, f"env_{i:04d}.h5"))
        mse = calculate_mse(data, time_from=time_from, weight=weight)
        mses = np.append(mses, mse)
    recovery_rate = calculate_recovery_rate(mses, threshold=threshold)
    print(f"Recovery rate is {recovery_rate:.3f}.")


def evaluate_mfa(mfa, wmse, threshold=1, verbose=False):
    """
    Is the mission feasibility assessment success?
    """
    eval = np.all(wmse <= threshold)
    if verbose:
        print(f"MSE of position trajectory is {wmse}.")
        if mfa == eval:
            print(f"MFA Success: MFA={mfa}, evaluation={eval}")
        else:
            print(f"MFA Fails: MFA={mfa}, evaluation={eval}")
    return mfa == eval


def evaluate_mfa_success_rate(
    N,
    time_from=5,
    threshold=1,
    weight=np.ones(6),
    dirpath="data",
    verbose=False,
    is_plot=False,
):
    evals = []
    wmses = []
    fidxs = []
    fcnts = []
    for i in range(N):
        data = fym.load(Path(dirpath, f"env_{i:04d}.h5"))
        mfa = np.all(data["env"]["mfa"])
        wmse = calculate_mse(data, time_from=time_from, weight=weight)
        evals = np.append(
            evals, evaluate_mfa(mfa, wmse, threshold=threshold, verbose=verbose)
        )
        if is_plot:
            fcnts = np.append(fcnts, max(data["env"]["fault_count"]))
            fidxs = np.append(fidxs, np.where(data["env"]["Lambda"][-1, :] != 1)[0])
            wmses = np.append(wmses, wmse)
    mfa_success_rate = np.mean(evals)
    print(f"MFA success rate is {mfa_success_rate:.3f}.")

    if is_plot:
        maxe = min(max(wmses), 1)
        for i in range(N):
            if fcnts[i] == 1:
                plt.scatter(fidxs[i] + np.random.rand() + 0.5, wmses[i] / maxe)
        plt.axvline(1.5, color="gray", linestyle="--", linewidth=1.2)
        plt.axvline(2.5, color="gray", linestyle="--", linewidth=1.2)
        plt.axvline(3.5, color="gray", linestyle="--", linewidth=1.2)
        plt.axvline(4.5, color="gray", linestyle="--", linewidth=1.2)
        plt.axvline(5.5, color="gray", linestyle="--", linewidth=1.2)
        plt.axhline(threshold / maxe, color="red", linestyle="--", label="threshold")
        plt.xlim([0.4, 6.6])
        plt.ylim([-0.1, 1.1])
        plt.grid(alpha=0.5, linestyle="--")
        plt.legend(loc="upper right")
        plt.xlabel("Fault Index")
        plt.ylabel("Normalized Mean Squared Error")
        plt.show()
        plt.savefig("eval_mfa.png", transparent=True, dpi=300)
