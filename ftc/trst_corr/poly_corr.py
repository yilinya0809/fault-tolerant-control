import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

Trst_corr = np.load("ftc/trst_corr/corr_safe.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc"]
theta_corr = Trst_corr["theta_corr"]
cost = Trst_corr["cost"]
success = Trst_corr["success"]


def boundary(Trst_corr):
    VT_corr = Trst_corr["VT_corr"]
    theta_corr = Trst_corr["theta_corr"]
    success = Trst_corr["success"]

    upper_bound = []
    lower_bound = []

    for i in range(len(VT_corr)):
        theta_candidate = []
        for j in range(len(theta_corr)):
            if success[i][j] == 1:
                theta_candidate.append(theta_corr[j])

        upper_bound.append(np.max(theta_candidate))
        lower_bound.append(np.min(theta_candidate))
    upper_bound = np.array(upper_bound)
    lower_bound = np.array(lower_bound)

    return upper_bound, lower_bound


def poly(degree, Trst_corr, upper_bound, lower_bound):
    VT_corr = Trst_corr["VT_corr"]

    poly_upper = np.polyfit(VT_corr, upper_bound, degree)
    poly_lower = np.polyfit(VT_corr, lower_bound, degree)

    poly_upper_func = np.poly1d(poly_upper)
    poly_lower_func = np.poly1d(poly_lower)

    central_line_points = (poly_upper_func(VT_corr) + poly_lower_func(VT_corr)) / 2

    poly_central = np.polyfit(VT_corr, central_line_points, degree)
    poly_central_func = np.poly1d(poly_central)

    return poly_upper_func, poly_lower_func, poly_central_func


def weighted_poly(degree, Trst_corr, VT_cruise, poly_upper, poly_lower):
    VT_corr = Trst_corr["VT_corr"]

    lamb = []
    weighted_points = []
    for i in range(len(VT_corr)):
        lamb.append(VT_corr[i] / VT_cruise)
        weighted_points.append(
            lamb[i] * poly_upper(VT_corr[i]) + (1 - lamb[i]) * poly_lower(VT_corr[i])
        )

    poly_weighted = np.polyfit(VT_corr, weighted_points, degree)
    poly_weighted_func = np.poly1d(poly_weighted)
    return poly_weighted_func


def exp_fitting(x, data):
    popt, pcov = curve_fit(lambda t, a, b, c: -a * np.exp(-b * t) + c, x, data)
    a, b, c = popt[0], popt[1], popt[2]
    # y = -a * np.exp(-b * VT_filtered) + c
    return a, b, c


if __name__ == "__main__":
    upper_bound, lower_bound = boundary(Trst_corr)

    mask = lower_bound > np.min(lower_bound)
    VT_filtered = VT_corr[mask]
    lower_bound_filtered = lower_bound[mask]

    popt, pcov = curve_fit(
        lambda x, a, b, c: -a * np.exp(-b * x) + c, VT_filtered, lower_bound_filtered
    )
    a, b, c = popt[0], popt[1], popt[2]
    y = -a * np.exp(-b * VT_filtered) + c

    degree1 = 3
    degree2 = 5
    upper1 = np.polyfit(VT_corr, upper_bound, degree1)
    upper2 = np.polyfit(VT_corr, upper_bound, degree2)
    upper_func1 = np.poly1d(upper1)
    upper_func2 = np.poly1d(upper2)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(VT_corr, lower_bound, label="lower bound")
    ax.plot(VT_filtered, y, "k", label="fitted curve")
    ax.scatter(VT_corr, upper_bound)
    ax.plot(VT_corr, upper_func1(VT_corr), "b", label="5th order")
    # ax.plot(VT_corr, upper_func2(VT_corr), "g", label="6th order")
    # ax.set_aspect("equal")
    ax.legend()
    plt.show()
