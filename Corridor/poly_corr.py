import fym
import matplotlib.pyplot as plt
import numpy as np
import scipy
import statsmodels.api as sm
from fym.utils.rot import angle2quat, quat2angle, quat2dcm
from mpl_toolkits.mplot3d import axes3d
from numpy import cos, sin
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from statsmodels.regression.quantile_regression import QuantReg

# # Trst_corr = np.load("corr_conti.npz")
# # Trst_corr = np.load("corr_2d.npz")
# VT_corr = Trst_corr["VT_corr"]
# # acc_corr = Trst_corr["acc_corr"]
# theta_corr = np.rad2deg(Trst_corr["theta_corr"])
# cost = Trst_corr["cost"]
# success = Trst_corr["success"]

def boundary(Trst_corr):
    VT_corr = Trst_corr["VT_corr"]
    # acc_corr = Trst_corr["acc_corr"]
    theta_corr = np.rad2deg(Trst_corr["theta_corr"])
    cost = Trst_corr["cost"]
    success = Trst_corr["success"]

    upper_bound = []
    lower_bound = []
        # theta_at_V = theta_corr[i, :]
        # upper_bound.append(np.max(theta_at_V))
        # lower_bound.append(np.min(theta_at_V))


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


# upper_margin = 1.0
# lower_margin = 1.0

# upper_bound_conservative = upper_bound - upper_margin
# lower_bound_conservative = lower_bound + lower_margin


def poly(degree, Trst_corr, upper_bound, lower_bound):
    VT_corr = Trst_corr["VT_corr"]
    theta_corr = np.rad2deg(Trst_corr["theta_corr"])
    cost = Trst_corr["cost"]
    success = Trst_corr["success"]


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
    theta_corr = np.rad2deg(Trst_corr["theta_corr"])
    cost = Trst_corr["cost"]
    success = Trst_corr["success"]

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


# frac = 0.4
# lowess_upper = sm.nonparametric.lowess(upper_bound, VT_corr, frac=frac)
# lowess_lower = sm.nonparametric.lowess(lower_bound, VT_corr, frac=frac)

# VT_corr_smooth_upper = lowess_upper[:, 0]
# upper_bound_smooth = lowess_upper[:, 1]
# VT_corr_smooth_lower = lowess_lower[:, 0]
# lower_bound_smooth = lowess_lower[:, 1]

# central_line_smooth = (upper_bound_smooth + lower_bound_smooth) / 2

# Plot the results


def plot():
    degree = 3
    upper_bound, lower_bound = boundary(VT_corr)
    upper, lower, central = poly(degree, VT_corr, upper_bound, lower_bound)

    VT_target = VT_corr[-1]
    weighted = weighted_poly(degree, VT_corr, VT_target, upper, lower)

    plt.figure(figsize=(10, 6))
    plt.plot(
        VT_corr, upper_bound, "o", label="Upper Bound Data", color="blue", alpha=0.3
    )
    plt.plot(
        VT_corr, lower_bound, "o", label="Lower Bound Data", color="orange", alpha=0.3
    )
    plt.plot(VT_corr, upper(VT_corr), "b--", label="Upper Bound Polynomial")
    plt.plot(VT_corr, lower(VT_corr), "y--", label="Lower Bound Polynomial")
    # plt.plot(VT_corr, central(VT_corr), '-', label='Central Line')
    plt.plot(VT_corr, weighted(VT_corr), "k-", label="Weighted Line")
    plt.xlabel("VT, m/s", fontsize=15)
    plt.ylabel("θ, deg", fontsize=15)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot()
