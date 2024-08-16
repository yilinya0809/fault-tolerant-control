import numpy as np
from scipy.optimize import curve_fit


def boundary(Trst_corr):
    VT_corr = Trst_corr["VT_corr"]
    theta_corr = np.rad2deg(Trst_corr["theta_corr"])
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


def log_func(x, a, b, c):
    return a * np.log(b * x) + c


def curve_fitting(func, x, y):
    popt, pcov = curve_fit(func, x, y)
    return popt
