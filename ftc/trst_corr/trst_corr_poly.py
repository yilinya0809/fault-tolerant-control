import matplotlib.pyplot as plt
import numpy as np

Trst_corr = np.load("ftc/trst_corr/corr.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc_corr"]
theta_corr = Trst_corr["theta_corr"]
cost = Trst_corr["cost"]
success = Trst_corr["success"]


def boundary(VT_corr):
    upper_bound = []
    lower_bound = []

    for i in range(len(VT_corr)):
        theta_at_V = theta_corr[i, :]
        upper_bound.append(np.max(theta_at_V))
        lower_bound.append(np.min(theta_at_V))

    upper_bound = np.array(upper_bound)
    lower_bound = np.array(lower_bound)
    return upper_bound, lower_bound


def poly(degree, VT_corr, upper_bound, lower_bound):
    poly_upper = np.polyfit(VT_corr, upper_bound, degree)
    poly_lower = np.polyfit(VT_corr, lower_bound, degree)

    poly_upper_func = np.poly1d(poly_upper)
    poly_lower_func = np.poly1d(poly_lower)

    central_line_points = (poly_upper_func(VT_corr) + poly_lower_func(VT_corr)) / 2

    poly_central = np.polyfit(VT_corr, central_line_points, degree)
    poly_central_func = np.poly1d(poly_central)

    return poly_upper_func, poly_lower_func, poly_central_func


def weighted_poly(degree, VT_corr, VT_target, poly_upper, poly_lower):
    lamb = []
    weighted_points = []
    for i in range(len(VT_corr)):
        lamb.append(VT_corr[i] / VT_target)
        weighted_points.append(
            lamb[i] * poly_upper(VT_corr[i]) + (1 - lamb[i]) * poly_lower(VT_corr[i])
        )

    poly_weighted = np.polyfit(VT_corr, weighted_points, degree)
    poly_weighted_func = np.poly1d(poly_weighted)
    return poly_weighted_func


def plot():
    degree = 3
    upper_bound, lower_bound = boundary(VT_corr)
    upper, lower, central = poly(degree, VT_corr, upper_bound, lower_bound)

    VT_target = VT_corr[-1]
    weighted = weighted_poly(degree, VT_corr, VT_target, upper, lower)

    plt.figure(figsize=(10, 6))
    plt.plot(VT_corr, upper_bound, "o", label="Upper Bound Data", color="tab:gray")
    plt.plot(VT_corr, lower_bound, "o", label="Lower Bound Data", color="tab:gray")
    plt.plot(VT_corr, upper(VT_corr), "b--", label="Upper Bound Polynomial")
    plt.plot(VT_corr, lower(VT_corr), "b--", label="Lower Bound Polynomial")
    # plt.plot(VT_corr, central(VT_corr), '-', label='Central Line')
    plt.plot(VT_corr, weighted(VT_corr), "k-", label="Weighted Line")
    plt.xlabel("V, m/s")
    plt.ylabel("θ, deg")
    plt.legend()
    plt.title("Dynamic Transition Corridor with Polynomial Boundaries")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot()
