import matplotlib.pyplot as plt
import numpy as np

Trst_corr = np.load("ftc/trst_corr/corr_back.npz")
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


def boundary2(Trst_corr):
    VT_corr = Trst_corr["VT_corr"]
    acc_corr = Trst_corr["acc"]
    theta_corr = Trst_corr["theta_corr"]
    success = Trst_corr["success"]

    upper_bound = []
    lower_bound = []

    for i in range(len(VT_corr)):
        theta_candidate = []
        for j in range(len(theta_corr)):
            if success[i][j] == 1:
                if acc_corr[i][j] > -3.3:
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


if __name__ == "__main__":
    upper_bound, lower_bound = boundary2(Trst_corr)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    VT, theta = np.meshgrid(VT_corr, np.rad2deg(theta_corr))
    contour = ax.contourf(
        VT, theta, acc_corr.T, levels=np.shape(theta_corr)[0], cmap="viridis", alpha=1.0
    )
    cbar = fig.colorbar(contour)
    cbar.ax.set_xlabel(r"$a_x^I,\, \mathrm{m/s^{2}}$", fontsize=20, labelpad=15)
    # fig, ax = plt.subplots(1, 1)
    # VT, theta = np.meshgrid(VT_corr, np.rad2deg(theta_corr))
    # ax.scatter(VT, theta, s=success.T, c="k")
    ax.scatter(VT_corr, np.rad2deg(lower_bound))
    ax.scatter(VT_corr, np.rad2deg(upper_bound))
    plt.show()
