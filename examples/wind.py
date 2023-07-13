import numpy as np
import matplotlib.pyplot as plt

def dtrb_wind(t):
    dtrb_wind = (2 * np.sin(np.pi * t + 7) + 3 * np.sin(0.7*np.pi * t + 1) + 1 + np.sin(0.2*np.pi*t + 3))
    return dtrb_wind

def plot():
    dt = 0.01
    time = np.arange(0, 20, dt)
    d_w = np.zeros((np.size(time), 1))
    for i in np.arange(np.size(time)):
        t = time[i]
        d_w[i] = dtrb_wind(t)

    plt.plot(time, d_w)
    plt.xlabel("time, sec")
    plt.ylabel("Wind disturbance, Nm")
    plt.xlim([time[0], time[-1]])
    plt.show()

if __name__== "__main__":
    plot()



