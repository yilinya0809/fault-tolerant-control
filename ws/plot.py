import fym
import matplotlib.pyplot as plt

data = fym.load("data.h5")
plt.plot(data["t"], data["pos"].squeeze())
plt.figure()
plt.plot(data["t"], data["vel"].squeeze())

plt.show()
