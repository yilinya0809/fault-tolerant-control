import fym
import matplotlib.pyplot as plt

data = fym.load("data.h5")
plt.figure()
plt.ylabel("position")
plt.xlabel("time")
plt.grid("True",'both')
plt.plot(data["t"], data["pos"].squeeze())
plt.figure()
plt.xlabel("time")
plt.ylabel("velocity")
plt.grid("True",'both')
plt.plot(data["t"], data["vel"].squeeze())

plt.show()
