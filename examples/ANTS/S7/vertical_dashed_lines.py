import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(x, y1)
ax1.axvline(x=3, color='red', linestyle='--', label='x=3')
ax1.legend()

ax2.plot(x, y2)
ax2.axvline(x=3, color='blue', linestyle='--', label='x=3')
ax2.legend()

plt.tight_layout()
plt.show()