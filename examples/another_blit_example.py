import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots()

ax.set_axis_off()
z = np.random.randint(0, 100, size = (500, 500))
line = ax.imshow(z, cmap="jet", extent=[0,100,0,100], interpolation="none", animated=True)

my_size = np.zeros((8, 4095)).shape

div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
cbar = fig.colorbar(line, cax=cax)

plt.show(block=False)
plt.pause(0.05)

bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(line)

fig.canvas.blit(fig.bbox)

i = 0

while(True):

    if i % 2 == 0:
        z = np.random.randint(0, 500, size = my_size)
        line.set_clim(0, 500)

    else:
        z = np.random.randint(0, 200, size = my_size)
        line.set_clim(0, 200)

    fig.canvas.restore_region(bg)
    line.set_data(z)
    ax.draw_artist(line)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()

    i += 1

    time.sleep(0.01)