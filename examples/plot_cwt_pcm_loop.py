import time
from matplotlib import pyplot as plt
import numpy as np

"""
Example from guy who works at stack over flow
FROM: https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot/40139416#40139416

"""

def live_update_demo(blit = False):
    """
    Initialize the Plot
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    """
    The Data
    """
    x1 = np.linspace(0, 50., num=100)
    X1,Y1 = np.meshgrid(x1, x1)
    img1 = ax1.imshow(X1, vmin=-1, vmax=1, interpolation="None", cmap="RdBu")
    img1_shape = img1.get_shape()

    x2 = np.linspace(0, 4096., num = 4096)
    y2 = np.linspace(0, 116., num = 116)
    empty_coefs = np.zeros((x2.size-1, y2.size-1))

    pcm = ax2.pcolormesh(y2, x2, empty_coefs)
    pcm.set_animated(True)

    """
    Initialize Blit
    """
    fig.canvas.draw()   # note that the first draw comes before setting data 

    if blit:
        # cache the background
        axbackground = fig.canvas.copy_from_bbox(ax1.bbox)
        ax2background = fig.canvas.copy_from_bbox(ax2.bbox)

    plt.show(block=False)

    """
    FPS Calculation
    """
    t_start = time.time()

    time.sleep(.001)

    k = 0
    i = 0

    while(True):
        data1 = np.sin(X1/3.+k)*np.cos(Y1/3.+k)
        data2 = np.sin(X2/3.+k)*np.cos(Y2/3.+k)

        img1.set_data(data1)
        pcm.set_data(data2)

        tx = 'Mean Frame Rate: {fps:.3f} FPS'.format(fps= ((i+1) / (time.time() - t_start)) ) 
        print(tx)

        k += 0.11

        if blit:
            # restore background
            fig.canvas.restore_region(axbackground)
            fig.canvas.restore_region(ax2background)

            # redraw just the points
            ax1.draw_artist(img1)
            ax2.draw_artist(img2)

            # fill in the axes rectangle
            fig.canvas.blit(ax1.bbox)
            fig.canvas.blit(ax2.bbox)

        else:
            # redraw everything
            fig.canvas.draw()

        fig.canvas.flush_events()
        #alternatively you could use
        #plt.pause(0.000000000001) 
        # however plt.pause calls canvas.draw(), as can be read here:
        #http://bastibe.de/2013-05-30-speeding-up-matplotlib.html

        # TODO Revert after testing
        i += 1
        
live_update_demo(True)   # 175 fps
#live_update_demo(False) # 28 fps