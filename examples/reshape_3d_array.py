import time
import numpy as np

array = np.random.randint(0, 4, ((128, 128, 128)), dtype='uint8')
scale_factor = (4, 4, 4)
bincount = 4

def prev_func(array):
    # Reshape to free dimension of size scale_factor to apply scaledown method to
    m, n, r = np.array(array.shape) // scale_factor
    arr = array.reshape((m, scale_factor[0], n, scale_factor[1], r, scale_factor[2]))
    arr = np.swapaxes(arr, 1, 2).swapaxes(2, 4)
    arr = arr.reshape((m, n, r, np.prod(scale_factor)))
    # Obtain the element that occurred the most
    arr = np.apply_along_axis(lambda x: max(set(x), key=lambda y: list(x).count(y)),
                              axis=3, arr=arr)
    return arr

def new_func(array):
    # Reshape to free dimension of size scale_factor to apply scaledown method to
    m, n, r = np.array(array.shape) // scale_factor
    arr = array.reshape((m, scale_factor[0], n, scale_factor[1], r, scale_factor[2]))
    arr = np.swapaxes(arr, 1, 2).swapaxes(2, 4)
    arr = arr.reshape((m, n, r, np.prod(scale_factor)))
    # Collapse dimensions
    arr = arr.reshape(-1,np.prod(scale_factor))
    # Get blockwise frequencies -> Get most frequent items
    arr = np.array([(arr==b).sum(axis=1) for b in range(bincount)]).argmax(axis=0)
    arr = arr.reshape((m,n,r))
    return arr

N = 10

start1 = time.time()
for i in range(N):
    out1 = prev_func(array)
end1 = time.time()
print('Prev:',(end1-start1)/N)

start2 = time.time()
for i in range(N):
    out2 = new_func(array)
end2 = time.time()
print('New:',(end2-start2)/N)

print('Difference:',(out1-out2).sum())