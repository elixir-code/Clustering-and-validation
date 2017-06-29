# Fast inner loop for DBSCAN.
# Author: Lars Buitinck
# License: 3-clause BSD

# distutils: language = c++

cimport cython
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
import h5py

# Work around Cython bug: C++ exceptions are not caught unless thrown within
# a cdef function with an "except +" declaration.
cdef inline void push(vector[np.npy_intp] &stack, np.npy_intp i) except +:
    stack.push_back(i)

#related to Index check : check if index in bounds
@cython.boundscheck(False)
#related to python negative indexing : check if python indexing -1,... used
@cython.wraparound(False)

def dbscan_inner(np.ndarray[np.uint8_t, ndim=1, mode='c'] is_core,
                 neighborhoods,
                 np.ndarray[np.npy_intp, ndim=1, mode='c'] labels):
    cdef np.npy_intp i, label_num = 0, v, cnt = 0
    cdef np.ndarray[np.npy_intp, ndim=1] neighb
    cdef vector[np.npy_intp] stack

    for i in range(labels.shape[0]):
        
        #if already label assigned or is not a core point
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # This is very similar to the classic algorithm for computing connected
        # components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        while True:
            if labels[i] == -1:
                labels[i] = label_num

                print(cnt," stack size : ",stack.size())
                cnt += 1
                
                if is_core[i]:
                    neighb = neighborhoods[i]
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        if labels[v] == -1:
                            push(stack, v)

            if stack.size() == 0:
                break
            i = stack.back()
            stack.pop_back()

        label_num += 1
