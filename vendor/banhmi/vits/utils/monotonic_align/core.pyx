# cython: language_level=3
"""Monotonic Alignment Search: the dynamic program VITS uses to find the
most likely monotonic (non-crossing, non-skipping) alignment between text
and audio frames, given a per-(phoneme, frame) log-likelihood matrix.

This is the one piece of the whole model that's genuinely worth keeping as
a compiled extension: it's an O(t_x * t_y) DP with a Python-level loop over
every (phoneme, frame) pair per batch item, which is far too slow as plain
Python/PyTorch for anything but tiny inputs.
"""
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _maximum_path_one(
    int[:, ::1] path,
    float[:, ::1] value,
    int t_y,
    int t_x,
    float max_neg_val=-1e9,
) nogil:
    """Fills `value` in place with the best cumulative log-likelihood
    reaching each (y, x) cell under a monotonic alignment, then walks the
    argmax backwards from the final cell to mark the chosen path."""
    cdef int x, y
    cdef float v_prev, v_cur
    cdef int index = t_x - 1

    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg_val if x == y else value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(
    int[:, :, ::1] paths,
    float[:, :, ::1] values,
    int[::1] t_ys,
    int[::1] t_xs,
) nogil:
    cdef int batch_size = paths.shape[0]
    cdef int i
    for i in prange(batch_size, nogil=True):
        _maximum_path_one(paths[i], values[i], t_ys[i], t_xs[i])
