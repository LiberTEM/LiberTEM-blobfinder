import numpy as np
import scipy.ndimage
import pytest

from libertem_blobfinder import base
import libertem_blobfinder.base.correlation  # noqa F401

from utils import _mk_random


@pytest.mark.with_numba
def test_refinement():
    data = np.array([
        (0, 0, 0, 0, 0, 1),
        (0, 1, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (2, 3, 0, 0, 0, 0),
        (0, 2, 0, 0, 0, -10)
    ])

    assert np.allclose(
        base.correlation.refine_center(center=(1, 1), r=1, corrmap=data), (1, 1)
    )
    assert np.allclose(
        base.correlation.refine_center(center=(2, 2), r=1, corrmap=data), (1, 1)
    )
    assert np.allclose(
        base.correlation.refine_center(center=(1, 4), r=1, corrmap=data), (0.5, 4.5)
    )

    y, x = (4, 1)
    ry, rx = base.correlation.refine_center(center=(y, x), r=1, corrmap=data)
    assert (ry > y) and (ry < (y + 1))
    assert (rx < x) and (rx > (x - 1))

    y, x = (4, 4)
    ry, rx = base.correlation.refine_center(center=(y, x), r=1, corrmap=data)
    assert (ry < y) and (ry > (y - 1))
    assert (rx < x) and (rx > (x - 1))


@pytest.mark.with_numba
def test_crop_disks_from_frame():
    crop_size = 2
    peaks = [
        [0, 0],
        [2, 2],
        [5, 5],
    ]
    frame = _mk_random(size=(6, 6), dtype="float32")
    crop_buf = np.zeros((len(peaks), 2*crop_size, 2*crop_size))
    base.correlation.crop_disks_from_frame(
        peaks=np.array(peaks),
        frame=frame,
        crop_size=crop_size,
        out_crop_bufs=crop_buf
    )

    #
    # how is the region around the peak cropped? like this (x denotes the peak position),
    # this is an example for radius 2, padding 0 -> crop_size = 4
    #
    # ---------
    # | | | | |
    # |-|-|-|-|
    # | | | | |
    # |-|-|-|-|
    # | | |x| |
    # |-|-|-|-|
    # | | | | |
    # ---------

    # first peak: top-leftmost; only the bottom right part of the crop_buf should be filled:
    assert np.all(crop_buf[0] == [
        [0, 0,           0,          0],
        [0, 0,           0,          0],
        [0, 0, frame[0, 0], frame[0, 1]],
        [0, 0, frame[1, 0], frame[1, 1]],
    ])

    # second peak: the whole crop area fits into the frame -> use full crop_buf
    assert np.all(crop_buf[1] == frame[0:4, 0:4])

    # third peak: bottom-rightmost; almost-symmetric to first case
    print(crop_buf[2])
    assert np.all(crop_buf[2] == [
        [frame[3, 3], frame[3, 4], frame[3, 5], 0],
        [frame[4, 3], frame[4, 4], frame[4, 5], 0],
        [frame[5, 3], frame[5, 4], frame[5, 5], 0],
        [          0,           0,           0, 0],  # noqa: E201
    ])


@pytest.mark.with_numba
def test_com():
    data = np.random.random((7, 9))
    ref = scipy.ndimage.measurements.center_of_mass(data)
    com = base.correlation.center_of_mass(data)
    print(ref, com, np.array(ref) - np.array(com))
    assert np.allclose(ref, com)
