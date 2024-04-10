from contextlib import contextmanager

import numpy as np
import sparse
import scipy.sparse as sp
import pytest

from libertem.common.backend import get_use_cpu, get_use_cuda, set_use_cpu, set_use_cuda
from libertem.utils.devices import detect

import libertem_blobfinder.base.masks as m
from libertem_blobfinder.base.utils import make_cartesian, make_polar, frame_peaks
from libertem_blobfinder.common.gridmatching import calc_coords


def _naive_mask_apply(masks, data):
    """
    masks: list of masks
    data: 4d array of input data

    returns array of shape (num_masks, scan_y, scan_x)
    """
    assert len(data.shape) == 4
    for mask in masks:
        assert mask.shape == data.shape[2:], "mask doesn't fit frame size"

    dtype = np.result_type(*[m.dtype for m in masks], data.dtype)
    res = np.zeros((len(masks),) + tuple(data.shape[:2]), dtype=dtype)
    for n in range(len(masks)):
        mask = to_dense(masks[n])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = data[i, j].ravel().dot(mask.ravel())
                res[n, i, j] = item
    return res


# This function introduces asymmetries so that errors won't average out so
# easily with large data sets
def _mk_random(size, dtype='float32'):
    dtype = np.dtype(dtype)
    if dtype.kind == 'c':
        choice = [0, 1, -1, 0+1j, 0-1j]
    else:
        choice = [0, 1]
    data = np.random.choice(choice, size=size).astype(dtype)
    coords2 = tuple(np.random.choice(range(c)) for c in size)
    coords10 = tuple(np.random.choice(range(c)) for c in size)
    data[coords2] = np.random.choice(choice) * sum(size)
    data[coords10] = np.random.choice(choice) * 10 * sum(size)
    return data


def _fullgrid(zero, a, b, index, skip_zero=False):
    i, j = np.mgrid[-index:index, -index:index]
    indices = np.concatenate(np.array((i, j)).T)
    if skip_zero:
        select = (np.not_equal(indices[:, 0], 0) + np.not_equal(indices[:, 1], 0))
        indices = indices[select]
    return calc_coords(zero, a, b, indices)


@contextmanager
def set_device_class(device_class):
    '''
    This context manager is designed to work with the inline executor.
    It simplifies running tests with several device classes by skipping
    unavailable device classes and handling setting and re-setting the environment variables
    correctly.
    '''
    prev_cuda_id = get_use_cuda()
    prev_cpu_id = get_use_cpu()
    try:
        if device_class in ('cupy', 'cuda'):
            d = detect()
            cudas = d['cudas']
            if not d['cudas']:
                pytest.skip(f"No CUDA device, skipping test with device class {device_class}.")
            if device_class == 'cupy' and not d['has_cupy']:
                pytest.skip(f"No CuPy, skipping test with device class {device_class}.")
            set_use_cuda(cudas[0])
        else:
            set_use_cpu(0)
        print(f'running with {device_class}')
        yield
    finally:
        if prev_cpu_id is not None:
            assert prev_cuda_id is None
            set_use_cpu(prev_cpu_id)
        elif prev_cuda_id is not None:
            assert prev_cpu_id is None
            set_use_cuda(prev_cuda_id)
        else:
            raise RuntimeError('No previous device ID, this should not happen.')


def prod(iterable):
    '''
    Safe product for large integer size calculations.

    :meth:`numpy.prod` uses 32 bit for default :code:`int` on Windows 64 bit. This
    function uses infinite width integers to calculate the product and
    throws a ValueError if it encounters types other than the supported ones.
    '''
    result = 1

    for item in iterable:
        result *= int(item)
    return result


def to_dense(a):
    if isinstance(a, sparse.SparseArray):
        return a.todense()
    elif sp.issparse(a):
        return a.toarray()
    else:
        return np.array(a)


def cbed_frame(
        fy=128, fx=128, zero=None, a=None, b=None, indices=None,
        radius=4, all_equal=False, margin=None):
    if zero is None:
        zero = (fy//2, fx//2)
    zero = np.array(zero)
    if a is None:
        a = (fy//8, 0)
    a = np.array(a)
    if b is None:
        b = make_cartesian(make_polar(a) - (0, np.pi/2))
    b = np.array(b)
    if indices is None:
        indices = np.mgrid[-10:11, -10:11]
    if margin is None:
        margin = radius
    indices, peaks = frame_peaks(fy=fy, fx=fx, zero=zero, a=a, b=b, r=margin, indices=indices)

    data = np.zeros((1, fy, fx), dtype=np.float32)

    dists = np.linalg.norm(peaks - zero, axis=-1)
    max_val = max(dists.max() + 1, len(peaks) + 1)

    for i, p in enumerate(peaks):
        data += m.circular(
            centerX=p[1],
            centerY=p[0],
            imageSizeX=fx,
            imageSizeY=fy,
            radius=radius,
            antialiased=True,
        ) * (1 if all_equal else max(1, max_val - dists[i] + i))

    return (data, indices, peaks)
