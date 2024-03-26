from contextlib import contextmanager

import numpy as np
import pytest

from libertem.common.backend import get_use_cpu, get_use_cuda, set_use_cpu, set_use_cuda
from libertem.utils.devices import detect
from libertem.masks import to_dense
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
