import numpy as np


def make_cartesian(polar):
    '''
    Accept list of polar vectors, return list of cartesian vectors

    Parameters
    ----------
    polars : numpy.ndarray of tuples [(r1, phi1), (r2, phi2), ...]
        Polar vectors

    Returns
    -------
    numpy.ndarray of tuples [(y, x), (y, x), ...]
    '''
    xes = np.cos(polar[..., 1]) * polar[..., 0]
    yes = np.sin(polar[..., 1]) * polar[..., 0]
    return np.array((yes.T, xes.T)).T


def make_polar(cartesian):
    '''
    Accept list of cartesian vectors, return list of polar vectors

    Parameters
    ----------
    cartesian : numpy.ndarray of tuples [(y, x), (y, x), ...]
        Cartesian vectors

    Returns
    -------

    Polar vector as numpy.ndarray of tuples [(r1, phi1), (r2, phi2), ...]
    '''
    ds = np.linalg.norm(cartesian, axis=-1)
    # (y, x)
    alphas = np.arctan2(cartesian[..., 0], cartesian[..., 1])
    return np.array((ds.T, alphas.T)).T


def regularize_indices(indices):
    s = indices.shape
    # Output of mgrid
    if (len(s) == 3) and (s[0] == 2):
        result = np.concatenate(indices.T)
    # List of (i, j) pairs
    elif (len(s) == 2) and (s[1] == 2):
        result = indices
    else:
        raise ValueError(
            "Shape of indices is %s, expected (n, 2) or (2, n, m)" % str(indices.shape))
    return result


def frame_peaks(fy, fx, zero, a, b, r, indices):
    indices = regularize_indices(indices)
    peaks = calc_coords(zero, a, b, indices)
    selector = within_frame(peaks, r, fy, fx)
    return indices[selector], peaks[selector]


def calc_coords(zero, a, b, indices):
    '''
    Calculate coordinates from lattice vectors a, b and indices
    '''
    coefficients = np.array((a, b))
    return zero + np.dot(indices, coefficients)


def within_frame(peaks, r, fy, fx):
    '''
    Return a boolean vector indicating peaks that are within (r, r) and (fy - r, fx - r)
    '''
    selector = (peaks >= (r, r)) * (peaks < (fy - r, fx - r))
    return selector.all(axis=-1)


def cbed_frame(
        fy=128, fx=128, zero=None, a=None, b=None, indices=None,
        radius=4, all_equal=False, margin=None):
    from libertem_blobfinder.base.masks import circular  # otherwise circular import

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
        data += circular(
            centerX=p[1],
            centerY=p[0],
            imageSizeX=fx,
            imageSizeY=fy,
            radius=radius,
            antialiased=True,
        ) * (1 if all_equal else max(1, max_val - dists[i] + i))

    return (data, indices, peaks)
