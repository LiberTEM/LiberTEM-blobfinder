import os

import numpy as np
import numba


# FIXME There's work on flexible FFT backends in scipy
# https://github.com/scipy/scipy/wiki/GSoC-2019-project-ideas#revamp-scipyfftpack
# and discussions about pyfftw performance vs other implementations
# https://github.com/pyFFTW/pyFFTW/issues/264
# For that reason we shoud review the state of Python FFT implementations
# regularly and adapt our choices accordingly
try:
    import pyfftw
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
    zeros = pyfftw.zeros_aligned
except ImportError:
    fft = np.fft
    zeros = np.zeros

# Necessary to work with JIT disabled for coverage and testing purposes
# https://github.com/LiberTEM/LiberTEM/issues/539
if os.getenv('NUMBA_DISABLE_JIT'):
    def to_fixed_tuple(arr, l):
        return tuple(arr)
else:
    from numba.np.unsafe.ndarray import to_fixed_tuple


@numba.njit
def center_of_mass(arr):
    r_y = r_x = np.float32(0)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r_y += np.float32(arr[y, x]*y)
            r_x += np.float32(arr[y, x]*x)
    s = arr.sum()
    return (np.float32(r_y/s), np.float32(r_x/s))


@numba.njit
def refine_center(center, r, corrmap):
    (y, x) = center
    s = corrmap.shape
    r = min(r, y, x, s[0] - y - 1, s[1] - x - 1)
    if r <= 0:
        return (np.float32(y), np.float32(x))
    else:
        # FIXME See and compare with Extension of Phase Correlation to Subpixel Registration
        # Hassan Foroosh
        # That one or a close/similar/cited one
        cutout = corrmap[y-r:y+r+1, x-r:x+r+1]
        m = np.min(cutout)
        ry, rx = center_of_mass(cutout - m)
        refined_y = y + ry - r
        refined_x = x + rx - r
        # print(y, x, refined_y, refined_x, "\n", cutout)
        return (np.float32(refined_y), np.float32(refined_x))


@numba.njit
def peak_elevation(center, corrmap, height, r_min=1.5, r_max=np.float('inf')):
    '''
    Return the slope of the tightest cone around :code:`center` with height :code:`height`
    that touches :code:`corrmap` between :code:`r_min` and :code:`r_max`.

    The correlation of two disks -- mask and perfect diffraction spot -- has the shape of a cone.
    The function's return value correlates with the quality of a correlation. Higher slope
    means a strong peak and
    no side maxima, while weak signal or side maxima lead to a flatter slope.

    Parameters
    ----------
    center : numpy.ndarray
        (y, x) coordinates of the center within the :code:`corrmap`
    corrmap : numpy.ndarray
        Correlation map
    height : float
        The height is provided as a parameter since center can be float values from refinement
        and the height value is conveniently available from the calling function.
    r_min : float, optional
        Masks out a small local plateau around the peak that would distort and dominate
        the calculation.
    r_max : float, optional
        Mask out neighboring peaks if a large area with several legitimate peaks is
        correlated.

    Returns
    -------
    elevation : float
        Elevation of the tightest cone that fits the correlation map within the given
        parameter range.
    '''
    peak_y, peak_x = center
    (size_y, size_x) = corrmap.shape
    result = np.float32(np.inf)

    for y in range(size_y):
        for x in range(size_x):
            dist = np.sqrt((y - peak_y)**2 + (x - peak_x)**2)
            if (dist >= r_min) and (dist < r_max):
                result = min((result, np.float32((height - corrmap[y, x]) / dist)))
    return max(0, result)


def do_correlations(template, crop_parts):
    '''
    Calculate the correlation of the pre-calculated template with a stack
    of cropped peaks using fast correlation.

    Parameters
    ----------
    template : numpy.ndarray
        Real Fourier transform of the correlation pattern.
        The source pattern should have the same size as the cropped parts. Please note that
        the real Fourier transform (fft.rfft2) of the source pattern has a different shape!
    crop_parts : numpy.ndarray
        Stack of peaks cropped from the frame.

    Returns
    -------
    corrs : numpy.ndarray
        Correlation of the correlation pattern and the peaks.
    '''
    spec_parts = fft.rfft2(crop_parts)
    corrspecs = template * spec_parts
    corrs = fft.fftshift(fft.irfft2(corrspecs), axes=(-1, -2))
    return corrs


@numba.njit
def unravel_index(index, shape):
    sizes = np.zeros(len(shape), dtype=np.int64)
    result = np.zeros(len(shape), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]
    remainder = index
    for i in range(len(shape)):
        result[i] = remainder // sizes[i]
        remainder %= sizes[i]
    return to_fixed_tuple(result, len(shape))


@numba.njit
def evaluate_correlations(corrs, peaks, crop_size,
        out_centers, out_refineds, out_heights, out_elevations):
    for i in range(len(corrs)):
        corr = corrs[i]
        center = unravel_index(np.argmax(corr), corr.shape)
        refined = np.array(refine_center(center, 2, corr), dtype=np.float32)
        height = np.float32(corr[center])
        out_centers[i] = _shift(np.array(center), peaks[i], crop_size)
        out_refineds[i] = _shift(refined, peaks[i], crop_size)
        out_heights[i] = height
        out_elevations[i] = np.float32(peak_elevation(refined, corr, height))


def log_scale(data, out):
    return np.log(data - np.min(data) + 1, out=out)


def log_scale_cropbufs_inplace(crop_bufs):
    m = np.min(crop_bufs, axis=(-1, -2)) - 1
    np.log(crop_bufs - m[:, np.newaxis, np.newaxis], out=crop_bufs)


@numba.njit
def crop_disks_from_frame(peaks, frame, crop_size, out_crop_bufs):

    def frame_coord_y(peak, y):
        return y + peak[0] - crop_size

    def frame_coord_x(peak, x):
        return x + peak[1] - crop_size

    fy, fx = frame.shape
    for i in range(len(peaks)):
        peak = peaks[i]
        for y in range(out_crop_bufs.shape[1]):
            yy = frame_coord_y(peak, y)
            y_outside = yy < 0 or yy >= fy
            for x in range(out_crop_bufs.shape[2]):
                xx = frame_coord_x(peak, x)
                x_outside = xx < 0 or xx >= fx
                if y_outside or x_outside:
                    out_crop_bufs[i, y, x] = 0
                else:
                    out_crop_bufs[i, y, x] = frame[yy, xx]


@numba.njit
def _shift(relative_center, anchor, crop_size):
    return relative_center + anchor - np.array((crop_size, crop_size))


def get_buf_count(crop_size, n_peaks, dtype, limit=2**19):
    '''
    Calculate the optimal number of peaks in a stack to fit
    within the limit.

    Parameters
    ----------
    crop_size : int
        The cropped parts will have size (2 * crop-size, 2 * crop_size)
    n_peaks : int
        Number of peaks
    dtype : numpy.dtype
        dtype of the data for size calculation
    limit : int, optional
        Upper limit, default 1/2 MB to be L3 cache friendly

    Returns
    -------
    int
    '''
    dtype = np.dtype(dtype)
    full_size = (2 * crop_size)**2 * dtype.itemsize
    return min(max(1, limit // full_size), n_peaks)


def allocate_crop_bufs(crop_size, n_peaks, dtype, limit=2**19):
    '''
    Allocate buffer for stack of cropped peaks

    The size is optimized to fit within :code:`limit`. An aligned buffer for the FFT
    back-end is created if possible.

    Parameters
    ----------
    crop_size : int
        The cropped parts will have size (2 * crop-size, 2 * crop_size)
    n_peaks : int
        Number of peaks
    dtype : numpy.dtype
        dtype of the buffer
    limit : int, optional
        Upper limit, default 1/2 MB to be L3 cache friendly

    Returns
    -------
    crop_bufs: np.ndarray
        Shape (n, 2*crop_size, 2*crop_size)
    '''
    buf_count = get_buf_count(crop_size, n_peaks, dtype, limit)
    crop_bufs = zeros((buf_count, 2 * crop_size, 2 * crop_size), dtype=dtype)
    return crop_bufs


def process_frame_fast(template, crop_size, frame, peaks,
        out_centers, out_refineds, out_heights, out_elevations,
        crop_bufs):
    '''
    Find the parameters of peaks in a diffraction pattern by correlation with a template

    This function is designed to be used in an optimized pipeline with a pre-calculated
    Fourier transform of the match pattern and optional pre-allocated buffers.
    It is the engine of the :class:`libertem_blobfinder.udf.correlation.FastCorrelationUDF` for
    stand-alone use independent of LiberTEM.

    :meth:`libertem_blobfinder.common.correlation.process_frames_fast` offers a more
    convenient interface for batch processing.

    Parameters
    ----------
    template : numpy.ndarray
        Real Fourier transform of the correlation pattern.
        The source pattern should have size (2 * crop_size, 2 * crop_size). Please note that
        the real Fourier transform (fft.rfft2) of the source pattern has a different shape!
    crop_size : int
        Half the size of the correlation pattern. Given as a parameter since real Fourier
        transform changes the size.
    frame : np.ndarray
        Frame data. Currently, only Real values are supported.
    peaks : np.ndarray
        List of peaks of shape (n_peaks, 2)
    out_centers : np.ndarray
        Output buffer for center positions of shape (n_peaks, 2) and integer dtype.
    out_refineds : np.ndarray
        Output buffer for refined center positions of shape (n_peaks, 2) and float dtype.
    out_heights : np.ndarray
        Output buffer for peak height in log scaled frame. Shape (n_peaks, ) and float dtype.
    out_elevations : np.ndarray
        Output buffer for peak elevation in log scaled frame. Shape (n_peaks, ) and float dtype.
    crop_bufs : np.ndarray
        Aligned buffer for pyfftw. Shape (n, 2 * crop_size, 2 * crop_size) and float dtype.
        n doesn't have to match the number of peaks. Instead, it should be chosen for good L3 cache
        efficiency. :meth:`allocate_crop_bufs` can be used to allocate this buffer.

    Returns
    -------
    None
        The values are placed in the provided output buffers.

    Example
    -------

    >>> from libertem_blobfinder.common.patterns import RadialGradient
    >>> from libertem_blobfinder.base.correlation import allocate_crop_bufs
    >>>
    >>> frames, indices, peaks = libertem.utils.generate.cbed_frame(radius=4)
    >>> pattern = RadialGradient(radius=4)
    >>> crop_size = pattern.get_crop_size()
    >>> template = pattern.get_template(sig_shape=(2 * crop_size, 2 * crop_size))
    >>>
    >>> centers = np.zeros((len(frames), len(peaks), 2), dtype=np.uint16)
    >>> refineds = np.zeros((len(frames), len(peaks), 2), dtype=np.float32)
    >>> heights = np.zeros((len(frames), len(peaks)), dtype=np.float32)
    >>> elevations = np.zeros((len(frames), len(peaks)), dtype=np.float32)
    >>>
    >>> crop_bufs = allocate_crop_bufs(crop_size, len(peaks), frames.dtype)
    >>>
    >>> for i, f in enumerate(frames):
    ...     process_frame_fast(
    ...         template=template, crop_size=crop_size,
    ...         frame=f, peaks=peaks.astype(np.int32),
    ...         out_centers=centers[i], out_refineds=refineds[i],
    ...         out_heights=heights[i], out_elevations=elevations[i],
    ...         crop_bufs=crop_bufs
    ...     )
    >>> assert np.allclose(refineds[0], peaks, atol=0.1)
    '''
    buf_count = len(crop_bufs)
    block_count = (len(peaks) - 1) // buf_count + 1
    for block in range(block_count):
        start = block * buf_count
        stop = min((block + 1) * buf_count, len(peaks))
        size = stop - start
        crop_disks_from_frame(
            peaks=peaks[start:stop], frame=frame, crop_size=crop_size,
            out_crop_bufs=crop_bufs[:size]
        )
        log_scale_cropbufs_inplace(crop_bufs[:size])
        corrs = do_correlations(template, crop_bufs[:size])
        evaluate_correlations(
            corrs=corrs, peaks=peaks[start:stop], crop_size=crop_size,
            out_centers=out_centers[start:stop], out_refineds=out_refineds[start:stop],
            out_heights=out_heights[start:stop], out_elevations=out_elevations[start:stop]
        )


def process_frame_full(template, crop_size, frame, peaks,
        out_centers=None, out_refineds=None, out_heights=None, out_elevations=None,
        frame_buf=None, buf_count=None):
    '''
    Find the parameters of peaks in a diffraction pattern by correlation with a template

    This function is designed to be used in an optimized pipeline with a pre-calculated
    Fourier transform of the match pattern and optional pre-allocated buffers. It is the
    engine of the :class:`libertem_blobfinder.udf.correlation.FullFrameCorrelationUDF`
    for stand-alone use independent of LiberTEM.

    :meth:`libertem_blobfinder.common.correlation.process_frames_full` offers a more
    convenient interface for batch processing.

    Parameters
    ----------
    template : numpy.ndarray
        Real Fourier transform of the correlation pattern.
        The source pattern should have size (2 * crop_size, 2 * crop_size). Please note that
        the real Fourier transform (fft.rfft2) of the source pattern has a different shape!
    crop_size : int
        Half the size of the correlation pattern. Given as a parameter since real Fourier
        transform changes the size.
    frame : np.ndarray
        Frame data. Currently, only real values are supported.
    peaks : np.ndarray
        List of peaks of shape (n_peaks, 2)
    out_centers : np.ndarray, optional
        Output buffer for center positions of shape (n_peaks, 2) and integer dtype. Will be
        allocated if needed.
    out_refineds : np.ndarray, optional
        Output buffer for refined center positions of shape (n_peaks, 2) and float dtype. Will be
        allocated if needed.
    out_heights : np.ndarray, optional
        Output buffer for peak height in log scaled frame. Shape (n_peaks, ) and float dtype. Will
        be allocated if needed.
    out_elevations : np.ndarray, optional
        Output buffer for peak elevation in log scaled frame. Shape (n_peaks, ) and float dtype.
        Will be allocated if needed.
    frame_buf : np.ndarray
        Aligned buffer for FFT back-end, such as pyfftw. Shape of a frame and float dtype.
        :meth:`libertem_blobfinder.base.correlation.zero` can be used.
    buf_count : int
        Number of peaks to process per outer loop iteration. This allows optimization of L3 cache
        efficiency.


    Returns
    -------
    None
        The values are placed in the provided output buffers.

    Example
    -------

    >>> from libertem_blobfinder.common.patterns import RadialGradient
    >>> from libertem_blobfinder.base.correlation import get_buf_count, zeros
    >>>
    >>> frames, indices, peaks = libertem.utils.generate.cbed_frame()
    >>> pattern = RadialGradient(radius=4)
    >>> crop_size = pattern.get_crop_size()
    >>> template = pattern.get_template(sig_shape=frames[0].shape)
    >>>
    >>> centers = np.zeros((len(frames), len(peaks), 2), dtype=np.uint16)
    >>> refineds = np.zeros((len(frames), len(peaks), 2), dtype=np.float32)
    >>> heights = np.zeros((len(frames), len(peaks)), dtype=np.float32)
    >>> elevations = np.zeros((len(frames), len(peaks)), dtype=np.float32)
    >>>
    >>> frame_buf = zeros(frames[0].shape, dtype=np.float32)
    >>> buf_count = get_buf_count(crop_size, len(peaks), frame_buf.dtype)
    >>>
    >>> for i, f in enumerate(frames):
    ...     process_frame_full(
    ...         template=template, crop_size=crop_size,
    ...         frame=f, peaks=peaks.astype(np.int32),
    ...         out_centers=centers[i], out_refineds=refineds[i],
    ...         out_heights=heights[i], out_elevations=elevations[i],
    ...         frame_buf=frame_buf, buf_count=buf_count
    ...     )
    >>> assert np.allclose(refineds[0], peaks, atol=0.1)
    '''
    log_scale(frame, out=frame_buf)
    spec_part = fft.rfft2(frame_buf)
    corrspec = template * spec_part
    corr = fft.fftshift(fft.irfft2(corrspec))
    crop_bufs = np.zeros((buf_count, 2 * crop_size, 2 * crop_size), dtype=corr.dtype)
    block_count = (len(peaks) - 1) // buf_count + 1
    for block in range(block_count):
        start = block * buf_count
        stop = min(len(peaks), (block + 1) * buf_count)
        size = stop - start
        crop_disks_from_frame(
            peaks=peaks[start:stop], frame=corr, crop_size=crop_size,
            out_crop_bufs=crop_bufs[:size]
        )
        evaluate_correlations(
            corrs=crop_bufs[:size], peaks=peaks[start:stop], crop_size=crop_size,
            out_centers=out_centers[start:stop], out_refineds=out_refineds[start:stop],
            out_heights=out_heights[start:stop], out_elevations=out_elevations[start:stop]
        )
