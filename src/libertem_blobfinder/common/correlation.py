from typing import Union

import numpy as np
from skimage.feature import peak_local_max

from libertem_blobfinder.common.patterns import MatchPattern
from libertem_blobfinder.base import correlation


def get_correlation(sum_result, match_pattern: MatchPattern):
    '''
    Calculate the correlation between :code:`sum_result` and :code:`match_pattern`.

    .. versionadded:: 0.4.0.dev0

    Parameters
    ----------

    sum_result: numpy.ndarray
        2D result frame as correlation input
    match_pattern : MatchPattern
        Instance of :class:`~libertem_blobfinder.MatchPattern` to correlate
        :code:`sum_result` with
    '''
    spec_mask = match_pattern.get_template(sig_shape=sum_result.shape)
    spec_sum = correlation.fft.rfft2(sum_result)
    corrspec = spec_mask * spec_sum
    return correlation.fft.fftshift(correlation.fft.irfft2(corrspec))


def get_peaks(sum_result, match_pattern: MatchPattern, num_peaks):
    '''
    Find peaks of the correlation between :code:`sum_result` and :code:`match_pattern`.

    The result  can then be used as input to
    :meth:`~libertem_blobfinder.common.fullmatch.FullMatcher.full_match`
    to extract grid parameters, :meth:`~libertem_blobfinder.correlation.run_fastcorrelation`
    to find the position in each frame or to construct a mask to extract feature vectors with
    :meth:`~libertem_blobfinder.common.patterns.feature_vector`.

    Parameters
    ----------

    sum_result: numpy.ndarray
        2D result frame as correlation input
    match_pattern : MatchPattern
        Instance of :class:`~libertem_blobfinder.MatchPattern` to correlate
        :code:`sum_result` with
    num_peaks : int
        Number of peaks to find

    Example
    -------
    >>> from libertem_blobfinder.base.utils import cbed_frame
    >>>
    >>> frame, _, _ = cbed_frame(radius=4)
    >>> pattern = libertem_blobfinder.common.patterns.RadialGradient(radius=4)
    >>> peaks = get_peaks(frame[0], pattern, 7)
    >>> print(peaks)
    [[64 64]
     [64 80]
     [80 80]
     [80 64]
     [48 80]
     [48 64]
     [64 96]]
    '''
    corr = get_correlation(sum_result, match_pattern)
    peaks = peak_local_max(corr, num_peaks=num_peaks)
    return peaks


def process_frames_fast(
    pattern: MatchPattern, frames, peaks,
    upsample: Union[bool, int] = False
):
    '''
    Find the parameters of peaks in a diffraction pattern by correlation with a match pattern.

    This method crops regions of interest around the peaks from the frames before correlation,
    which is usually fastest for a moderate amount of moderately sized peaks per frame.

    .. note::
        :class:`~libertem_blobfinder.udf.correlation.FastCorrelationUDF` is a
        parallelized, distributed version for large-scale data.

    Parameters
    ----------
    pattern : MatchPattern
        Pattern to correlate with.
    frames : np.ndarray
        Frame data. Currently, only Real values are supported.
    peaks : np.ndarray
        List of peaks of shape (n_peaks, 2)
    upsample: Union[bool, int], optional
        Use DFT upsampling for the refinement step, by default False. Supplying
        True will choose a reasonable default upsampling factor, while any
        positive integer > 1 will upsample the correlation peak by this factor.
        DFT upsampling can provide more accurate center values, especially when
        peak shifts are small, but does require more computation time.

    Returns
    -------
    centers : np.ndarray
        Center positions of shape (n_peaks, 2) and integer dtype.
    refineds : np.ndarray
        Refined center positions of shape (n_peaks, 2) and float dtype.
    heights : np.ndarray
        Peak height in log scaled frame. Shape (n_peaks, ) and float dtype.
    elevations : np.ndarray
        Peak elevation in log scaled frame. Shape (n_peaks, ) and float dtype

    Example
    -------
    >>> from libertem_blobfinder.base.utils import cbed_frame
    >>>
    >>> frames, indices, peaks = cbed_frame()
    >>> pattern = libertem_blobfinder.common.patterns.RadialGradient(radius=4)
    >>> (centers, refineds, heights, elevations) = process_frames_fast(
    ...     pattern=pattern,
    ...     frames=frames,
    ...     peaks=peaks.astype(np.int32),
    ... )
    >>> assert np.allclose(refineds[0], peaks, atol=0.1)
    '''

    crop_size = pattern.get_crop_size()
    template = pattern.get_template(sig_shape=(2 * crop_size, 2 * crop_size))

    centers = np.zeros((len(frames), len(peaks), 2), dtype=np.int16)
    refineds = np.zeros((len(frames), len(peaks), 2), dtype=np.float32)
    heights = np.zeros((len(frames), len(peaks)), dtype=np.float32)
    elevations = np.zeros((len(frames), len(peaks)), dtype=np.float32)

    crop_bufs = correlation.allocate_crop_bufs(crop_size, len(peaks), frames.dtype)

    for i, f in enumerate(frames):
        correlation.process_frame_fast(
            template=template, crop_size=crop_size,
            frame=f, peaks=peaks.astype(np.int32),
            out_centers=centers[i], out_refineds=refineds[i],
            out_heights=heights[i], out_elevations=elevations[i],
            crop_bufs=crop_bufs, upsample=upsample,
        )
    return (centers, refineds, heights, elevations)


def process_frames_full(
    pattern: MatchPattern, frames, peaks,
    upsample: Union[bool, int] = False
):
    '''
    Find the parameters of peaks in a diffraction pattern by correlation with a match pattern.

    This method crops regions of interest around the peaks after correlation,
    which can be faster for many peaks on smaller frames.

    .. note::
        :class:`~libertem_blobfinder.udf.correlation.FullFrameCorrelationUDF` is a
        parallelized, distributed version for large-scale data.


    Parameters
    ----------
    pattern : MatchPattern
        Pattern to correlate with.
    frame : np.ndarray
        Frame data. Currently, only real values are supported.
    peaks : np.ndarray
        List of peaks of shape (n_peaks, 2)
    upsample: Union[bool, int], optional
        Use DFT upsampling for the refinement step, by default False. Supplying
        True will choose a reasonable default upsampling factor, while any
        positive integer > 1 will upsample the correlation peak by this factor.
        DFT upsampling can provide more accurate center values, especially when
        peak shifts are small, but does require more computation time.

    Returns
    -------
    centers : np.ndarray
        Center positions of shape (n_peaks, 2) and integer dtype.
    refineds : np.ndarray
        Refined center positions of shape (n_peaks, 2) and float dtype.
    heights : np.ndarray
        Peak height in log scaled frame. Shape (n_peaks, ) and float dtype.
    elevations : np.ndarray
        Peak elevation in log scaled frame. Shape (n_peaks, ) and float dtype

    Example
    -------

    >>> from libertem_blobfinder.base.utils import cbed_frame
    >>>
    >>> frames, indices, peaks = cbed_frame(radius=4)
    >>> pattern = libertem_blobfinder.common.patterns.RadialGradient(radius=4)
    >>> (centers, refineds, heights, elevations) = process_frames_full(
    ...     pattern=pattern,
    ...     frames=frames,
    ...     peaks=peaks.astype(np.int32)
    ... )
    >>> assert np.allclose(refineds[0], peaks, atol=0.1)
    '''
    crop_size = pattern.get_crop_size()
    template = pattern.get_template(sig_shape=frames[0].shape)

    centers = np.zeros((len(frames), len(peaks), 2), dtype=np.uint16)
    refineds = np.zeros((len(frames), len(peaks), 2), dtype=np.float32)
    heights = np.zeros((len(frames), len(peaks)), dtype=np.float32)
    elevations = np.zeros((len(frames), len(peaks)), dtype=np.float32)

    frame_buf = correlation.zeros(frames[0].shape, dtype=np.float32)

    buf_count = correlation.get_buf_count(crop_size, len(peaks), frame_buf.dtype)

    for i, f in enumerate(frames):
        correlation.process_frame_full(
            template=template, crop_size=crop_size,
            frame=f, peaks=peaks.astype(np.int32),
            out_centers=centers[i], out_refineds=refineds[i],
            out_heights=heights[i], out_elevations=elevations[i],
            frame_buf=frame_buf, buf_count=buf_count, upsample=upsample,
        )
    return (centers, refineds, heights, elevations)
