import functools

import numpy as np

from libertem.udf import UDF
import libertem.masks as masks
from libertem.common.container import MaskContainer

from libertem_blobfinder.common.patterns import MatchPattern
import libertem_blobfinder.base.correlation as ltbc
from libertem_blobfinder.common.correlation import get_peaks


class CorrelationUDF(UDF):
    '''
    Abstract base class for peak correlation implementations
    '''
    def __init__(self, peaks, *args, **kwargs):
        '''
        Parameters
        ----------

        peaks : numpy.ndarray
            Numpy array of (y, x) coordinates with peak positions in px to correlate
        '''
        super().__init__(peaks=np.round(peaks).astype(int), *args, **kwargs)

    def get_result_buffers(self):
        '''
        The common buffers for all correlation methods.

        :code:`centers`:
            (y, x) integer positions.
        :code:`refineds`:
            (y, x) positions with subpixel refinement.
        :code:`peak_values`:
            Peak height in the log scaled frame.
        :code:`peak_elevations`:
            Peak quality (result of :meth:`peak_elevation`).

        See source code for details of the buffer declaration.
        '''
        num_disks = len(self.params.peaks)

        return {
            'centers': self.buffer(
                kind="nav", extra_shape=(num_disks, 2), dtype="u2"
            ),
            'refineds': self.buffer(
                kind="nav", extra_shape=(num_disks, 2), dtype="float32"
            ),
            'peak_values': self.buffer(
                kind="nav", extra_shape=(num_disks,), dtype="float32",
            ),
            'peak_elevations': self.buffer(
                kind="nav", extra_shape=(num_disks,), dtype="float32",
            ),
        }

    def output_buffers(self):
        '''
        This function allows abstraction of the result buffers from
        the default implementation in :meth:`get_result_buffers`.

        Override this function if you wish to redirect the results to different
        buffers, for example ragged arrays or binned processing.
        '''
        r = self.results
        return (r.centers, r.refineds, r.peak_values, r.peak_elevations)

    def postprocess(self):
        pass

    def get_peaks(self):
        return self.params.peaks

    def get_zero_shift(self, index=None):
        return np.array((0, 0))


class FastCorrelationUDF(CorrelationUDF):
    '''
    Fourier-based fast correlation-based refinement of peak positions within a search frame
    for each peak.
    '''
    def __init__(self, peaks, match_pattern, zero_shift=None, *args, **kwargs):
        '''
        Parameters
        ----------

        peaks : numpy.ndarray
            Numpy array of (y, x) coordinates with peak positions in px to correlate
        match_pattern : MatchPattern
            Instance of :class:`~libertem_blobfinder.MatchPattern`
        zero_shift : Union[AUXBufferWrapper, numpy.ndarray], optional
            Zero shift, for example descan error. Can be :code:`None`, :code:`numpy.array((y, x))`
            or AUX data with :code:`(y, x)` for each frame.
        '''
        # For testing purposes, allow to inject a different limit via
        # an internal kwarg
        # It has to come through kwarg because of how UDFs are run
        self.limit = kwargs.get('__limit', 2**19)  # 1/2 MB
        super().__init__(
            peaks=peaks, match_pattern=match_pattern, zero_shift=zero_shift, *args, **kwargs
        )

    def get_task_data(self):
        ""
        n_peaks = len(self.get_peaks())
        mask = self.get_pattern()
        crop_size = mask.get_crop_size()
        template = mask.get_template(sig_shape=(2 * crop_size, 2 * crop_size))
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        crop_bufs = ltbc.allocate_crop_bufs(crop_size, n_peaks, dtype=dtype, limit=self.limit)
        kwargs = {
            'crop_bufs': crop_bufs,
            'template': template,
        }
        return kwargs

    def get_zero_shift(self, index=None):
        if self.params.zero_shift is None:
            result = np.array((0, 0))
        elif index is None:
            # Called when masked with view
            result = self.params.zero_shift[:]
        else:
            # Called when not masked, in postprocess() etc.
            result = self.params.zero_shift[index]
        return result

    def get_pattern(self):
        return self.params.match_pattern

    def get_template(self):
        return self.task_data.template

    def process_frame(self, frame):
        match_pattern = self.get_pattern()
        (centers, refineds, peak_values, peak_elevations) = self.output_buffers()
        ltbc.process_frame_fast(
            template=self.get_template(), crop_size=match_pattern.get_crop_size(),
            frame=frame, peaks=self.get_peaks() + np.round(self.get_zero_shift()).astype(np.int),
            out_centers=centers, out_refineds=refineds,
            out_heights=peak_values, out_elevations=peak_elevations,
            crop_bufs=self.task_data.crop_bufs
        )


class FullFrameCorrelationUDF(CorrelationUDF):
    '''
    Fourier-based correlation-based refinement of peak positions within a search
    frame for each peak using a single correlation step. This can be faster for
    correlating a large number of peaks in small frames in comparison to
    :class:`FastCorrelationUDF`. However, it is more sensitive to interference
    from strong peaks next to the peak of interest.

    .. versionadded:: 0.3.0
    '''
    def __init__(self, peaks, match_pattern, zero_shift=None, *args, **kwargs):
        '''
        Parameters
        ----------

        peaks : numpy.ndarray
            Numpy array of (y, x) coordinates with peak positions in px to correlate
        match_pattern : MatchPattern
            Instance of :class:`~libertem_blobfinder.MatchPattern`
        zero_shift : Union[AUXBufferWrapper, numpy.ndarray], optional
            Zero shift, for example descan error. Can be :code:`None`, :code:`numpy.array((y, x))`
            or AUX data with :code:`(y, x)` for each frame.
        '''
        # For testing purposes, allow to inject a different limit via
        # an internal kwarg
        # It has to come through kwarg because of how UDFs are run
        self.limit = kwargs.get('__limit', 2**19)  # 1/2 MB

        super().__init__(
            peaks=peaks, match_pattern=match_pattern, zero_shift=zero_shift, *args, **kwargs
        )

    def get_task_data(self):
        ""
        mask = self.get_pattern()
        n_peaks = len(self.params.peaks)
        template = mask.get_template(sig_shape=self.meta.dataset_shape.sig)
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        frame_buf = ltbc.zeros(shape=self.meta.dataset_shape.sig, dtype=dtype)
        crop_size = mask.get_crop_size()
        kwargs = {
            'template': template,
            'frame_buf': frame_buf,
            'buf_count': ltbc.get_buf_count(crop_size, n_peaks, dtype, self.limit),
        }
        return kwargs

    def get_zero_shift(self, index=None):
        if self.params.zero_shift is None:
            result = np.array((0, 0))
        elif index is None:
            # Called when masked with view
            result = self.params.zero_shift[:]
        else:
            # Called when not masked, in postprocess() etc.
            result = self.params.zero_shift[index]
        return result

    def get_pattern(self):
        return self.params.match_pattern

    def get_template(self):
        return self.task_data.template

    def process_frame(self, frame):
        match_pattern = self.get_pattern()
        (centers, refineds, peak_values, peak_elevations) = self.output_buffers()
        ltbc.process_frame_full(
            template=self.get_template(),
            crop_size=match_pattern.get_crop_size(),
            frame=frame,
            peaks=self.get_peaks() + np.round(self.get_zero_shift()).astype(np.int),
            out_centers=centers,
            out_refineds=refineds,
            out_heights=peak_values,
            out_elevations=peak_elevations,
            frame_buf=self.task_data.frame_buf,
            buf_count=self.task_data.buf_count,
        )


class SparseCorrelationUDF(CorrelationUDF):
    '''
    Direct correlation using sparse matrices

    This method allows to adjust the number of correlation steps independent of the template size.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------

        peaks : numpy.ndarray
            Numpy array of (y, x) coordinates with peak positions in px to correlate
        match_pattern : MatchPattern
            Instance of :class:`~libertem_blobfinder.MatchPattern`
        steps : int
            The template is correlated with 2 * steps + 1 symmetrically around the peak position
            in x and y direction. This defines the maximum shift that can be
            detected. The number of calculations grows with the square of this value, that means
            keeping this as small as the data allows speeds up the calculation.
        '''
        super().__init__(*args, **kwargs)

    def get_result_buffers(self):
        """
        This method adds the :code:`corr` buffer to the result of
        :meth:`CorrelationUDF.get_result_buffers`. See source code for the
        exact buffer declaration.
        """
        super_buffers = super().get_result_buffers()
        num_disks = len(self.params.peaks)
        steps = self.params.steps * 2 + 1
        my_buffers = {
            'corr': self.buffer(
                kind="nav", extra_shape=(num_disks * steps**2,), dtype="float32"
            ),
        }
        super_buffers.update(my_buffers)
        return super_buffers

    def get_task_data(self):
        ""
        match_pattern = self.params.match_pattern
        crop_size = match_pattern.get_crop_size()
        size = (2 * crop_size + 1, 2 * crop_size + 1)
        template = match_pattern.get_mask(sig_shape=size)
        steps = self.params.steps
        peak_offsetY, peak_offsetX = np.mgrid[-steps:steps + 1, -steps:steps + 1]

        offsetY = self.params.peaks[:, 0, np.newaxis, np.newaxis] + peak_offsetY - crop_size
        offsetX = self.params.peaks[:, 1, np.newaxis, np.newaxis] + peak_offsetX - crop_size

        offsetY = offsetY.flatten()
        offsetX = offsetX.flatten()

        stack = functools.partial(
            masks.sparse_template_multi_stack,
            mask_index=range(len(offsetY)),
            offsetX=offsetX,
            offsetY=offsetY,
            template=template,
            imageSizeX=self.meta.dataset_shape.sig[1],
            imageSizeY=self.meta.dataset_shape.sig[0]
        )
        # CSC matrices in combination with transposed data are fastest
        container = MaskContainer(mask_factories=stack, dtype=np.float32,
            use_sparse='scipy.sparse.csc')

        kwargs = {
            'mask_container': container,
            'crop_size': crop_size,
        }
        return kwargs

    def process_tile(self, tile):
        tile_slice = self.meta.slice
        c = self.task_data.mask_container
        tile_t = np.zeros(
            (np.prod(tile.shape[1:]), tile.shape[0]),
            dtype=tile.dtype
        )
        ltbc.log_scale(tile.reshape((tile.shape[0], -1)).T, out=tile_t)

        sl = c.get(key=tile_slice, transpose=False)
        self.results.corr[:] += sl.dot(tile_t).T

    def postprocess(self):
        """
        The correlation results are evaluated during postprocessing since this
        implementation uses tiled processing where the correlations are
        incomplete in :meth:`process_tile`.
        """
        steps = 2 * self.params.steps + 1
        corrmaps = self.results.corr.reshape((
            -1,  # frames
            len(self.params.peaks),  # peaks
            steps,  # Y steps
            steps,  # X steps
        ))
        peaks = self.params.peaks
        (centers, refineds, peak_values, peak_elevations) = self.output_buffers()
        for f in range(corrmaps.shape[0]):
            ltbc.evaluate_correlations(
                corrs=corrmaps[f], peaks=peaks, crop_size=self.params.steps,
                out_centers=centers[f], out_refineds=refineds[f],
                out_heights=peak_values[f], out_elevations=peak_elevations[f]
            )


def run_fastcorrelation(ctx, dataset, peaks, match_pattern: MatchPattern, roi=None, progress=False):
    """
    Wrapper function to construct and run a :class:`FastCorrelationUDF`

    Parameters
    ----------
    ctx : libertem.api.Context
    dataset : libertem.io.dataset.base.DataSet
    peaks : numpy.ndarray
        List of peaks with (y, x) coordinates
    match_pattern : libertem_blobfinder.patterns.MatchPattern
    roi : numpy.ndarray, optional
        Boolean mask of the navigation dimension to select region of interest (ROI)
    progress : bool, optional
        Show progress bar

    Returns
    -------
    buffers : Dict[libertem.common.buffers.BufferWrapper]
        See :meth:`CorrelationUDF.get_result_buffers` for details.
    """
    peaks = peaks.astype(np.int)
    udf = FastCorrelationUDF(peaks=peaks, match_pattern=match_pattern)
    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi, progress=progress)


def run_blobfinder(ctx, dataset, match_pattern: MatchPattern, num_peaks, roi=None, progress=False):
    """
    Wrapper function to find peaks in a dataset and refine their position using
    :class:`FastCorrelationUDF`

    Parameters
    ----------
    ctx : libertem.api.Context
    dataset : libertem.io.dataset.base.DataSet
    match_pattern : libertem_blobfinder.patterns.MatchPattern
    num_peaks : int
        Number of peaks to look for
    roi : numpy.ndarray, optional
        Boolean mask of the navigation dimension to select region of interest (ROI)
    progress : bool, optional
        Show progress bar

    Returns
    -------
    sum_result : numpy.ndarray
        Log-scaled sum frame of the dataset/ROI
    centers, refineds, peak_values, peak_elevations : libertem.common.buffers.BufferWrapper
        See :meth:`CorrelationUDF.get_result_buffers` for details.
    peaks : numpy.ndarray
        List of found peaks with (y, x) coordinates
    """
    sum_analysis = ctx.create_sum_analysis(dataset=dataset)
    sum_result = ctx.run(sum_analysis, roi=roi)

    sum_result = ltbc.log_scale(sum_result.intensity.raw_data, out=None)
    peaks = get_peaks(
        sum_result=sum_result,
        match_pattern=match_pattern,
        num_peaks=num_peaks,
    )

    pass_2_results = run_fastcorrelation(
        ctx=ctx,
        dataset=dataset,
        peaks=peaks,
        match_pattern=match_pattern,
        roi=roi,
        progress=progress
    )

    return (sum_result, pass_2_results['centers'],
        pass_2_results['refineds'], pass_2_results['peak_values'],
        pass_2_results['peak_elevations'], peaks)
