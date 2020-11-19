import numpy as np

from libertem.udf.base import UDF

from libertem_blobfinder.base.correlation import crop_disks_from_frame, allocate_crop_bufs


class IntegrationUDF(UDF):
    def __init__(self, centers, pattern):
        '''
        Integrate peak intensity at positions that are specified for each frame.

        Parameters
        ----------
        centers : AUXBufferWrapper
            Peak positions (y, x) as AUX buffer wrapper of kind "nav", extra_shape (num_peaks, 2)
            and integer dtype.
        pattern : libertem_blobfinder.common.patterns.MatchPattern
            Match pattern with the weight for each pixels.
            :class:`libertem_blobfinder.common.patterns.BackgroundSubtraction` or
            :class:`libertem_blobfinder.common.patterns.Circular` can be good choices.

        Example
        -------

        >>> from libertem_blobfinder.udf.integration import IntegrationUDF
        >>> from libertem_blobfinder.common.patterns import BackgroundSubtraction

        >>> nav_shape = tuple(dataset.shape.nav)
        >>> sig_shape = tuple(dataset.shape.sig)
        >>> extra_shape = (3, 2)  # three peaks with coordinates (y, x)
        >>> peaks_shape = nav_shape + extra_shape

        >>> # Generate some random positions as an example
        >>> peaks = np.random.randint(low=0, high=np.min(sig_shape), size=peaks_shape)

        >>> # Create an AuxBufferWrapper for the peaks
        >>> centers = IntegrationUDF.aux_data(
        ...     data=peaks,
        ...     kind='nav',
        ...     dtype=np.int,
        ...     extra_shape=extra_shape
        ... )

        >>> udf = IntegrationUDF(
        ...     centers=centers,
        ...     pattern=BackgroundSubtraction(radius=5, radius_outer=6)
        ... )

        >>> res = ctx.run_udf(udf=udf, dataset=dataset)

        >>> nav_shape
        (16, 16)
        >>> # Integration result for each frame and peak
        >>> res['integration'].data.shape
        (16, 16, 3)
        '''
        super().__init__(centers=centers, pattern=pattern)

    def get_result_buffers(self):
        '''
        :code:`integration`:
            Integrated intensity for each peak. Kind "nav", extra_shape (num_peaks, )
        '''
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        return {
            'integration': self.buffer(
                kind='nav', extra_shape=(self.params.centers.shape[-2], ),
                dtype=dtype
            )
        }

    def get_task_data(self):
        '''
        '''
        n_peaks = self.params.centers.shape[-2]
        mask = self.params.pattern
        crop_size = mask.get_crop_size()
        pattern = mask.get_mask(sig_shape=(2 * crop_size, 2 * crop_size))
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        crop_bufs = allocate_crop_bufs(crop_size, n_peaks, dtype=dtype, limit=1e12)
        kwargs = {
            'crop_bufs': crop_bufs,
            'pattern': pattern,
        }
        return kwargs

    def process_frame(self, frame):
        '''
        '''
        crop_size = self.params.pattern.get_crop_size()
        crop_disks_from_frame(
            peaks=self.params.centers,
            frame=frame,
            crop_size=crop_size,
            out_crop_bufs=self.task_data.crop_bufs,
        )
        self.results.integration[:] = np.sum(
            self.task_data.crop_bufs * self.task_data.pattern, axis=(-1, -2)
        )
