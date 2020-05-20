import numpy as np

from libertem.udf.base import UDF

from libertem_blobfinder.base.correlation import crop_disks_from_frame, allocate_crop_bufs


class IntegrationUDF(UDF):
    def __init__(self, centers, pattern):
        super().__init__(centers=centers, pattern=pattern)

    def get_result_buffers(self):
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        return {
            'integration': self.buffer(
                kind='nav', extra_shape=(self.params.centers.shape[-2], ),
                dtype=dtype
            )
        }

    def get_task_data(self):
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
        crop_size = self.params.pattern.get_crop_size()
        crop_disks_from_frame(
            peaks=self.params.centers,
            frame=frame,
            crop_size=crop_size,
            out_crop_bufs=self.task_data.crop_bufs,
        )
        self.results.integration[:] = np.sum(self.task_data.crop_bufs * self.task_data.pattern, axis=(-1, -2))
