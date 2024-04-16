import numpy as np
from typing import Tuple

from libertem_blobfinder.base import masks
from skimage.util import crop


class MatchPattern:
    '''
    Abstract base class for correlation patterns.

    This class provides an API to provide a template for fast correlation-based peak finding.
    '''
    def __init__(self, search):
        '''
        Parameters
        ----------

        search : float
            Range from the center point in px to include in the correlation, defining the size
            of the square correlation pattern.
            Will be ceiled to the next int for performing the correlation.
        '''
        self.search = search

    def get_crop_size(self):
        return int(np.ceil(self.search))

    def get_mask(self, sig_shape):
        raise NotImplementedError

    def get_template(self, sig_shape):
        return np.fft.rfft2(self.get_mask(sig_shape))


class Circular(MatchPattern):
    '''
    Circular pattern with radius :code:`radius`.

    This pattern is useful for constructing feature vectors using
    :meth:`~libertem_blobfinder.common.patterns.feature_vector`.

    .. versionadded:: 0.3.0
    '''
    def __init__(self, radius, search=None):
        '''
        Parameters
        ----------

        radius : float
            Radius of the circular pattern in px
        search : float, optional
            Range from the center point in px to include in the correlation, 2x radius by default.
            Defining the size of the square correlation pattern.
        '''
        if search is None:
            search = 2*radius
        if search < radius:
            raise ValueError(
                f"search {search} < radius {radius}, "
                "search must contain the pattern."
            )
        self.radius = radius
        super().__init__(search=search)

    def get_mask(self, sig_shape):
        return masks.circular(
            centerY=sig_shape[0] // 2,
            centerX=sig_shape[1] // 2,
            imageSizeY=sig_shape[0],
            imageSizeX=sig_shape[1],
            radius=self.radius,
            antialiased=True,
        )


class RadialGradient(MatchPattern):
    '''
    Radial gradient from zero in the center to one at :code:`radius`.

    This pattern rejects the influence of internal intensity variations of the CBED disk.
    '''
    def __init__(self, radius, search=None):
        '''
        Parameters
        ----------

        radius : float
            Radius of the circular pattern in px
        search : float, optional
            Range from the center point in px to include in the correlation, 2x radius by default.
            Defining the size of the square correlation pattern.
        '''
        if search is None:
            search = 2*radius
        if search < radius:
            raise ValueError(
                f"search {search} < radius {radius}, "
                "search must contain the pattern."
            )
        self.radius = radius
        super().__init__(search=search)

    def get_mask(self, sig_shape):
        return masks.radial_gradient(
            centerY=sig_shape[0] // 2,
            centerX=sig_shape[1] // 2,
            imageSizeY=sig_shape[0],
            imageSizeX=sig_shape[1],
            radius=self.radius,
            antialiased=True,
        )


class BackgroundSubtraction(MatchPattern):
    '''
    Solid circular disk surrounded with a balancing negative area

    This pattern rejects background and avoids false positives at positions between peaks
    '''
    def __init__(self, radius, search=None, radius_outer=None):
        '''
        Parameters
        ----------

        radius : float
            Radius of the circular pattern in px
        search : float, optional
            Range from the center point in px to include in the correlation.
            :code:`max(2*radius, radius_outer)` by default.
            Defining the size of the square correlation pattern.
        radius_outer : float, optional
            Radius of the negative region in px. 1.5x radius by default.
        '''
        if radius_outer is None:
            radius_outer = radius * 1.5
        if search is None:
            search = max(2*radius, radius_outer)
        if radius_outer <= radius:
            raise ValueError(f"radius_outer {radius_outer} <= radius {radius}, must be larger.")
        if search < radius_outer:
            raise ValueError(
                f"search {search} < radius_outer {radius_outer}, "
                "search must contain the pattern."
            )
        self.radius = radius
        self.radius_outer = radius_outer
        super().__init__(search=search)

    def get_mask(self, sig_shape):
        return masks.background_subtraction(
            centerY=sig_shape[0] // 2,
            centerX=sig_shape[1] // 2,
            imageSizeY=sig_shape[0],
            imageSizeX=sig_shape[1],
            radius=self.radius_outer,
            radius_inner=self.radius,
            antialiased=True
        )


class UserTemplate(MatchPattern):
    '''
    User-defined template
    '''
    def __init__(self, template: np.ndarray, search=None):
        '''
        Parameters
        ----------

        template : numpy.ndarray
            Correlation template as 2D numpy.ndarray
        search : float, optional
            Range from the center point in px to include in the correlation.
            Half diagonal of the template by default.
            Defining the size of the square correlation pattern.
        '''
        if search is None:
            # Half diagonal
            search = np.sqrt(template.shape[0]**2 + template.shape[1]**2) / 2
        self.template = template
        super().__init__(search=search)

    def get_mask(self, sig_shape: Tuple[int, int]) -> np.ndarray:
        # Pad or Crop each dimension of self.template to
        # match sig_shape at the ouput. For odd pads/crops
        # the extra pixel is added/removed at the end of the axis
        result = self.template.copy()
        neutral = (0, 0)
        for ax, (target, source) in enumerate(zip(sig_shape, self.template.shape)):
            if target > source:
                extra = target - source
                fn = np.pad
            elif target < source:
                extra = source - target
                fn = crop
            else:
                continue
            before = after = extra // 2
            if (before + after) != extra:
                # In the default case sig_shape is always even (2 * crop_size)
                # so this path implies the template has an odd dimension.
                # therefore a choice here of how to centre the array
                # before += 1
                after += 1
            result = fn(
                result,
                tuple(
                    (before, after) if ax == i else neutral
                    for i in range(result.ndim)
                )
            )
        assert result.shape == tuple(sig_shape)
        return result.astype(self.template.dtype)


class RadialGradientBackgroundSubtraction(UserTemplate):
    '''
    Combination of radial gradient with background subtraction
    '''
    def __init__(self, radius, search=None, radius_outer=None, delta=1, radial_map=None):
        '''
        See :meth:`~libertem_blobfinder.base.masks.radial_gradient_background_subtraction`
        for details.

        Parameters
        ----------

        radius : float
            Radius of the circular pattern in px
        search : float, optional
            Range from the center point in px to include in the correlation.
            :code:`max(2*radius, radius_outer)` by default
            Defining the size of the square correlation pattern.
        radius_outer : float, optional
            Radius of the negative region in px. 1.5x radius by default.
        delta : float, optional
            Width of the transition region between positive and negative in px
        radial_map : numpy.ndarray, optional
            Radius value of each pixel in px. This can be used to distort the shape as needed
            or work in physical coordinates instead of pixels.
            A suitable map can be generated with :meth:`libertem_blobfinder.base.masks.polar_map`.

        Example
        -------

        >>> import matplotlib.pyplot as plt

        >>> (radius, phi) = libertem_blobfinder.base.masks.polar_map(
        ...     centerX=64, centerY=64,
        ...     imageSizeX=128, imageSizeY=128,
        ...     stretchY=2., angle=np.pi/4
        ... )

        >>> template = RadialGradientBackgroundSubtraction(
        ...     radius=30, radial_map=radius)

        >>> # This shows an elliptical template that is stretched
        >>> # along the 45 ° bottom-left top-right diagonal
        >>> plt.imshow(template.get_mask(sig_shape=(128, 128)))
        <matplotlib.image.AxesImage object at ...>
        >>> plt.show() # doctest: +SKIP
        '''
        if radius_outer is None:
            radius_outer = radius * 1.5
        if search is None:
            search = max(2*radius, radius_outer)
        if radius_outer <= radius:
            raise ValueError(f"radius_outer {radius_outer} <= radius {radius}, must be larger.")
        if search < radius_outer:
            raise ValueError(
                f"search {search} < radius_outer {radius_outer}, "
                "search must contain the pattern."
            )
        if radial_map is None:
            r = max(radius, radius_outer)
            radial_map, _ = masks.polar_map(
                centerX=r + 1,
                centerY=r + 1,
                imageSizeX=int(np.ceil(2*r + 2)),
                imageSizeY=int(np.ceil(2*r + 2)),
            )
        self.radius = radius
        self.radius_outer = radius_outer
        self.delta = delta
        self.radial_map = radial_map
        template = masks.radial_gradient_background_subtraction(
            r=self.radial_map,
            r0=self.radius,
            r_outer=self.radius_outer,
            delta=self.delta
        )
        super().__init__(template=template, search=search)

    def get_mask(self, sig_shape):
        # Recalculate in case someone has changed parameters
        self.template = masks.radial_gradient_background_subtraction(
            r=self.radial_map,
            r0=self.radius,
            r_outer=self.radius_outer,
            delta=self.delta
        )
        return super().get_mask(sig_shape)


def feature_vector(imageSizeX, imageSizeY, peaks, match_pattern: MatchPattern):
    '''
    This function generates a sparse mask stack to extract a feature vector.

    A match template based on the parameters in :code:`parameters` is placed at
    each peak position in an individual mask layer. This mask stack can then
    be used in :class:`libertem.udf.masks.ApplyMasksUDF` to generate a feature
    vector for each frame.

    Summing up the mask stack along the first axis generates a mask that can be used for virtual
    darkfield imaging of all peaks together.

    Parameters
    ----------

    imageSizeX,imageSizeY : int
        Frame size in px
    peaks : numpy.ndarray
        Peak positions in px as numpy.ndarray of shape (n, 2) with integer type
    match_pattern : MatchPattern
        Instance of :class:`~MatchPattern`
    '''
    crop_size = match_pattern.get_crop_size()
    return masks.sparse_template_multi_stack(
        mask_index=range(len(peaks)),
        offsetX=peaks[:, 1] - crop_size,
        offsetY=peaks[:, 0] - crop_size,
        template=match_pattern.get_mask((2*crop_size + 1, 2*crop_size + 1)),
        imageSizeX=imageSizeX,
        imageSizeY=imageSizeY,
    )
