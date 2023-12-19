import functools

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.signal import correlate2d

import libertem.masks as m
from libertem.utils.generate import cbed_frame
from libertem.io.dataset.memory import MemoryDataSet
from libertem.udf.masks import ApplyMasksUDF

from libertem_blobfinder import common
import libertem_blobfinder.common.correlation
from libertem_blobfinder.common.correlation import process_frames_fast
import libertem_blobfinder.common.patterns
from libertem_blobfinder.common.patterns import UserTemplate
import libertem_blobfinder.udf.refinement  # noqa F401


def test_custom_template():
    template = m.radial_gradient(centerX=10, centerY=10, imageSizeX=21, imageSizeY=23, radius=7)
    custom = common.patterns.UserTemplate(template=template, search=18)

    assert custom.get_crop_size() == 18

    same = custom.get_mask((23, 21))
    larger = custom.get_mask((25, 23))
    smaller = custom.get_mask((10, 10))

    assert np.allclose(same, template)
    assert np.allclose(larger[1:-1, 1:-1], template)
    assert np.allclose(template[6:-7, 5:-6], smaller)


def test_custom_template_fuzz():
    for i in range(10):
        integers = np.arange(1, 15)
        center_y = np.random.choice(integers)
        center_x = np.random.choice(integers)

        size_y = np.random.choice(integers)
        size_x = np.random.choice(integers)

        radius = np.random.choice(integers)
        search = np.random.choice(integers)

        mask_y = np.random.choice(integers)
        mask_x = np.random.choice(integers)

        print("center_y:", center_y)
        print("center_x:", center_x)
        print("size_y:", size_y)
        print("size_x:", size_x)
        print("radius:", radius)
        print("search:", search)
        print("mask_y:", mask_y)
        print("mask_x:", mask_x)

        template = m.radial_gradient(
            centerX=center_x, centerY=center_y,
            imageSizeX=size_x, imageSizeY=size_y,
            radius=radius
        )
        custom = common.patterns.UserTemplate(template=template, search=search)

        mask = custom.get_mask((mask_y, mask_x))  # noqa


def test_featurevector(lt_ctx):
    shape = np.array([128, 128])
    zero = shape // 2
    a = np.array([24, 0.])
    b = np.array([0., 30])
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 5
    radius_outer = 10

    template = m.background_subtraction(
        centerX=radius_outer + 1,
        centerY=radius_outer + 1,
        imageSizeX=radius_outer*2 + 2,
        imageSizeY=radius_outer*2 + 2,
        radius=radius_outer,
        radius_inner=radius + 1,
        antialiased=False
    )

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius, all_equal=True)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)

    match_pattern = common.patterns.UserTemplate(template=template)

    stack = functools.partial(
        common.patterns.feature_vector,
        imageSizeX=shape[1],
        imageSizeY=shape[0],
        peaks=peaks,
        match_pattern=match_pattern,
    )

    m_udf = ApplyMasksUDF(
        mask_factories=stack, mask_count=len(peaks), mask_dtype=np.float32
    )
    res = lt_ctx.run_udf(dataset=dataset, udf=m_udf)

    peak_data, _, _ = cbed_frame(*shape, zero, a, b, np.array([(0, 0)]), radius, all_equal=True)
    peak_sum = peak_data.sum()

    assert np.allclose(res['intensity'].data.sum(), data.sum())
    assert np.allclose(res['intensity'].data, peak_sum)


@pytest.mark.with_numba
def test_standalone_fast():
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([34.3, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 42.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 8

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    template = m.radial_gradient(
        centerX=radius+1,
        centerY=radius+1,
        imageSizeX=2*radius+2,
        imageSizeY=2*radius+2,
        radius=radius
    )

    match_patterns = [
        common.patterns.RadialGradient(radius=radius, search=radius*1.5),
        common.patterns.BackgroundSubtraction(
            radius=radius, radius_outer=radius*1.5, search=radius*1.8),
        common.patterns.RadialGradientBackgroundSubtraction(
            radius=radius, radius_outer=radius*1.5, search=radius*1.8),
        common.patterns.UserTemplate(template=template, search=radius*1.5)
    ]

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    for match_pattern in match_patterns:
        print("refining using template %s" % type(match_pattern))
        (centers, refineds, heights, elevations) = common.correlation.process_frames_fast(
            pattern=match_pattern,
            frames=data, peaks=peaks.astype(np.int32),
        )

        print(peaks - refineds)

        assert np.allclose(refineds[0], peaks, atol=0.5)


@pytest.mark.with_numba
def test_standalone_full():
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([34.3, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 42.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 8

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    template = m.radial_gradient(
        centerX=radius+1,
        centerY=radius+1,
        imageSizeX=2*radius+2,
        imageSizeY=2*radius+2,
        radius=radius
    )

    match_patterns = [
        common.patterns.RadialGradient(radius=radius, search=radius*1.5),
        common.patterns.BackgroundSubtraction(
            radius=radius, radius_outer=radius*1.5, search=radius*1.8),
        common.patterns.RadialGradientBackgroundSubtraction(
            radius=radius, radius_outer=radius*1.5, search=radius*1.8),
        common.patterns.UserTemplate(template=template, search=radius*1.5)
    ]

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    for match_pattern in match_patterns:
        print("refining using template %s" % type(match_pattern))
        (centers, refineds, heights, elevations) = common.correlation.process_frames_full(
            pattern=match_pattern,
            frames=data, peaks=peaks.astype(np.int32),
        )

        print(peaks - refineds)

        assert np.allclose(refineds[0], peaks, atol=0.5)


@pytest.mark.parametrize(
        'where', ((0, 0), (2, 3)),
)
@pytest.mark.parametrize(
        'expected', ((1, 1), (3, 2)),
)
def test_scipy_correlate2d(where, expected):
    frame = np.zeros((5, 6))
    frame[where] = 1
    pattern = frame.copy()
    correlated = correlate2d(frame, pattern, mode='same')
    ref_center = np.unravel_index(
        np.argmax(correlated),
        frame.shape
    )
    centers, refineds, heights, elevations = process_frames_fast(
        pattern=UserTemplate(pattern),
        frames=np.array([frame]),
        peaks=np.array([expected]),
        upsample=False
    )
    print(frame)
    print(ref_center, centers[0, 0])
    assert_allclose(ref_center, centers[0, 0])
