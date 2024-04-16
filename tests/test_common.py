import numpy as np
import pytest

import libertem_blobfinder.base.masks as m
from libertem_blobfinder.common import patterns, correlation
from libertem_blobfinder.base.utils import cbed_frame


def test_circular_limits():
    with pytest.raises(ValueError):
        patterns.Circular(radius=5, search=4)


def test_radial_gradient_limits():
    with pytest.raises(ValueError):
        patterns.RadialGradient(radius=5, search=4)


def test_background_subtraction_limits():
    with pytest.raises(ValueError):
        patterns.BackgroundSubtraction(radius=5, search=4, radius_outer=6)
    with pytest.raises(ValueError):
        patterns.BackgroundSubtraction(radius=5, search=13, radius_outer=5)


def test_radial_background_subtraction_limits():
    with pytest.raises(ValueError):
        patterns.RadialGradientBackgroundSubtraction(radius=5, search=4, radius_outer=6)
    with pytest.raises(ValueError):
        patterns.RadialGradientBackgroundSubtraction(radius=5, search=13, radius_outer=5)


def test_custom_template():
    template = m.radial_gradient(centerX=10, centerY=10, imageSizeX=21, imageSizeY=23, radius=7)
    custom = patterns.UserTemplate(template=template, search=18)

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
        custom = patterns.UserTemplate(template=template, search=search)

        mask = custom.get_mask((mask_y, mask_x))  # noqa


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
        patterns.RadialGradient(radius=radius, search=radius*1.5),
        patterns.BackgroundSubtraction(
            radius=radius, radius_outer=radius*1.5, search=radius*1.8),
        patterns.RadialGradientBackgroundSubtraction(
            radius=radius, radius_outer=radius*1.5, search=radius*1.8),
        patterns.UserTemplate(template=template, search=radius*1.5)
    ]

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    for match_pattern in match_patterns:
        print("refining using template %s" % type(match_pattern))
        (centers, refineds, heights, elevations) = correlation.process_frames_fast(
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
        patterns.RadialGradient(radius=radius, search=radius*1.5),
        patterns.BackgroundSubtraction(
            radius=radius, radius_outer=radius*1.5, search=radius*1.8),
        patterns.RadialGradientBackgroundSubtraction(
            radius=radius, radius_outer=radius*1.5, search=radius*1.8),
        patterns.UserTemplate(template=template, search=radius*1.5)
    ]

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    for match_pattern in match_patterns:
        print("refining using template %s" % type(match_pattern))
        (centers, refineds, heights, elevations) = correlation.process_frames_full(
            pattern=match_pattern,
            frames=data, peaks=peaks.astype(np.int32),
        )

        print(peaks - refineds)

        assert np.allclose(refineds[0], peaks, atol=0.5)
