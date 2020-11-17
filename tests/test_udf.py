import numpy as np
import pytest
import matplotlib.pyplot as plt

import libertem.analysis.gridmatching as grm
import libertem.masks as m
from libertem.utils.generate import cbed_frame
from libertem.io.dataset.memory import MemoryDataSet
from libertem.udf.base import UDF

from libertem_blobfinder import common, udf
import libertem_blobfinder.common.correlation
import libertem_blobfinder.common.patterns
import libertem_blobfinder.udf.refinement
import libertem_blobfinder.udf.correlation
import libertem_blobfinder.udf.utils  # noqa F401

from utils import _mk_random


@pytest.mark.parametrize(
    "progress", [True, False]
)
@pytest.mark.with_numba
def test_smoke(lt_ctx, progress):
    """
    just check if the analysis runs without throwing exceptions:
    """
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)
    match_pattern = common.patterns.RadialGradient(radius=4)
    udf.correlation.run_blobfinder(
        ctx=lt_ctx, dataset=dataset, num_peaks=1, match_pattern=match_pattern,
        progress=progress
    )


@pytest.mark.parametrize(
    "progress", [True, False]
)
def test_run_refine_fastmatch(lt_ctx, progress):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    drop = np.random.choice([True, False], size=len(indices), p=[0.9, 0.1])
    indices = indices[drop]

    radius = 10

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)
    matcher = grm.Matcher()

    template = m.radial_gradient(
        centerX=radius+1,
        centerY=radius+1,
        imageSizeX=2*radius+2,
        imageSizeY=2*radius+2,
        radius=radius
    )

    match_patterns = [
        common.patterns.RadialGradient(radius=radius),
        common.patterns.Circular(radius=radius),
        common.patterns.BackgroundSubtraction(radius=radius),
        common.patterns.RadialGradientBackgroundSubtraction(radius=radius),
        common.patterns.UserTemplate(template=template)
    ]

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    for match_pattern in match_patterns:
        print("refining using template %s" % type(match_pattern))
        (res, real_indices) = udf.refinement.run_refine(
            ctx=lt_ctx,
            dataset=dataset,
            zero=zero + np.random.uniform(-1, 1, size=2),
            a=a + np.random.uniform(-1, 1, size=2),
            b=b + np.random.uniform(-1, 1, size=2),
            matcher=matcher,
            match_pattern=match_pattern,
            progress=progress
        )
        print(peaks - grm.calc_coords(
            res['zero'].data[0],
            res['a'].data[0],
            res['b'].data[0],
            indices)
        )

        assert np.allclose(res['zero'].data[0], zero, atol=0.5)
        assert np.allclose(res['a'].data[0], a, atol=0.2)
        assert np.allclose(res['b'].data[0], b, atol=0.2)


def test_run_refine_affinematch(lt_ctx):
    for i in range(1):
        try:
            shape = np.array([128, 128])

            zero = shape / 2 + np.random.uniform(-1, 1, size=2)
            a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
            b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)

            indices = np.mgrid[-2:3, -2:3]
            indices = np.concatenate(indices.T)

            radius = 10

            data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

            dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                                    num_partitions=1, sig_dims=2)

            matcher = grm.Matcher()
            match_pattern = common.patterns.RadialGradient(radius=radius)

            affine_indices = peaks - zero

            for j in range(5):
                zzero = zero + np.random.uniform(-1, 1, size=2)
                aa = np.array([1, 0]) + np.random.uniform(-0.05, 0.05, size=2)
                bb = np.array([0, 1]) + np.random.uniform(-0.05, 0.05, size=2)

                (res, real_indices) = udf.refinement.run_refine(
                    ctx=lt_ctx,
                    dataset=dataset,
                    zero=zzero,
                    a=aa,
                    b=bb,
                    indices=affine_indices,
                    matcher=matcher,
                    match_pattern=match_pattern,
                    match='affine'
                )

                assert np.allclose(res['zero'].data[0], zero, atol=0.5)
                assert np.allclose(res['a'].data[0], [1, 0], atol=0.05)
                assert np.allclose(res['b'].data[0], [0, 1], atol=0.05)
        except Exception:
            print("zero = np.array([%s, %s])" % tuple(zero))
            print("a = np.array([%s, %s])" % tuple(a))
            print("b = np.array([%s, %s])" % tuple(b))

            print("zzero = np.array([%s, %s])" % tuple(zzero))
            print("aa = np.array([%s, %s])" % tuple(aa))
            print("bb = np.array([%s, %s])" % tuple(bb))
            raise


def test_run_refine_sparse(lt_ctx):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 10

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)

    matcher = grm.Matcher()
    match_pattern = common.patterns.RadialGradient(radius=radius)

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    (res, real_indices) = udf.refinement.run_refine(
        ctx=lt_ctx,
        dataset=dataset,
        zero=zero + np.random.uniform(-0.5, 0.5, size=2),
        a=a + np.random.uniform(-0.5, 0.5, size=2),
        b=b + np.random.uniform(-0.5, 0.5, size=2),
        matcher=matcher,
        match_pattern=match_pattern,
        correlation='sparse',
        steps=3
    )

    print(peaks - grm.calc_coords(
        res['zero'].data[0],
        res['a'].data[0],
        res['b'].data[0],
        indices)
    )

    assert np.allclose(res['zero'].data[0], zero, atol=0.5)
    assert np.allclose(res['a'].data[0], a, atol=0.2)
    assert np.allclose(res['b'].data[0], b, atol=0.2)


def test_run_refine_fullframe(lt_ctx):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 10

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)

    matcher = grm.Matcher()
    match_pattern = common.patterns.RadialGradient(radius=radius)

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    (res, real_indices) = udf.refinement.run_refine(
        ctx=lt_ctx,
        dataset=dataset,
        zero=zero + np.random.uniform(-0.5, 0.5, size=2),
        a=a + np.random.uniform(-0.5, 0.5, size=2),
        b=b + np.random.uniform(-0.5, 0.5, size=2),
        matcher=matcher,
        match_pattern=match_pattern,
        correlation='fullframe',
    )

    print(peaks - grm.calc_coords(
        res['zero'].data[0],
        res['a'].data[0],
        res['b'].data[0],
        indices)
    )

    assert np.allclose(res['zero'].data[0], zero, atol=0.5)
    assert np.allclose(res['a'].data[0], a, atol=0.2)
    assert np.allclose(res['b'].data[0], b, atol=0.2)


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "cls",
    [
        udf.correlation.FastCorrelationUDF,
        udf.correlation.FullFrameCorrelationUDF,
    ]
)
def test_run_refine_blocktests(lt_ctx, cls):
    shape = np.array([128, 128])
    zero = shape / 2
    a = np.array([27.17, 0.])
    b = np.array([0., 29.19])
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 7
    match_pattern = common.patterns.RadialGradient(radius=radius)
    crop_size = match_pattern.get_crop_size()

    data, indices, peaks = cbed_frame(
        *shape, zero=zero, a=a, b=b, indices=indices, radius=radius, margin=crop_size
    )

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)

    # The crop buffer is float32
    # FIXME adapt as soon as UDFs have dtype support
    nbytes = (2*crop_size)**2 * np.dtype(np.float32).itemsize

    for limit in (
            1,
            nbytes - 1,
            nbytes,
            nbytes + 1,
            (len(peaks) - 1)*nbytes - 1,
            (len(peaks) - 1)*nbytes,
            (len(peaks) - 1)*nbytes + 1,
            len(peaks)*nbytes - 1,
            len(peaks)*nbytes,
            len(peaks)*nbytes + 1,
            *np.random.randint(low=1, high=len(peaks)*nbytes + 3, size=5)):
        m_udf = cls(peaks=peaks, match_pattern=match_pattern, __limit=limit)
        res = lt_ctx.run_udf(udf=m_udf, dataset=dataset)
        print(limit)
        print(res['refineds'].data[0])
        print(peaks)
        print(peaks - res['refineds'].data[0])
        assert np.allclose(res['refineds'].data[0], peaks, atol=0.5)


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "cls,dtype,kwargs",
    [
        (udf.correlation.FastCorrelationUDF, np.int, {}),
        (udf.correlation.FastCorrelationUDF, np.float, {}),
        (udf.correlation.FastCorrelationUDF, np.float, {'zero_shift': (2, 3)}),
        (udf.correlation.SparseCorrelationUDF, np.int, {'steps': 3}),
        (udf.correlation.SparseCorrelationUDF, np.float, {'steps': 3}),
        (udf.correlation.SparseCorrelationUDF, np.float, {'steps': 3, 'zero_shift': (2, 7)}),
    ]
)
def test_correlation_methods(lt_ctx, cls, dtype, kwargs):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 8

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)

    template = m.radial_gradient(
        centerX=radius+1,
        centerY=radius+1,
        imageSizeX=2*radius+2,
        imageSizeY=2*radius+2,
        radius=radius
    )

    match_patterns = [
        common.patterns.RadialGradient(radius=radius),
        common.patterns.Circular(radius=radius),
        common.patterns.BackgroundSubtraction(radius=radius),
        common.patterns.RadialGradientBackgroundSubtraction(radius=radius),
        common.patterns.UserTemplate(template=template)
    ]

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    for match_pattern in match_patterns:
        print("refining using template %s" % type(match_pattern))
        if cls is udf.correlation.SparseCorrelationUDF and kwargs.get('zero_shift'):
            with pytest.raises(ValueError):
                m_udf = cls(match_pattern=match_pattern, peaks=peaks.astype(dtype), **kwargs)
        else:
            m_udf = cls(match_pattern=match_pattern, peaks=peaks.astype(dtype), **kwargs)
            res = lt_ctx.run_udf(dataset=dataset, udf=m_udf)
            print(peaks)
            print(res['refineds'].data[0])
            print(peaks - res['refineds'].data[0])
            print(res['peak_values'].data[0])
            print(res['peak_elevations'].data[0])

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # plt.imshow(data[0])
            # for p in np.flip(res['refineds'].data[0], axis=-1):
            #     ax.add_artist(plt.Circle(p, radius, fill=False, color='y'))
            # plt.show()

            assert np.allclose(res['refineds'].data[0], peaks, atol=0.5)


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "cls,dtype,kwargs",
    [
        (udf.correlation.FullFrameCorrelationUDF, np.int, {}),
        (udf.correlation.FullFrameCorrelationUDF, np.float, {}),
    ]
)
def test_correlation_method_fullframe(lt_ctx, cls, dtype, kwargs):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([34.3, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 42.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 8

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)

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
        m_udf = cls(match_pattern=match_pattern, peaks=peaks.astype(dtype), **kwargs)
        res = lt_ctx.run_udf(dataset=dataset, udf=m_udf)
        print(peaks - res['refineds'].data[0])

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # plt.imshow(data[0])
        # for p in np.flip(res['refineds'].data[0], axis=-1):
        #     ax.add_artist(plt.Circle(p, radius, fill=False, color='y'))
        # plt.show()

        assert np.allclose(res['refineds'].data[0], peaks, atol=0.5)


@pytest.mark.parametrize(
    "navshape", ((1, 1), (1, ))
)
def test_visualize_smoke(navshape, lt_ctx):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 10

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    data = data.reshape((*navshape, *shape))

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)
    matcher = grm.Matcher()

    match_pattern = common.patterns.RadialGradientBackgroundSubtraction(radius=radius)

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    (res, real_indices) = udf.refinement.run_refine(
        ctx=lt_ctx,
        dataset=dataset,
        zero=zero + np.random.uniform(-1, 1, size=2),
        a=a + np.random.uniform(-1, 1, size=2),
        b=b + np.random.uniform(-1, 1, size=2),
        matcher=matcher,
        match_pattern=match_pattern
    )

    fig, axes = plt.subplots()
    if len(navshape) == 1:
        y = None
    elif len(navshape) == 2:
        y = 0
    else:
        raise ValueError(f"Nav shape too long, supported are 1D and 2D: {navshape}")
    udf.utils.visualize_frame(
        ctx=lt_ctx, ds=dataset, result=res, indices=real_indices,
        r=radius, y=y, x=0, axes=axes
    )
    # plt.show()


def test_run_refine_fastmatch_zeroshift(lt_ctx):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    drop = np.random.choice([True, False], size=len(indices), p=[0.9, 0.1])
    indices = indices[drop]

    radius = 10
    # Exactly between peaks, worst case
    shift = (a + b) / 2

    data_0, indices_0, peaks_0 = cbed_frame(*shape, zero, a, b, indices, radius)
    data_1, indices_1, peaks_1 = cbed_frame(*shape, zero + shift, a, b, indices, radius)

    data = np.concatenate((data_0, data_1), axis=0)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)
    matcher = grm.Matcher()

    match_patterns = [
        # Least reliable pattern
        common.patterns.Circular(radius=radius),
    ]

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    for match_pattern in match_patterns:
        print("refining using template %s" % type(match_pattern))
        zero_shift = np.array([(0., 0.), shift]).astype(np.float32)
        (res, real_indices) = udf.refinement.run_refine(
            ctx=lt_ctx,
            dataset=dataset,
            zero=zero + np.random.uniform(-1, 1, size=2),
            a=a + np.random.uniform(-1, 1, size=2),
            b=b + np.random.uniform(-1, 1, size=2),
            matcher=matcher,
            match_pattern=match_pattern,
            zero_shift=UDF.aux_data(zero_shift, kind='nav', extra_shape=(2,))
        )
        print(peaks_0 - grm.calc_coords(
            res['zero'].data[0],
            res['a'].data[0],
            res['b'].data[0],
            indices_0)
        )

        print(peaks_1 - grm.calc_coords(
            res['zero'].data[1],
            res['a'].data[1],
            res['b'].data[1],
            indices_1)
        )

        assert np.allclose(res['zero'].data[0], zero, atol=0.5)
        assert np.allclose(res['zero'].data[1], zero + shift, atol=0.5)
        assert np.allclose(res['a'].data, a, atol=0.2)
        assert np.allclose(res['b'].data, b, atol=0.2)
