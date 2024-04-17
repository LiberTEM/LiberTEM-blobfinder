Changelog
=========

.. _continuous:

0.7.0.dev0 (continuous)
#######################

.. toctree::
   :glob:

   changelog/*/*

.. _latest:
.. _`v0-6-1`:

0.6.1 / 2024-04-17
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10986304.svg
  :target: https://doi.org/10.5281/zenodo.10986304

This is a no-change release to hopefully fix our zenodo integration.

.. _`v0-6-0`:

0.6.0 / 2024-04-17
##################

This version now supports Python up to 3.12 and requires at least Python 3.9.
We are now using hatchling to build the package.

Features
--------

* Fourier upsampling is now implemented for calculating the
  'refineds' peak positions in correlation UDFs. This approach
  can give more precise results when peak shifts are sub-pixel
  at the expense of increased computation, and is available
  using the :code:`upsample=True` argument to UDF and associated
  functions (see :issue:`39`, :pr:`70`).

* All correlation UDFs now support GPU processing
  with :code:`cupy`, in addition to sparse input
  with the Fast and Sparse correlation UDFs. Conversion
  from unsupported backends is automatically handled
  using the `sparseconverter <https://github.com/LiberTEM/sparseconverter>`_
  package. (see :pr:`61`)

Misc
----

* The :mod:`libertem_blobfinder.common` module and associated tests
  have been refactored to remove any dependency on LiberTEM. notably
  this includes all mask-generating functions previously found in
  :mod:`libertem.masks`, which are now found also in
  :mod:`libertem_blobfinder.base.masks`. As part of this change
  the :code:`common` extra dependency group has been removed (:pr:`87`).

* Move :code:`gridmatching` and :code:`fullmatch` from :code:`libertem.analysis` to
  :mod:`libertem_blobfinder.common.gridmatching` and :mod:`libertem_blobfinder.common.fullmatch` since
  they make more sense here (:pr:`83`).


.. _`v0-5-0`:

0.5.0 / 2023-05-08
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7907860.svg
   :target: https://doi.org/10.5281/zenodo.7907860

This version now supports Python up to version 3.11, and drops support for
Python 3.6.

Features
--------

* Integration of peak intensities at per-frame positions using
  :class:`libertem_blobfinder.udf.integration.IntegrationUDF`. This can be used
  if peaks are shifted so much that integration at equal positions for all
  frames using :meth:`libertem_blobfinder.common.patterns.feature_vector`
  doesn't work anymore. (:pr:`27`)
* The :code:`run_*` functions that wrap various UDFs now support the
  :code:`progress` argument of :meth:`libertem.api.Context.run_udf` that was
  introduced in LiberTEM 0.5.0.dev0. (:pr:`22`)

* Allow specifying per-frame origin shift (:pr:`23`)
    * Introduce the :code:`zero_shift` parameter to
      :class:`libertem_blobfinder.udf.correlation.CorrelationUDF`,
      :meth:`libertem_blobfinder.udf.correlation.run_fastcorrelation` and
      :meth:`libertem_blobfinder.udf.refinement.run_refine` to specify
      a per-frame shift of the zero order position. This allows processing data
      with strong descan error.

.. _`v0-4-1`:

0.4.1
#####

Bug fixes
---------

* Ensure compatibility with numba>=0.50 :pr:`28`

.. _`v0-4-0`:

0.4.0
#####

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3663437.svg
   :target: https://doi.org/10.5281/zenodo.3663437

The first release independent of LiberTEM. Changes relative to LiberTEM version 0.3.0:

Bug fixes
---------

* Fix bounds checking, size and index calculation bugs
  (https://github.com/LiberTEM/LiberTEM/issues/539,
  https://github.com/LiberTEM/LiberTEM/pull/548)

Features
--------

* Access to the correlation pattern for peak finding through
  :meth:`~libertem_blobfinder.common.correlation.get_correlation`
  (https://github.com/LiberTEM/LiberTEM/pull/571)

Misc
----

* The code was extracted from LiberTEM and restructured. See :pr:`1,14,16`
  and :ref:`blobfinder api` for details on the new structure!

Obsolescence
------------

* Importing blobfinder from LiberTEM is deprecated and will only
  be supported until 0.6.0. Import from this new package instead.
