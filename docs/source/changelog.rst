Changelog
=========

.. _continuous:
.. _`v0-6-0`:

0.6.0.dev0 (continuous)
#######################

.. toctree::
   :glob:

   changelog/*/*

.. _latest:
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
