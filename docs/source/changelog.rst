Changelog
=========

.. _continuous:
.. _`v0-5-0`:

0.5.0.dev0 (continuous)
#######################

.. Commented out until first entry is ready
.. .. toctree::
..    :glob:
..
..    changelog/*/*

.. _latest:

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
