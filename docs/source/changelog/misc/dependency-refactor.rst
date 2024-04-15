[Misc] Remove dependency on LiberTEM except for :code:`udf`
===========================================================

* The :mod:`libertem_blobfinder.common` module and associated tests
  have been refactored to remove any dependency on LiberTEM. notably
  this includes all mask-generating functions previously found in
  :mod:`libertem.masks`, which are now found also in
  :mod:`libertem_blobfinder.base.masks`. As part of this change
  the :code:`common` extra dependency group has been removed (:pr:`87`).
