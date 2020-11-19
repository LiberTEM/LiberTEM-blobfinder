[Feature] Precise peak integration
==================================

* Integration of peak intensities at per-frame positions using
  :class:`libertem_blobfinder.udf.integration.IntegrationUDF`. This can be used
  if peaks are shifted so much that integration at equal positions for all
  frames using :meth:`libertem_blobfinder.common.patterns.feature_vector`
  doesn't work anymore. (:pr:`27`)
