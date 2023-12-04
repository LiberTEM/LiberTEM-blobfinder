[Feature] Upsampling DFT for peak position refinement
=====================================================
* Fourier upsampling is now implemented for calculating the
  'refineds' peak positions in correlation UDFs. This approach
  can give more precise results when peak shifts are sub-pixel
  at the expense of increased computation, and is available
  using the :code:`upsample=True` argument to UDF and associated
  functions (see :issue:`39`). (:pr:`70`).