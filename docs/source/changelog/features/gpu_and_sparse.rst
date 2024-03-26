[Feature] GPU and Sparse backend support
========================================
* All correlation UDFs now support GPU processing
  with :code:`cupy`, in addition to sparse input
  with the Fast and Sparse correlation UDFs. Conversion
  from unsupported backends is automatically handled
  using the `sparseconverter <https://github.com/LiberTEM/sparseconverter>`_
  package. (see :pr:`61`)