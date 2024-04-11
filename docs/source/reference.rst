.. _`blobfinder api`:

Reference
=========

LiberTEM-blobfinder is structured into three parts:

1. A "base" package with numerics functions and utilities.
2. A "common" package providing user-facing funtcions to run
   Blobfinder without any dependency on LiberTEM itself.
3. A "udf" package with classes and functions to use this
   functionality with full LiberTEM integration.

.. _`blobfinder base`:

Basic numerics functions
------------------------

.. automodule:: libertem_blobfinder.base.correlation
   :members: 
   :special-members: __init__

.. automodule:: libertem_blobfinder.base.masks
   :members:

.. _`blobfinder common`:

Common classes and functions
----------------------------

These functions and classes provide Blobfinder's funtcionality, but can
be used without the LiberTEM core infrastructure.

.. automodule:: libertem_blobfinder.common.patterns
   :members: 
   :special-members: __init__

.. automodule:: libertem_blobfinder.common.correlation
   :members: 
   :special-members: __init__

.. _`matching api`:

Matching
~~~~~~~~

These modules contain classes and helper functions that extract and manipulate lattices from correlation results.

.. automodule:: libertem_blobfinder.common.gridmatching
   :members:
   :show-inheritance:

.. automodule:: libertem_blobfinder.common.fullmatch
   :members:
   :show-inheritance:

.. _`blobfinder udf`:

User-defined functions
----------------------

These functions and classes depend on LiberTEM core infrastructure.

Correlation
~~~~~~~~~~~

UDFs and utility functions to find peaks and refine their positions by using
correlation.

.. automodule:: libertem_blobfinder.udf.correlation
   :members:
   :show-inheritance:
   :special-members: __init__

Refinement
~~~~~~~~~~

UDFs and utility functions to refine grid parameters from peak positions.

.. automodule:: libertem_blobfinder.udf.refinement
   :members:
   :show-inheritance:
   :special-members: __init__

Integration
~~~~~~~~~~~

UDFs to integrate peak intensity with positions specified per frame. If the peak
positions are sufficiently similar for all frames, you can use
:meth:`libertem_blobfinder.common.patterns.feature_vector` together with
:class:`libertem.udf.masks.ApplyMasksUDF` instead.

.. automodule:: libertem_blobfinder.udf.integration
   :members:
   :show-inheritance:
   :special-members: __init__

Utilities
~~~~~~~~~

General utility functions.

.. automodule:: libertem_blobfinder.udf.utils
   :members:
   :show-inheritance:
   :special-members: __init__
