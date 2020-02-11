.. _`installation`:

Installation
============

.. note::
    See also `general LiberTEM installation instructions
    <https://libertem.github.io/LiberTEM/install.html>`_ for information on
    setting up an environment for LiberTEM.

LiberTEM-blobfinder is designed to be installable without the main LiberTEM
package to use some of the functionality in other projects. Installing with

.. code-block:: shell

    $ pip install libertem-blobfinder

only installs a minimum set of dependencies for :mod:`libertem_blobfinder.base`.
Optional dependencies are managed through extras. The full set of dependencies
can be installed with

.. code-block:: shell

    $ pip install libertem-blobfinder[pyfftw,common,udf]

The extra :code:`common` installs dependencies for
:mod:`libertem_blobfinder.common`, and :code:`udf` for
:mod:`libertem_blobfinder.udf`. LiberTEM-blobfinder can optionally use PyFFTW
instead of the built-in NumPy FFT for fast Fourier transforms.

Please note that this package is part of a `larger restructuring effort for
LiberTEM <https://github.com/LiberTEM/LiberTEM/issues/261>`_. That means changes
in the set of dependencies and import locations can be expected.

Installing from a Git clone
---------------------------

For using the latest features or development work, you can install from a Git clone:

.. code-block:: shell

    $ git clone https://github.com/LiberTEM/LiberTEM-blobfinder
    $ cd LiberTEM-blobfinder
    $ pip install -e .
