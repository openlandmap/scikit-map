LUCAS module
============

The module ``pyeumap.lucas`` allows accessing features from the
LUCAS dataset.

Define request
--------------

Example of usage:

.. code-block:: python
          
   request = LucasRequest()
   request.bbox = (4472010, 2838000, 4960000, 3112000)

.. automodule:: pyeumap.lucas.request
    :members:
   
Download features based on request
----------------------------------

Example of usage:

.. code-block:: python

   io = LucasIO()
   io.download(request)
   io.to_gpkg("lucas.gpkg")

.. automodule:: pyeumap.lucas.io
    :members:

LC class aggregation
--------------------

.. automodule:: pyeumap.lucas.analyze
    :members:
