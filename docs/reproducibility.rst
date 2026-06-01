Reproducibility
===============

Smoke test:

.. code-block:: bash

   python run.py forecast --plant-id 171 --test-mode --test-model Linear --output-dir results/smoke

Full single-plant benchmark:

.. code-block:: bash

   python run.py forecast --plant-id 171 --output-dir results/plant171

Multi-plant benchmark:

.. code-block:: bash

   python run.py multi_plant --plants 171 172 186 --output-dir results/sample_plants

The experiment runner appends rows after each configuration and can resume from
existing result CSVs.

Each plant result is written as ``results_<plant_id>_all.csv``. Failed rows are
kept for auditability and are not counted as completed by
``python run.py status``.

See ``REPRODUCIBILITY.md`` in the repository root for the detailed command list.
