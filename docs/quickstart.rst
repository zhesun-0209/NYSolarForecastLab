Quick Start
===========

Basic Usage
-----------

1. Install the dependencies and enter the repository root.

2. Generate configurations:

.. code-block:: bash

   python run.py config

3. Run a fast smoke test:

.. code-block:: bash

   python run.py forecast --plant-id 171 --test-mode --test-model Linear --output-dir results/smoke
   python run.py status --output-dir results/smoke

The smoke run writes ``results/smoke/results_171_all.csv`` and should contain
two successful Linear Regression rows.

4. Run the full grid for one plant:

.. code-block:: bash

   python run.py forecast --plant-id 171 --output-dir results/plant171

For the full paper-scale release, place additional ``Project<ID>.csv`` files in
``data/``, regenerate configs, and use ``python run.py multi_plant``.

Optional visual check:

.. code-block:: bash

   python examples/plot_monthly_generation.py
