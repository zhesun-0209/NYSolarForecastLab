Results
========

Forecast and multi-plant runs write one CSV per plant:

.. code-block:: text

   results_<plant_id>_all.csv

The runner appends rows after each configuration, so existing files can be used
to resume interrupted experiments. Rows with ``status=FAILED`` remain visible in
the CSV for auditability, but they are not counted as completed by
``python run.py status``.

Core Columns
------------

``experiment_name``
   Unique model/input/lookback/time-encoding configuration.

``model`` and ``complexity``
   Forecasting method and low/high complexity setting.

``feature_combo`` or ``scenario``
   Input setting such as PV, PV+HW, PV+NWP, PV+NWP+, NWP, or NWP+.

``mae``, ``rmse``, ``r2``, ``nrmse``
   Test-set metrics on inverse-transformed capacity factor values.

``status`` and ``error``
   Success/failure bookkeeping used by resume and status checks.
