Data
====

Plant CSV files live under ``data/`` and must be named ``Project<ID>.csv``.
The repository includes three benchmark-format example plants: 171, 172, and
186. The default experiment period is 2022-01-01 through 2024-09-28.

The full 100-plant data release is available at
https://doi.org/10.7910/DVN/3VKAGM. The paper notes that the complete release is
larger than 1.5 GB, so the GitHub repository includes only the three example
plants needed for quick verification.

The repository code is MIT licensed. Dataset use is separate: the paper's data
availability statement says the materials are for non-commercial research use
and should be cited together with the paper.

Required columns include ``Year``, ``Month``, ``Day``, ``Hour``, ``Capacity
Factor``, historical weather features, and matching numerical-weather-prediction
columns with the ``_pred`` suffix.

See ``data/README.md`` in the repository root for the complete schema notes.
