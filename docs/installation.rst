Installation
============

Requirements
------------

- Python 3.10 or higher
- PyTorch 1.12.0 or higher
- NumPy, Pandas, Scikit-learn
- XGBoost, LightGBM (for ML models)

Installation Steps
------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/zhesun-0209/NYSolarForecastLab.git
   cd NYSolarForecastLab

2. Install dependencies:

.. code-block:: bash

   python -m pip install -r requirements.txt

For documentation and test tooling:

.. code-block:: bash

   python -m pip install -r requirements-dev.txt

Optional editable package install:

.. code-block:: bash

   python -m pip install -e .

3. Verify installation:

.. code-block:: bash

   python run.py --help
