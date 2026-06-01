# Configuration file for the Sphinx documentation builder.

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

project = 'NYSolarForecastLab'
copyright = '2026, NYSolarForecastLab contributors'
author = 'Zhaoyao Bao, Zhe Sun, Yishuo Jiang, Chi Xie, Lijun Sun, and H. Oliver Gao'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
