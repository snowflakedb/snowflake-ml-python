***********************************************************
``snowflake-ml-python`` API Reference
***********************************************************

The ``snowflake-ml-python`` package is a set of tools for creating and working with machine learning models in Python on Snowflake.
See the `Snowflake ML Developer Guide <https://docs.snowflake.com/developer-guide/snowpark-ml/index>`_ for more information.

Additional ML APIs are available in ``snowflake-ml-python``  when running on `Container Runtime <https://docs.snowflake.com/developer-guide/snowflake-ml/container-runtime-ml>`_.

``snowflake-ml-python`` also includes Python APIs for Snowflake Cortex.

Acknowledgements
   Portions of the Snowpark ML API Reference are derived from
   `scikit-learn <https://scikit-learn.org/stable/user_guide.html>`_,
   `xgboost <https://xgboost.readthedocs.io/en/stable/>`_, and
   `lightgbm <https://lightgbm.readthedocs.io/en/stable/>`_ documentation.

   | **scikit-learn** Copyright © 2007-2025 The scikit-learn developers. All rights reserved.
   | **xgboost** Copyright © 2019 by xgboost contributors.
   | **lightgbm** Copyright © Microsoft Corporation.
   |

Container Runtime APIs
=============================

The following APIs are available only in the version of ``snowpark-ml-python`` available in the Container Runtime, accessible in Snowflake Notebooks running on Snowpark Container Services (SPCS).

.. toctree::
   :maxdepth: 2

   distributors
   data_sharded_data_connector

Standard APIs
=============

The following APIs are available in both the Container Runtime and in the standalone client version of ``snowpark-ml-python`` accessible through conda and pip, in Snowsight Python worksheets, and in Snowflake notebooks running on a warehouse.

.. toctree::
   :maxdepth: 3

   index-standard
   index-container
   index-cortex
