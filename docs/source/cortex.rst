===================
snowflake.ml.cortex
===================

.. automodule:: snowflake.cortex
    :noindex:

.. currentmodule:: snowflake.cortex

.. rubric:: Classes

.. autosummary::
    :toctree: api/cortex

    CompleteOptions
    Finetune
    FinetuneJob
    FinetuneStatus

.. rubric:: Functions

.. note::

    Functions in this module are also available through "CamelCase" names (for example, ``ClassifyText`` is the same as
    ``classify_text``). These names are deprecated as of ``snowflake-ml-python`` version 1.7.3, and will be removed in
    a future release. Use the "snake_case" names shown here.

.. autosummary::
    :toctree: api/cortex

    classify_text
    complete
    embed_text_1024
    embed_text_768
    extract_answer
    sentiment
    summarize
    translate
