# Small UI for Model Registry

This `model_registry` SDK is meant to be client-side library only. But for UI lovers, we are including a small
[StreamLit](https://docs.streamlit.io/) app to browse models. It is meant to be a read-only browser and currently has no capabilities to modify the model registry.

## How to run

1. [Install Streamlit](https://docs.streamlit.io/library/get-started/installation)

- As this UI is an optional part of the package, streamlit is not required as a hard dependency. Only using the UI will require the streamlit package to be installed by the user.

2. Run the UI using the following command:
   ```
   streamlit run snowflake/ml/model_registry/ui/streamlit_app.py
   ```
   NOTE: There are two ways to get to the file:
   1. Use the explicit path by `git` cloning `snowml` repo
   2. By installing `snowflake-ml-python` package and running directly using installed file. To get path of the installed package, you can do something like:
      `python -c 'import site; print(site.getsitepackages())'` plus relative path or using
      `python -c 'from snowflake.ml.model_registry.ui import streamlit_app; import os; print(os.path.abspath(streamlit_app.__file__))`.
