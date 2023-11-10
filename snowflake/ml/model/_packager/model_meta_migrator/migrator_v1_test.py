import os
import tempfile

from absl.testing import absltest

from snowflake.ml.model._packager.model_meta import model_meta

YAML_1 = """
conda_dependencies:
- absl-py==1.3.0
- anyio==3.5.0
- cloudpickle==2.0.0
- numpy==1.24.3
- packaging==23.0
- pandas==1.5.3
- pytorch==2.0.1
- pyyaml==6.0
- snowflake-snowpark-python==1.5.1
- tokenizers==0.13.2
- transformers==4.29.2
- typing-extensions==4.5.0
creation_timestamp: '2023-09-21 18:12:39.409911'
cuda_version: '11.7'
local_ml_library_version: 1.0.9+df2e394bae177167b9d9a8becc792ed899f3432d
metadata: null
model_type: huggingface_pipeline
models:
  llama-2-7b-chat:
    artifacts: {}
    model_type: huggingface_pipeline
    name: llama-2-7b-chat
    options:
      batch_size: '1'
      task: text-generation
    path: model
name: llama-2-7b-chat
pip_requirements: []
python_version: 3.8.13
signatures:
  __call__:
    inputs:
    - name: inputs
      type: STRING
    outputs:
    - name: outputs
      type: STRING
version: 1
"""

CONDA_FILE = """
channels:
- https://repo.anaconda.com/pkgs/snowflake
- nodefaults
dependencies:
- python==3.8.13
- absl-py==1.3.0
- anyio==3.5.0
- cloudpickle==2.0.0
- numpy==1.24.3
- packaging==23.0
- pandas==1.5.3
- pyyaml==6.0
- snowflake-snowpark-python==1.5.1
- typing-extensions==4.5.0
- transformers==4.29.2
- tokenizers==0.13.2
- pytorch==2.0.1
name: snow-env
"""

YAML_2 = """
conda_dependencies:
- absl-py==1.3.0
- anyio==3.5.0
- cloudpickle==2.0.0
- numpy==1.24.3
- packaging==23.0
- pandas==1.5.3
- pytorch==2.0.1
- pyyaml==6.0
- snowflake-ml-python==1.0.9
- snowflake-snowpark-python==1.5.1
- tokenizers==0.13.2
- transformers==4.29.2
- typing-extensions==4.5.0
creation_timestamp: '2023-09-21 18:12:39.409911'
cuda_version: '11.7'
metadata: null
model_type: huggingface_pipeline
models:
  llama-2-7b-chat:
    artifacts: {}
    model_type: huggingface_pipeline
    name: llama-2-7b-chat
    options:
      batch_size: '1'
      task: text-generation
    path: model
name: llama-2-7b-chat
pip_requirements: []
python_version: 3.8.13
signatures:
  __call__:
    inputs:
    - name: inputs
      type: STRING
    outputs:
    - name: outputs
      type: STRING
version: 1
"""


class MigratorV1Test(absltest.TestCase):
    def test_yaml_load(self) -> None:
        for yaml_str in [YAML_1, YAML_2]:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, model_meta.MODEL_METADATA_FILE), "w", encoding="utf-8") as f:
                    f.write(yaml_str)

                os.makedirs(os.path.join(tmpdir, "env"), exist_ok=True)
                with open(os.path.join(tmpdir, "env", "conda.yaml"), "w", encoding="utf-8") as f:
                    f.write(CONDA_FILE)

                with open(os.path.join(tmpdir, "env", "requirements.txt"), "w", encoding="utf-8") as f:
                    f.write("")

                model_meta.ModelMetadata.load(tmpdir)


if __name__ == "__main__":
    absltest.main()
