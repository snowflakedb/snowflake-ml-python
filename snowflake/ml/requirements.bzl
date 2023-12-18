# DO NOT EDIT!
# Generate by running 'bazel run --config=pre_build //bazel/requirements:sync_requirements'

EXTRA_REQUIREMENTS = {
    "all": [
        "lightgbm==3.3.5",
        "mlflow>=2.1.0,<2.4",
        "peft>=0.5.0,<1",
        "sentencepiece>=0.1.95,<0.2",
        "shap==0.42.1",
        "tensorflow>=2.9,<3,!=2.12.0",
        "tokenizers>=0.10,<1",
        "torchdata>=0.4,<1",
        "transformers>=4.32.1,<5"
    ],
    "lightgbm": [
        "lightgbm==3.3.5"
    ],
    "llm": [
        "peft>=0.5.0,<1"
    ],
    "mlflow": [
        "mlflow>=2.1.0,<2.4"
    ],
    "shap": [
        "shap==0.42.1"
    ],
    "tensorflow": [
        "tensorflow>=2.9,<3,!=2.12.0"
    ],
    "torch": [
        "torchdata>=0.4,<1"
    ],
    "transformers": [
        "sentencepiece>=0.1.95,<0.2",
        "tokenizers>=0.10,<1",
        "transformers>=4.32.1,<5"
    ]
}

REQUIREMENTS = [
    "absl-py>=0.15,<2",
    "anyio>=3.5.0,<4",
    "cachetools>=3.1.1,<5",
    "cloudpickle>=2.0.0",
    "fsspec[http]>=2022.11,<2024",
    "importlib_resources>=5.1.4, <6",
    "numpy>=1.23,<2",
    "packaging>=20.9,<24",
    "pandas>=1.0.0,<2",
    "pyarrow",
    "pytimeparse>=1.1.8,<2",
    "pyyaml>=6.0,<7",
    "retrying>=1.3.3,<2",
    "s3fs>=2022.11,<2024",
    "scikit-learn>=1.2.1,<1.4",
    "scipy>=1.9,<2",
    "snowflake-connector-python[pandas]>=3.0.4,<4",
    "snowflake-snowpark-python>=1.8.0,<2",
    "sqlparse>=0.4,<1",
    "typing-extensions>=4.1.0,<5",
    "xgboost>=1.7.3,<2"
]
