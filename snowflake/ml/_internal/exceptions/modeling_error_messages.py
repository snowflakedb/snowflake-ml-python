ATTRIBUTE_NOT_SET = (
    "{} is not set. To read more about Snowpark ML general API differences, please refer to: "
    "https://docs.snowflake.com/en/developer-guide/snowpark-ml/snowpark-ml-modeling#general-api"
    "-differences."
)
SIZE_MISMATCH = "Size mismatch: {}={}, {}={}."
INVALID_MODEL_PARAM = (
    "Invalid parameter {} for model {}. Valid parameters: {}."
    "Note: Scikit learn params cannot be set until the model has been fit."
)
UNSUPPORTED_MODEL_CONVERSION = "Object doesn't support {}. Please use {}."
INCOMPATIBLE_NEW_SKLEARN_PARAM = "Incompatible scikit-learn version: {} requires scikit-learn>={}. Installed: {}."
REMOVED_SKLEARN_PARAM = "Incompatible scikit-learn version: {} is removed in scikit-learn>={}. Installed: {}."
