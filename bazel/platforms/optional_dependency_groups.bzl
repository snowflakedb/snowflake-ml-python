# Order matters: py_rules.bzl routes a target to the FIRST group whose package
# set is a superset of the target's optional_dependencies. "ml" is listed first
# so scikit-learn-only targets resolve to the "ml" env (which also carries
# xgboost/lightgbm/catboost/mlflow) instead of being captured by "keras"/"llm",
# which include scikit-learn only because the shared modeling test utilities
# import it transitively. The @unsorted-dict-items directive keeps buildifier
# from alphabetizing these entries and reintroducing that bug.
OPTIONAL_DEPENDENCY_GROUPS = {
    # @unsorted-dict-items
    "ml": ["lightgbm", "catboost", "mlflow", "altair", "streamlit", "prophet", "shap", "scikit-learn", "xgboost"],
    "keras": ["torch", "tensorflow", "keras", "scikit-learn"],
    "llm": ["llm", "scikit-learn"],
    "torch": ["torch", "transformers", "mlflow", "scikit-learn", "xgboost"],
}
