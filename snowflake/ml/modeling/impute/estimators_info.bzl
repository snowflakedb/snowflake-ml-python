estimator_info_list = [
    struct(class_name = "IterativeImputer", normalized_class_name = "iterative_imputer"),
    struct(class_name = "KNNImputer", normalized_class_name = "knn_imputer"),
    struct(class_name = "MissingIndicator", normalized_class_name = "missing_indicator"),
]

snowpark_pandas_estimator_info_list = estimator_info_list + [
    struct(class_name = "SimpleImputer", normalized_class_name = "simple_imputer"),
]
