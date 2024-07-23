estimator_info_list = [
    struct(class_name = "PolynomialFeatures", normalized_class_name = "polynomial_features"),
]

snowpark_pandas_estimator_info_list = estimator_info_list + [
    struct(class_name = "Binarizer", normalized_class_name = "binarizer"),
    struct(class_name = "KBinsDiscretizer", normalized_class_name = "k_bins_discretizer"),
    struct(class_name = "LabelEncoder", normalized_class_name = "label_encoder"),
    struct(class_name = "MaxAbsScaler", normalized_class_name = "max_abs_scaler"),
    struct(class_name = "MinMaxScaler", normalized_class_name = "min_max_scaler"),
    struct(class_name = "Normalizer", normalized_class_name = "normalizer"),
    struct(class_name = "OneHotEncoder", normalized_class_name = "one_hot_encoder"),
    struct(class_name = "OrdinalEncoder", normalized_class_name = "ordinal_encoder"),
    struct(class_name = "RobustScaler", normalized_class_name = "robust_scaler"),
    struct(class_name = "StandardScaler", normalized_class_name = "standard_scaler"),
]
