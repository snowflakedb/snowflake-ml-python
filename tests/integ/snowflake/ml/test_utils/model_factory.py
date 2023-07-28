#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import datasets, svm

from snowflake.ml.model import custom_model
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import MinMaxScaler, OneHotEncoder
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.snowpark import DataFrame, Session


class ModelFactory:
    @staticmethod
    def prepare_sklearn_model() -> Tuple[svm.SVC, npt.ArrayLike, npt.ArrayLike]:
        digits = datasets.load_digits()
        target_digit = 6
        num_training_examples = 10
        svc_gamma = 0.001
        svc_C = 10.0

        clf = svm.SVC(gamma=svc_gamma, C=svc_C, probability=True)

        def one_vs_all(dataset: npt.NDArray[np.float64], digit: int) -> List[bool]:
            return [x == digit for x in dataset]

        # Train a classifier using num_training_examples and use the last 100 examples for test.
        train_features = digits.data[:num_training_examples]
        train_labels = one_vs_all(digits.target[:num_training_examples], target_digit)
        clf.fit(train_features, train_labels)

        test_features = digits.data[-100:]
        test_labels = one_vs_all(digits.target[-100:], target_digit)

        return clf, test_features, test_labels

    @staticmethod
    def prepare_snowml_model() -> Tuple[XGBClassifier, pd.DataFrame]:
        iris = datasets.load_iris()
        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        input_cols = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        label_cols = "TARGET"
        output_cols = "PREDICTED_TARGET"

        clf_xgb = XGBClassifier(
            input_cols=input_cols, output_cols=output_cols, label_cols=label_cols, drop_input_cols=True
        )

        clf_xgb.fit(df)

        return clf_xgb, df.drop(columns=label_cols).head(10)

    @staticmethod
    def prepare_snowml_pipeline(session: Session) -> Tuple[Pipeline, DataFrame]:
        iris = datasets.load_iris()
        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        def add_simple_category(df: pd.DataFrame) -> pd.DataFrame:
            bins = (-1, 4, 5, 6, 10)
            group_names = ["Unknown", "1_quartile", "2_quartile", "3_quartile"]
            categories = pd.cut(df.SEPALLENGTH, bins, labels=group_names)
            df["SIMPLE"] = categories
            return df

        df_cat = add_simple_category(df)
        iris_df = session.create_dataframe(df_cat)

        numeric_features = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        categorical_features = ["SIMPLE"]
        numeric_features_output = [x + "_O" for x in numeric_features]
        label_cols = "TARGET"

        pipeline = Pipeline(
            steps=[
                (
                    "OHEHOT",
                    OneHotEncoder(input_cols=categorical_features, output_cols="cat_output", drop_input_cols=True),
                ),
                (
                    "SCALER",
                    MinMaxScaler(
                        clip=True,
                        input_cols=numeric_features,
                        output_cols=numeric_features_output,
                        drop_input_cols=True,
                    ),
                ),
                # TODO: Remove drop_input_cols=True after SNOW-853632 gets fixed.
                ("CLASSIFIER", LogisticRegression(label_cols=label_cols, drop_input_cols=True)),
            ]
        )
        pipeline.fit(iris_df)

        return pipeline, iris_df.drop(label_cols).limit(10)

    @staticmethod
    def prepare_snowml_model_logistic() -> Tuple[LogisticRegression, pd.DataFrame]:
        iris = datasets.load_iris()
        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        input_cols = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        label_cols = "TARGET"
        output_cols = "PREDICTED_TARGET"

        estimator = LogisticRegression(
            input_cols=input_cols, output_cols=output_cols, label_cols=label_cols, random_state=0, max_iter=100
        ).fit(df)

        return estimator, df.drop(columns=label_cols).head(10)

    @staticmethod
    def prepare_gpt2_model(local_cache_dir: str = None) -> Tuple[custom_model.CustomModel, pd.DataFrame]:
        """
        Pretrained GPT2 model from huggingface.
        """
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=local_cache_dir)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=local_cache_dir)

        class HuggingFaceModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
                torch_device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(torch_device)
                tokenizer.padding_side = "left"

                # Define PAD Token = EOS Token = 50256
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

                prompts = input_df.values.flatten().tolist()
                inputs = tokenizer(prompts, return_tensors="pt", padding=True)
                torch.manual_seed(0)
                outputs = model.generate(
                    input_ids=inputs["input_ids"].to(torch_device),
                    attention_mask=inputs["attention_mask"].to(torch_device),
                )
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                return pd.DataFrame({"output": generated_texts})

        gpt2_model = HuggingFaceModel(custom_model.ModelContext())
        test_data = pd.DataFrame(["Hello, how are you?", "Once upon a time"])

        return gpt2_model, test_data
