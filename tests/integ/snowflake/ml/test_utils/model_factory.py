from enum import Enum
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
import torch
from sklearn import datasets, svm

from snowflake.ml.model import custom_model
from snowflake.ml.modeling.linear_model import (  # type: ignore[attr-defined]
    LogisticRegression,
)
from snowflake.ml.modeling.pipeline import Pipeline  # type: ignore[attr-defined]
from snowflake.ml.modeling.preprocessing import (  # type: ignore[attr-defined]
    MinMaxScaler,
    OneHotEncoder,
)
from snowflake.ml.modeling.xgboost import XGBClassifier  # type: ignore[attr-defined]
from snowflake.snowpark import DataFrame, Session


class DEVICE(Enum):
    CUDA = "cuda"
    CPU = "cpu"


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
    def prepare_snowml_model_xgb() -> Tuple[XGBClassifier, pd.DataFrame, pd.DataFrame]:
        """Prepare SnowML XGBClassifier model.

        Returns:
            a XGB classifier.
            a dataframe of test features.
            a dataframe of training dataset.
        """
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

        return (clf_xgb, df.drop(columns=label_cols).head(10), df)

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
    def prepare_gpt2_model(local_cache_dir: Optional[str] = None) -> Tuple[custom_model.CustomModel, pd.DataFrame]:
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

    @staticmethod
    def prepare_torch_model(
        dtype: torch.dtype = torch.float32, force_remote_gpu_inference: bool = False
    ) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
        class TorchModel(torch.nn.Module):
            def __init__(self, n_input: int, n_hidden: int, n_out: int, dtype: torch.dtype = torch.float32) -> None:
                super().__init__()
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(n_input, n_hidden, dtype=dtype),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden, n_out, dtype=dtype),
                    torch.nn.Sigmoid(),
                )

            def forward_training(self, tensor: torch.Tensor) -> torch.Tensor:
                return cast(torch.Tensor, self.model(tensor))

            def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                device = DEVICE.CUDA if force_remote_gpu_inference else DEVICE.CPU
                return self.predict_with_device(tensor, device)

            def predict_with_device(self, tensor: torch.Tensor, device: DEVICE) -> torch.Tensor:
                self.model.eval()
                self.model.to(device.value)
                with torch.no_grad():
                    tensor = tensor.to(device.value)
                    return cast(torch.Tensor, self.model(tensor))

        n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
        x = np.random.rand(batch_size, n_input)
        data_x = torch.from_numpy(x).to(dtype=dtype)
        data_y = (torch.rand(size=(batch_size, 1)) < 0.5).to(dtype=dtype)

        model = TorchModel(n_input, n_hidden, n_out, dtype=dtype)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        for _epoch in range(100):
            pred_y = model.forward_training(data_x)
            loss = loss_function(pred_y, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model, data_x, data_y

    @staticmethod
    def prepare_jittable_torch_model(
        dtype: torch.dtype = torch.float32, force_remote_gpu_inference: bool = False
    ) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
        class TorchModel(torch.nn.Module):
            def __init__(self, n_input: int, n_hidden: int, n_out: int, dtype: torch.dtype = torch.float32) -> None:
                super().__init__()
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(n_input, n_hidden, dtype=dtype),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden, n_out, dtype=dtype),
                    torch.nn.Sigmoid(),
                )

            def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                return self.model(tensor)  # type: ignore[no-any-return]

        n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
        x = np.random.rand(batch_size, n_input)
        data_x = torch.from_numpy(x).to(dtype=dtype)
        data_y = (torch.rand(size=(batch_size, 1)) < 0.5).to(dtype=dtype)

        model = TorchModel(n_input, n_hidden, n_out, dtype=dtype)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        for _epoch in range(100):
            pred_y = model(data_x)
            loss = loss_function(pred_y, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model, data_x, data_y

    @staticmethod
    def prepare_keras_model(
        dtype: tf.dtypes.DType = tf.float32,
    ) -> Tuple[tf.keras.Model, tf.Tensor, tf.Tensor]:
        class KerasModel(tf.keras.Model):
            def __init__(self, n_hidden: int, n_out: int) -> None:
                super().__init__()
                self.fc_1 = tf.keras.layers.Dense(n_hidden, activation="relu")
                self.fc_2 = tf.keras.layers.Dense(n_out, activation="sigmoid")

            def call(self, tensor: tf.Tensor) -> tf.Tensor:
                input = tensor
                x = self.fc_1(input)
                x = self.fc_2(x)
                return x

        n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
        x = np.random.rand(batch_size, n_input)
        data_x = tf.convert_to_tensor(x, dtype=dtype)
        raw_data_y = tf.random.uniform((batch_size, 1))
        raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
        data_y = tf.cast(raw_data_y, dtype=dtype)

        model = KerasModel(n_hidden, n_out)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError()
        )
        model.fit(data_x, data_y, batch_size=batch_size, epochs=100)
        return model, data_x, data_y
