import uuid
from typing import Callable, Tuple

import numpy as np
from absl.testing import absltest

from snowflake.ml.registry import Registry
from snowflake.snowpark import session
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


class RegistryTensorflowCompatTest(common_test_base.CommonTestBase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()
        self._run_id = uuid.uuid4().hex
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _prepare_registry_and_log_model_fn_factory_tf_keras_subclass(
        self,
    ) -> Tuple[Callable[[session.Session, str, str, str], None], Tuple[str, str, str]]:
        def prepare_registry_and_log_model(
            session: session.Session, test_db: str, test_schema: str, run_id: str
        ) -> None:
            import numpy as np
            import tensorflow as tf

            from snowflake.ml.registry import Registry

            registry = Registry(session=session, database_name=test_db, schema_name=test_schema)

            class KerasModel(tf.keras.Model):
                def __init__(self, n_hidden: int, n_out: int) -> None:
                    super().__init__()
                    self.fc_1 = tf.keras.layers.Dense(n_hidden, activation="relu")
                    self.fc_2 = tf.keras.layers.Dense(n_out, activation="sigmoid")

                def call(self, tensors: tf.Tensor) -> tf.Tensor:
                    input = tensors
                    x = self.fc_1(input)
                    x = self.fc_2(x)
                    return x

            n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
            x = np.random.rand(batch_size, n_input)
            data_x = tf.convert_to_tensor(x, dtype=tf.float32)
            raw_data_y = tf.random.uniform((batch_size, 1))
            raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
            data_y = tf.cast(raw_data_y, dtype=tf.float32)

            model = KerasModel(n_hidden, n_out)
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss=tf.keras.losses.MeanSquaredError(),
            )
            model.fit(data_x, data_y, batch_size=batch_size, epochs=100)

            registry.log_model(
                model_name="model",
                version_name="v" + run_id,
                model=model,
                sample_input_data=[data_x],
                conda_dependencies=["tensorflow==2.12.0", "wrapt==1.14.1"],
                options={"relax_version": False},
            )

        return prepare_registry_and_log_model, (self._test_db, self._test_schema, self._run_id)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_prepare_registry_and_log_model_fn_factory_tf_keras_subclass,  # type: ignore[arg-type]
        version_range=">=1.7.0",
        additional_packages=["tensorflow==2.12.0", "wrapt==1.14.1"],
    )
    def test_log_model_compat_tf_keras_subclass(self) -> None:
        registry = Registry(session=self.session, database_name=self._test_db, schema_name=self._test_schema)
        model_ref = registry.get_model("model").version("v" + self._run_id)
        x = np.random.rand(100, 10)
        try:
            model_ref.load().predict(x)
        except ValueError:
            model_ref.load(force=True).predict(x)

    def _prepare_registry_and_log_model_fn_factory_tf_keras_sequential(
        self,
    ) -> Tuple[Callable[[session.Session, str, str, str], None], Tuple[str, str, str]]:
        def prepare_registry_and_log_model(
            session: session.Session, test_db: str, test_schema: str, run_id: str
        ) -> None:
            import numpy as np
            import tensorflow as tf

            from snowflake.ml.registry import Registry

            registry = Registry(session=session, database_name=test_db, schema_name=test_schema)

            n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
            x = np.random.rand(batch_size, n_input)
            data_x = tf.convert_to_tensor(x, dtype=tf.float32)
            raw_data_y = tf.random.uniform((batch_size, 1))
            raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
            data_y = tf.cast(raw_data_y, dtype=tf.float32)

            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(n_hidden, activation="relu"),
                    tf.keras.layers.Dense(n_out, activation="sigmoid"),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError()
            )
            model.fit(data_x, data_y, batch_size=batch_size, epochs=100)

            registry.log_model(
                model_name="model",
                version_name="v" + run_id,
                model=model,
                sample_input_data=[data_x],
                conda_dependencies=["tensorflow==2.12.0", "wrapt==1.14.1"],
                options={"relax_version": False},
            )

        return prepare_registry_and_log_model, (self._test_db, self._test_schema, self._run_id)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_prepare_registry_and_log_model_fn_factory_tf_keras_sequential,  # type: ignore[arg-type]
        version_range=">=1.7.0",
        additional_packages=["tensorflow==2.12.0", "wrapt==1.14.1"],
    )
    def test_log_model_compat_tf_keras_sequential(self) -> None:
        registry = Registry(session=self.session, database_name=self._test_db, schema_name=self._test_schema)
        model_ref = registry.get_model("model").version("v" + self._run_id)
        x = np.random.rand(100, 10)
        try:
            model_ref.load().predict(x)
        except ValueError:
            model_ref.load(force=True).predict(x)

    def _prepare_registry_and_log_model_fn_factory_tf_keras_functional(
        self,
    ) -> Tuple[Callable[[session.Session, str, str, str], None], Tuple[str, str, str]]:
        def prepare_registry_and_log_model(
            session: session.Session, test_db: str, test_schema: str, run_id: str
        ) -> None:
            import numpy as np
            import tensorflow as tf

            from snowflake.ml.registry import Registry

            registry = Registry(session=session, database_name=test_db, schema_name=test_schema)

            n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
            x = np.random.rand(batch_size, n_input)
            data_x = tf.convert_to_tensor(x, dtype=tf.float32)
            raw_data_y = tf.random.uniform((batch_size, 1))
            raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
            data_y = tf.cast(raw_data_y, dtype=tf.float32)

            input = tf.keras.Input(shape=(n_input,))
            x = tf.keras.layers.Dense(n_hidden, activation="relu")(input)
            output = tf.keras.layers.Dense(n_out, activation="sigmoid")(x)
            model = tf.keras.Model(inputs=input, outputs=output)
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError()
            )
            model.fit(data_x, data_y, batch_size=batch_size, epochs=100)

            registry.log_model(
                model_name="model",
                version_name="v" + run_id,
                model=model,
                sample_input_data=[data_x],
                conda_dependencies=["tensorflow==2.12.0", "wrapt==1.14.1"],
                options={"relax_version": False},
            )

        return prepare_registry_and_log_model, (self._test_db, self._test_schema, self._run_id)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_prepare_registry_and_log_model_fn_factory_tf_keras_functional,  # type: ignore[arg-type]
        version_range=">=1.7.0",
        additional_packages=["tensorflow==2.12.0", "wrapt==1.14.1"],
    )
    def test_log_model_compat_tf_keras_functional(self) -> None:
        registry = Registry(session=self.session, database_name=self._test_db, schema_name=self._test_schema)
        model_ref = registry.get_model("model").version("v" + self._run_id)
        x = np.random.rand(100, 10)
        try:
            model_ref.load().predict(x)
        except ValueError:
            model_ref.load(force=True).predict(x)


if __name__ == "__main__":
    absltest.main()
