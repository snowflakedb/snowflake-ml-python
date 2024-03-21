import os
import random
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_handlers.sentence_transformers import (
    _sentence_transformer_encode,
)
from snowflake.ml.model._signatures import (
    snowpark_handler,
    utils as model_signature_utils,
)
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.model import warehouse_model_integ_test_utils
from tests.integ.snowflake.ml.test_utils import dataframe_utils, db_manager

MODEL_NAMES = ["intfloat/e5-base-v2"]  # cant load models in parallel
SENTENCE_TRANSFORMERS_CACHE_DIR = "SENTENCE_TRANSFORMERS_HOME"


class TestWarehouseSentenceTransformerInteg(parameterized.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.cleanup_schemas()
        self._db_manager.cleanup_stages()
        self._db_manager.cleanup_user_functions()

        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_cache_dir = os.getenv(SENTENCE_TRANSFORMERS_CACHE_DIR, None)
        os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = self.cache_dir.name

        # To create different UDF names among different runs
        self.run_id = uuid.uuid4().hex
        self._test_schema_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "model_deployment_sentence_transformers_model_test_schema"
        )
        self._db_manager.create_schema(self._test_schema_name)
        self._db_manager.use_schema(self._test_schema_name)

        self.deploy_stage_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "deployment_stage"
        )
        self.full_qual_stage = self._db_manager.create_stage(
            self.deploy_stage_name,
            schema_name=self._test_schema_name,
            sse_encrypted=False,
        )

    @classmethod
    def tearDownClass(self) -> None:
        self._db_manager.drop_stage(self.deploy_stage_name, schema_name=self._test_schema_name)
        self._db_manager.drop_schema(self._test_schema_name)
        self._session.close()
        if self._original_cache_dir:
            os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = self._original_cache_dir
        self.cache_dir.cleanup()

    def base_test_case(
        self,
        name: str,
        model: model_types.SupportedModelType,
        sample_input_data: model_types.SupportedModelType,
        test_input: model_types.SupportedDataType,
        deploy_params: Dict[
            str,
            Tuple[Dict[str, Any], Callable[[Union[pd.DataFrame, SnowparkDataFrame]], Any]],
        ],
        permanent_deploy: Optional[bool] = False,
        additional_dependencies: Optional[List[str]] = None,
    ) -> None:
        warehouse_model_integ_test_utils.base_test_case(
            self._db_manager,
            run_id=self.run_id,
            full_qual_stage=self.full_qual_stage,
            name=name,
            model=model,
            sample_input_data=sample_input_data,
            test_input=test_input,
            deploy_params=deploy_params,
            permanent_deploy=permanent_deploy,
            additional_dependencies=additional_dependencies,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_sentence_transformers(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        # We have to import here due to cache location issue.
        # Only by doing so can we make the cache dir setting effective.
        import sentence_transformers

        # Sample Data
        sentences = pd.DataFrame(
            {
                "SENTENCES": [
                    "Why don’t scientists trust atoms? Because they make up everything.",
                    "I told my wife she should embrace her mistakes. She gave me a hug.",
                    "Im reading a book on anti-gravity. Its impossible to put down!",
                    "Did you hear about the mathematician who’s afraid of negative numbers?",
                    "Parallel lines have so much in common. It’s a shame they’ll never meet.",
                ]
            }
        )
        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))
        embeddings = _sentence_transformer_encode(model, sentences)
        sig = {"encode": model_signature.infer_signature(sentences, embeddings)}
        embeddings = model_signature_utils.rename_pandas_df(embeddings, sig["encode"].outputs)

        self.base_test_case(
            name="sentence_transformers_model",
            model=model,
            sample_input_data=sentences,
            test_input=sentences,
            deploy_params={
                "encode": (
                    {},
                    lambda res: res.equals(embeddings),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_sentence_transformers_sp(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        # We have to import here due to cache location issue.
        # Only by doing so can we make the cache dir setting effective.
        import sentence_transformers

        # Sample Data
        sentences = pd.DataFrame(
            {
                "SENTENCES": [
                    "Why don’t scientists trust atoms? Because they make up everything.",
                    "I told my wife she should embrace her mistakes. She gave me a hug.",
                    "Im reading a book on anti-gravity. Its impossible to put down!",
                    "Did you hear about the mathematician who’s afraid of negative numbers?",
                    "Parallel lines have so much in common. It’s a shame they’ll never meet.",
                ]
            }
        )
        sentences_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, sentences)
        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))
        embeddings = _sentence_transformer_encode(model, sentences)
        sig = {"encode": model_signature.infer_signature(sentences, embeddings)}
        embeddings = model_signature_utils.rename_pandas_df(embeddings, sig["encode"].outputs)
        y_df_expected = pd.concat([sentences_sp.to_pandas(), embeddings], axis=1)

        self.base_test_case(
            name="sentence_transformers_model",
            model=model,
            sample_input_data=sentences,
            test_input=sentences_sp,
            deploy_params={
                "encode": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, atol=1e-6),
                ),
            },
            permanent_deploy=permanent_deploy,
        )


if __name__ == "__main__":
    absltest.main()
