# TODO[shchen], SNOW-889081, re-enable once server-side image build is supported.
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
#
# import functools
# import tempfile
# import uuid
#
# import numpy as np
# import pandas as pd
# import pytest
# import torch
from absl.testing import absltest

#
# from snowflake.ml.model import deploy_platforms
# from snowflake.ml.model._signatures import pytorch_handler, tensorflow_handler
# from tests.integ.snowflake.ml.registry.model_registry_integ_test_snowservice_base import (
#     TestModelRegistryIntegSnowServiceBase,
# )
# from tests.integ.snowflake.ml.test_utils import model_factory
#
#
# class TestModelRegistryIntegWithSnowServiceDeployment(TestModelRegistryIntegSnowServiceBase):
#     @pytest.mark.pip_incompatible
#     def test_sklearn_deployment_with_snowml_conda(self) -> None:
#         self._test_snowservice_deployment(
#             model_name="test_sklearn_model_with_snowml_conda",
#             model_version=uuid.uuid4().hex,
#             prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_sklearn_model,
#             embed_local_ml_library=False,
#             conda_dependencies=["snowflake-ml-python==1.0.2"],
#             prediction_assert_fn=lambda local_prediction, remote_prediction: np.testing.assert_allclose(
#                 remote_prediction.to_numpy(), np.expand_dims(local_prediction, axis=1)
#             ),
#             deployment_options={
#                 "platform": deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
#                 "target_method": "predict",
#                 "options": {
#                     "compute_pool": self._TEST_CPU_COMPUTE_POOL,
#                     "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
#                     "num_workers": 1,
#                 },
#             },
#         )
#
#     @pytest.mark.pip_incompatible
#     def test_sklearn_deployment_with_local_source_code(self) -> None:
#         self._test_snowservice_deployment(
#             model_name="test_sklearn_model_with_local_source_code",
#             model_version=uuid.uuid4().hex,
#             prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_sklearn_model,
#             prediction_assert_fn=lambda local_prediction, remote_prediction: np.testing.assert_allclose(
#                 remote_prediction.to_numpy(), np.expand_dims(local_prediction, axis=1)
#             ),
#             deployment_options={
#                 "platform": deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
#                 "target_method": "predict",
#                 "options": {
#                     "compute_pool": self._TEST_CPU_COMPUTE_POOL,
#                     "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
#                 },
#             },
#         )
#
#     @pytest.mark.pip_incompatible
#     def test_huggingface_custom_model_deployment(self) -> None:
#         with tempfile.TemporaryDirectory() as tmpdir:
#             self._test_snowservice_deployment(
#                 model_name="gpt2_model_gpu",
#                 model_version=uuid.uuid4().hex,
#                 conda_dependencies=["pytorch", "transformers"],
#                 prepare_model_and_feature_fn=functools.partial(
#                     model_factory.ModelFactory.prepare_gpt2_model,
#                     local_cache_dir=tmpdir,
#                 ),
#                 prediction_assert_fn=lambda local_prediction, remote_prediction: pd.testing.assert_frame_equal(
#                     remote_prediction, local_prediction, check_dtype=False
#                 ),
#                 deployment_options={
#                     "platform": deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
#                     "target_method": "predict",
#                     "options": {
#                         "compute_pool": self._TEST_CPU_COMPUTE_POOL,
#                         "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
#                         "num_workers": 1,
#                     },
#                 },
#             )
#
#     @pytest.mark.pip_incompatible
#     def test_torch_model_deployment_with_gpu(self) -> None:
#         self._test_snowservice_deployment(
#             model_name="torch_model",
#             model_version=uuid.uuid4().hex,
#             prepare_model_and_feature_fn=functools.partial(
#                 model_factory.ModelFactory.prepare_torch_model, force_remote_gpu_inference=True
#             ),
#             conda_dependencies=[
#                 "pytorch-nightly::pytorch",
#                 "pytorch-nightly::pytorch-cuda==12.1",
#                 "nvidia::cuda==12.1.*",
#             ],
#             prediction_assert_fn=lambda local_prediction, remote_prediction: torch.testing.assert_close(
#                 pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(remote_prediction)[0],
#                 local_prediction[0],
#                 check_dtype=False,
#             ),
#             deployment_options={
#                 "platform": deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
#                 "target_method": "forward",
#                 "options": {
#                     "compute_pool": self._TEST_GPU_COMPUTE_POOL,
#                     "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
#                     "num_workers": 1,
#                     "use_gpu": True,
#                 },
#             },
#         )
#
#     @pytest.mark.pip_incompatible
#     def test_keras_model_deployment(self) -> None:
#         self._test_snowservice_deployment(
#             model_name="keras_model",
#             model_version=uuid.uuid4().hex,
#             prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_keras_model,
#             prediction_assert_fn=lambda local_prediction, remote_prediction: np.testing.assert_allclose(
#                 tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(remote_prediction)[0].numpy(),
#                 local_prediction[0],
#                 atol=1e-6,
#             ),
#             deployment_options={
#                 "platform": deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
#                 "target_method": "predict",
#                 "options": {
#                     "compute_pool": self._TEST_CPU_COMPUTE_POOL,
#                     "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
#                 },
#             },
#         )
#
#
if __name__ == "__main__":
    absltest.main()
