# import os
# import tempfile
# import uuid

# import pandas as pd
# from absl.testing import absltest

# from snowflake.ml.model import (
#     _deployer,
#     _model as model_api,
#     deploy_platforms,
#     type_hints as model_types,
# )
# from snowflake.ml.model.models import llm
# from tests.integ.snowflake.ml.test_utils import (
#     db_manager,
#     spcs_integ_test_base,
#     test_env_utils,
# )


# class TestSPCSLLMModelInteg(spcs_integ_test_base.SpcsIntegTestBase):
#     @classmethod
#     def setUpClass(cls) -> None:
#         super().setUpClass()
#         cls.cache_dir = tempfile.TemporaryDirectory()
#         cls._original_hf_home = os.getenv("HF_HOME", None)
#         os.environ["HF_HOME"] = cls.cache_dir.name

#     @classmethod
#     def tearDownClass(cls) -> None:
#         super().tearDownClass()
#         if cls._original_hf_home:
#             os.environ["HF_HOME"] = cls._original_hf_home
#         else:
#             del os.environ["HF_HOME"]
#         cls.cache_dir.cleanup()

#     def setUp(self) -> None:
#         # Set up a unique id for each artifact, in addition to the class-level prefix. This is particularly useful
#         # when differentiating artifacts generated between different test cases, such as service function names.
#         self.uid = uuid.uuid4().hex[:4]

#     def test_text_generation_pipeline(
#         self,
#     ) -> None:
#         import peft

#         ft_model = peft.AutoPeftModelForCausalLM.from_pretrained(
#             "peft-internal-testing/tiny-OPTForCausalLM-lora",
#             device_map="auto",
#         )
#         tmpdir = self.create_tempdir().full_path
#         ft_model.save_pretrained(tmpdir)
#         model = llm.LLM(
#             model_id_or_path=tmpdir,
#         )

#         x_df = pd.DataFrame(
#             [["Hello world"]],
#         )
#         cls = TestSPCSLLMModelInteg
#         stage_path = f"@{cls._TEST_STAGE}/{self.uid}/model.zip"
#         deployment_stage_path = f"@{cls._TEST_STAGE}/{self.uid}"
#         model_api.save_model(  # type: ignore[call-overload]
#             name="model",
#             session=self._session,
#             model_stage_file_path=stage_path,
#             model=model,
#             options={"embed_local_ml_library": True},
#             conda_dependencies=[
#                 test_env_utils.get_latest_package_version_spec_in_server(self._session, "snowflake-snowpark-python"),
#             ],
#         )
# svc_func_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
#     self._RUN_ID,
#     f"func_{self.uid}",
# )
#         deployment_options: model_types.SnowparkContainerServiceDeployOptions = {
#             "compute_pool": cls._TEST_GPU_COMPUTE_POOL,
#             "num_gpus": 1,
#             # TODO(halu): Create an separate testing registry.
#             # Creating new registry for each single test is costly since no cache hit would ever occurs.
#             "image_repo": "sfengineering-mlplatformtest.registry.snowflakecomputing.com/"
#             "regtest_db/regtest_schema/halu_test",
#             "enable_remote_image_build": True,
#             "model_in_image": True,
#         }

#         deploy_info = _deployer.deploy(
#             name=svc_func_name,
#             session=cls._session,
#             model_stage_file_path=stage_path,
#             deployment_stage_path=deployment_stage_path,
#             model_id=svc_func_name,
#             platform=deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
#             options={
#                 **deployment_options,  # type: ignore[arg-type]
#             },  # type: ignore[call-overload]
#         )
#         assert deploy_info is not None
#         res = _deployer.predict(session=cls._session, deployment=deploy_info, X=x_df)
#         self.assertIn("generated_text", res)
#         self.assertEqual(len(res["generated_text"]), 1)
#         self.assertNotEmpty(res["generated_text"][0])


# if __name__ == "__main__":
#     absltest.main()
