import collections
import os
import tempfile
from typing import DefaultDict, List

import yaml
from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env as snowml_env
from snowflake.ml.model import _env


class EnvTest(absltest.TestCase):
    def test_conda_env_file(self) -> None:
        cd: DefaultDict[str, List[requirements.Requirement]]
        with tempfile.TemporaryDirectory() as tmpdir:
            cd = collections.defaultdict(list)
            _env.save_conda_env_file(tmpdir, cd)
            loaded_cd, _ = _env.load_conda_env_file(tmpdir)
            self.assertEqual(cd, loaded_cd)

        with tempfile.TemporaryDirectory() as tmpdir:
            cd = collections.defaultdict(list)
            cd["defaults"] = [requirements.Requirement("numpy")]
            _env.save_conda_env_file(tmpdir, cd)
            loaded_cd, _ = _env.load_conda_env_file(tmpdir)
            self.assertEqual(cd, loaded_cd)

        with tempfile.TemporaryDirectory() as tmpdir:
            cd = collections.defaultdict(list)
            cd["defaults"] = [requirements.Requirement("numpy>=1.22.4")]
            _env.save_conda_env_file(tmpdir, cd)
            loaded_cd, _ = _env.load_conda_env_file(tmpdir)
            self.assertEqual(cd, loaded_cd)

        with tempfile.TemporaryDirectory() as tmpdir:
            cd = collections.defaultdict(list)
            cd.update(
                {
                    "defaults": [requirements.Requirement("numpy>=1.22.4")],
                    "conda-forge": [requirements.Requirement("pytorch~=2.0")],
                }
            )
            _env.save_conda_env_file(tmpdir, cd)
            loaded_cd, _ = _env.load_conda_env_file(tmpdir)
            self.assertEqual(cd, loaded_cd)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, _env._CONDA_ENV_FILE_NAME), "w") as f:
                yaml.safe_dump(
                    stream=f,
                    data={
                        "name": "snow-env",
                        "dependencies": [
                            f"python={snowml_env.PYTHON_VERSION}",
                            "defaults::numpy>=1.22.4",
                            "conda-forge::pytorch~=2.0",
                            {"pip": "python-package"},
                        ],
                    },
                )
            loaded_cd, _ = _env.load_conda_env_file(tmpdir)
            self.assertEqual(
                {
                    "defaults": [requirements.Requirement("numpy>=1.22.4")],
                    "conda-forge": [requirements.Requirement("pytorch~=2.0")],
                },
                loaded_cd,
            )

    def test_generate_requirements_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rl: List[requirements.Requirement] = []
            _env.save_requirements_file(tmpdir, rl)
            loaded_rl = _env.load_requirements_file(tmpdir)
            self.assertEqual(rl, loaded_rl)

        with tempfile.TemporaryDirectory() as tmpdir:
            rl = [requirements.Requirement("python-package==1.0.1")]
            _env.save_requirements_file(tmpdir, rl)
            loaded_rl = _env.load_requirements_file(tmpdir)
            self.assertEqual(rl, loaded_rl)


if __name__ == "__main__":
    absltest.main()
