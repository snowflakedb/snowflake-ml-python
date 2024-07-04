import importlib
import sys
from typing import Any, Callable, List, Optional

import cloudpickle
import pandas as pd
from snowflake.ml.beta.snowpark_pandas import imports, patch

from snowflake import snowpark
from snowflake.ml._internal.utils import pkg_version_utils
from snowflake.snowpark import functions as F, types as T

modules = imports.import_modules()

INDEX = "_INDEX"
TABLE_NAME = "_TABLE_NAME"


def _invoke_once(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not hasattr(func, "invoked"):
            func.invoked = False  # type: ignore[attr-defined]

        if func.invoked:  # type: ignore[attr-defined]
            return
        else:
            func.invoked = True  # type: ignore[attr-defined]
            return func(*args, **kwargs)

    return wrapper


@_invoke_once
def init(session: Optional[snowpark.Session] = None, enable_relax_version: Optional[bool] = True) -> None:
    for module_name in modules:
        module = sys.modules[module_name]

        for class_name in modules[module_name][imports.TYPE_CLASS]:
            klass = getattr(module, class_name)

            sproc_update_functions = ["fit"]
            for func in sproc_update_functions:
                if patch.Patch.is_patchable(klass=klass, method_name=func):
                    setattr(
                        klass,
                        func,
                        patch.Patch.create_sproc_patch(session, patch.PatchMode.UPDATE, getattr(klass, func)),
                    )

            sproc_return_functions = ["score"]
            for func in sproc_return_functions:
                if patch.Patch.is_patchable(klass=klass, method_name=func):
                    setattr(
                        klass,
                        func,
                        patch.Patch.create_sproc_patch(session, patch.PatchMode.RETURN, getattr(klass, func)),
                    )

            udf_functions = ["predict", "transform", "predict_proba", "predict_log_proba", "decision_function"]
            for func in udf_functions:
                if patch.Patch.is_patchable(klass=klass, method_name=func):
                    setattr(klass, func, patch.Patch.create_udf_patch(session, getattr(klass, func)))

        for function_name in modules[module_name][imports.TYPE_FUNCTION]:
            if patch.Patch.is_patchable(module=module, method_name=function_name):
                setattr(
                    module,
                    function_name,
                    patch.Patch.create_sproc_patch(
                        session, patch.PatchMode.RETURN, getattr(module, function_name), static=True
                    ),
                )

    deps = _gather_deps()
    snowpark_pandas_dep = ["pandas==2.2.1", "modin==0.28.1"]

    if enable_relax_version and session is not None:
        assert session is not None
        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=deps + snowpark_pandas_dep, session=session
        )

    @F.sproc(  # type: ignore[misc]
        name=patch.PATCH_SPROC_NAME,
        replace=True,
        packages=relaxed_dependencies,  # type: ignore[arg-type]
        is_permanent=False,
    )
    def patch_sproc(
        session: snowpark.Session,
        model_file_location: str,
        method_data: bytes,
        pickled_args_data: bytes,
        pickled_kwargs_data: bytes,
    ) -> bytes:
        from snowflake.snowpark import files
        from snowflake.snowpark.modin.pandas import read_snowflake

        def replace_snowflake_table_name_with_pandas(obj: Any) -> Any:
            if isinstance(obj, dict) and obj.get(TABLE_NAME, None) is not None:
                snowflake_table_name = obj.get(TABLE_NAME)
                sp_df = read_snowflake(snowflake_table_name, index_col=INDEX)
                df = sp_df.to_pandas()
                df.columns = sp_df.columns
                return df
            # Covering all the iterable objects, including Lists, tuples, dictionaries, and sets
            if isinstance(obj, set):
                return {replace_snowflake_table_name_with_pandas(item) for item in obj}
            elif isinstance(obj, list):
                return [replace_snowflake_table_name_with_pandas(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: replace_snowflake_table_name_with_pandas(value) for key, value in obj.items()}
            elif isinstance(obj, tuple):
                return tuple(replace_snowflake_table_name_with_pandas(item) for item in obj)
            else:
                return obj

        def deserialize_snowpark_pandas_data_to_pandas_obj(data: bytes) -> Any:
            obj = cloudpickle.loads(data)
            return replace_snowflake_table_name_with_pandas(obj)

        with files.SnowflakeFile.open(model_file_location, mode="rb", require_scoped_url=False) as model_file:
            model = cloudpickle.load(model_file)
        method = cloudpickle.loads(method_data)
        args = deserialize_snowpark_pandas_data_to_pandas_obj(pickled_args_data)
        kwargs = deserialize_snowpark_pandas_data_to_pandas_obj(pickled_kwargs_data)
        return cloudpickle.dumps(getattr(model, method.__name__)(*args, **kwargs))  # type: ignore[no-any-return]

    @F.pandas_udf(  # type: ignore[arg-type, misc]
        name=patch.PATCH_UDF_NAME,
        replace=True,
        packages=deps,  # type: ignore[arg-type]
        is_permanent=False,
    )
    def patch_udf(
        ds: T.PandasSeries[dict],  # type: ignore[type-arg]
        model_file_location: T.PandasSeries[str],
        method_name: T.PandasSeries[str],
        cols: T.PandasSeries[List[str]],
        args_data: T.PandasSeries[bytes],
        kwargs_data: T.PandasSeries[bytes],
    ) -> T.PandasSeries[dict]:  # type: ignore[type-arg]
        from snowflake.snowpark import files

        dataset = pd.json_normalize(ds)
        dataset = dataset[cols[0]]
        with files.SnowflakeFile.open(model_file_location[0], mode="rb") as model_file:
            model = cloudpickle.load(model_file)
        args = cloudpickle.loads(args_data[0])
        kwargs = cloudpickle.loads(kwargs_data[0])
        res_arr = getattr(model, method_name[0])(dataset.drop(columns=[INDEX]), *args, **kwargs)

        # # reuse autogen
        # if isinstance(res_arr, list) and len(res_arr) > 0 and isinstance(res_arr[0], np.ndarray):
        #     res_arr = np.concatenate(res_arr, axis=1)
        #
        # if len(res_arr.shape) == 3:
        #     res_arr = np.hstack(res_arr)

        if len(res_arr.shape) > 1:
            series = pd.Series(res_arr.tolist())
            res_df = pd.DataFrame(series, columns=["RES"])
        else:
            res_df = pd.DataFrame(res_arr, columns=["RES"])
        res_df[INDEX] = dataset[INDEX]
        return res_df.to_dict("records")  # type: ignore[no-any-return]


def _gather_deps() -> List[str]:
    deps = [
        f"cloudpickle=={cloudpickle.__version__}",
        f"snowflake-snowpark-python=={snowpark.__version__}",
    ]

    # optional deps
    for module in ["xgboost", "lightgbm"]:
        if module in modules:
            deps.append(f"{module}=={sys.modules[module].__version__}")

    # scikit-learn is also optional, and is imported as 'sklearn'.
    if any("sklearn" in key for key in modules.keys()):
        sklearn = importlib.import_module("sklearn")
        deps.append(f"scikit-learn=={sklearn.__version__}")

    return deps
