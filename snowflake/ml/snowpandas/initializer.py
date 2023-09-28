import importlib
import sys
from typing import Any, Callable, Iterable, List, Optional, Tuple

import cloudpickle
import pandas as pd

from snowflake import snowpark
from snowflake.ml.snowpandas import imports, patch
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark.modin.pandas.translation.default2pandas import (
    stored_procedure_utils,
)

modules = imports.import_modules()

SNOW_PICKLED_OBJ_KEY = "_SNOW_PICKLED_OBJ"
SNOW_TYPE_KEY = "_SNOW_TYPE"
TYPE_SNOW_DATAFRAME = "SNOW_DATAFRAME"
TYPE_SNOW_SERIES = "SNOW_SERIES"
SNOW_SERIES_NAME = "SNOW_SERIES_NAME"

INDEX = "_INDEX"


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
def init(session: Optional[snowpark.Session] = None) -> None:
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

    @F.sproc(  # type: ignore[misc]
        name=patch.PATCH_SPROC_NAME,
        replace=True,
        packages=deps,  # type: ignore[arg-type]
        is_permanent=False,
        # TODO(hayu): Remove the snowpandas import after it's available in snowflake channel.
        imports=[stored_procedure_utils.SNOWPARK_PANDAS_IMPORT],
    )
    def patch_sproc(
        session: snowpark.Session,
        model_file_location: str,
        method_data: bytes,
        args_data: bytes,
        kwargs_data: bytes,
    ) -> bytes:
        from snowflake.snowpark import files

        def replace_queries_with_pandas(session: snowpark.Session, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
            def replace(val: Any) -> Any:
                if isinstance(val, dict):
                    if SNOW_PICKLED_OBJ_KEY in val:
                        res = stored_procedure_utils.StoredProcedureDefault._try_recover_snowpark_pandas_objects(
                            session, val[SNOW_PICKLED_OBJ_KEY][0], val[SNOW_PICKLED_OBJ_KEY][1]
                        ).to_pandas()
                        if val[SNOW_TYPE_KEY] == TYPE_SNOW_SERIES:
                            res = res[val[SNOW_SERIES_NAME]].squeeze()
                        return res
                    return {key: replace(value) for key, value in val.items()}
                elif isinstance(val, Iterable):
                    if isinstance(val, list):
                        return [replace(item) for item in val]
                    elif isinstance(val, tuple):
                        return tuple(replace(item) for item in val)
                    elif isinstance(val, set):
                        return {replace(item) for item in val}
                return val

            args = tuple(replace(arg) for arg in args)
            kwargs = {key: replace(val) for key, val in kwargs.items()}
            return args, kwargs

        with files.SnowflakeFile.open(model_file_location, mode="rb", require_scoped_url=False) as model_file:
            model = cloudpickle.load(model_file)
        method = cloudpickle.loads(method_data)
        args = cloudpickle.loads(args_data)
        kwargs = cloudpickle.loads(kwargs_data)
        args, kwargs = replace_queries_with_pandas(session, *args, **kwargs)

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

        dataset = pd.json_normalize(ds)[cols[0]]
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
        # TODO(hayu): Use local snowpark version.
        # f"snowflake-snowpark-python=={snowpark.__version__}",
        "snowflake-snowpark-python",
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
