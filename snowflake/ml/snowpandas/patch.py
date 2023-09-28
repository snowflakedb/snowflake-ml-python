import enum
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import cloudpickle
import pandas as pd

from snowflake import snowpark
from snowflake.ml._internal import file_utils
from snowflake.ml._internal.utils import identifier
from snowflake.snowpark import context, functions as F
from snowflake.snowpark._internal import utils
from snowflake.snowpark.modin.pandas.frontend import snow_dataframe, snow_series
from snowflake.snowpark.modin.pandas.translation.default2pandas import (
    stored_procedure_utils,
)

PATCH_SPROC_NAME = f"{utils.random_name_for_temp_object(utils.TempObjectType.PROCEDURE)}_patch_sproc"
PATCH_UDF_NAME = f"{utils.random_name_for_temp_object(utils.TempObjectType.FUNCTION)}_patch_udf"

SNOW_PICKLED_OBJ_KEY = "_SNOW_PICKLED_OBJ"
SNOW_TYPE_KEY = "_SNOW_TYPE"
TYPE_SNOW_DATAFRAME = "SNOW_DATAFRAME"
TYPE_SNOW_SERIES = "SNOW_SERIES"
SNOW_SERIES_NAME = "SNOW_SERIES_NAME"

INDEX = "_INDEX"


class PatchMode(enum.Enum):
    UPDATE = "UPDATE"
    RETURN = "RETURN"


class Patch:
    @staticmethod
    def create_sproc_patch(
        session: Optional[snowpark.Session], mode: PatchMode, method: Callable[..., Any], static: bool = False
    ) -> Callable[..., Any]:
        session = session or context.get_active_session()

        if static:

            def patch(*args: Any, **kwargs: Any) -> Any:
                has_snowpandas = _has_snowpandas(*args, **kwargs)
                if has_snowpandas:
                    args, kwargs = _replace_snowpandas_with_queries(*args, **kwargs)
                    res = cloudpickle.loads(
                        session.call(
                            PATCH_SPROC_NAME,
                            None,
                            cloudpickle.dumps(method),
                            cloudpickle.dumps(args),
                            cloudpickle.dumps(kwargs),
                        )
                    )
                else:
                    res = method(*args, **kwargs)

                # TODO: convert numpy to snowpandas
                if has_snowpandas and isinstance(res, pd.DataFrame):
                    res = snow_dataframe.SnowparkPandasDataFrame(res)
                return res

        else:

            def patch(self: Any, *args: Any, **kwargs: Any) -> Any:
                stage = session.get_session_stage()
                has_snowpandas = _has_snowpandas(*args, **kwargs)
                if has_snowpandas:
                    args, kwargs = _replace_snowpandas_with_queries(*args, **kwargs)
                    # staged model not exists
                    statement_params: Dict[str, Any] = {}
                    if (not hasattr(self, "_snowflake_model_file")) or (
                        not file_utils.stage_file_exists(session, stage, self._snowflake_model_file, statement_params)
                    ):
                        put_res = file_utils.stage_object(session, self, stage)
                        self._snowflake_model_file = put_res[0].target

                    model_file_location = f"{stage}/{self._snowflake_model_file}"
                    res = cloudpickle.loads(
                        session.call(
                            PATCH_SPROC_NAME,
                            model_file_location,
                            cloudpickle.dumps(method),
                            cloudpickle.dumps(args),
                            cloudpickle.dumps(kwargs),
                        )
                    )
                else:
                    res = method(self, *args, **kwargs)

                if mode == PatchMode.UPDATE:
                    self.__dict__.update(res.__dict__)
                    put_res = file_utils.stage_object(session, self, stage)
                    self._snowflake_model_file = put_res[0].target
                    return self
                elif mode == PatchMode.RETURN:
                    # TODO: convert numpy to snowpandas
                    if has_snowpandas and isinstance(res, pd.DataFrame):
                        res = snow_dataframe.SnowparkPandasDataFrame(res)
                    return res
                else:
                    raise ValueError(f"Invalid mode: {mode}")

        patch._patched = True  # type: ignore[attr-defined]
        return patch

    @staticmethod
    def create_udf_patch(session: Optional[snowpark.Session], method: Callable[..., Any]) -> Callable[..., Any]:
        session = session or context.get_active_session()

        def patch(self: Any, dataset: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(dataset, snow_dataframe.SnowparkPandasDataFrame):
                stage = session.get_session_stage()
                snowpark_df = dataset.to_snowpark(index_label=INDEX)

                # staged model not exists
                statement_params: Dict[str, Any] = {}
                if (not hasattr(self, "_snowflake_model_file")) or (
                    not file_utils.stage_file_exists(session, stage, self._snowflake_model_file, statement_params)
                ):
                    put_res = file_utils.stage_object(session, self, stage)
                    self._snowflake_model_file = put_res[0].target
                snowpark_res = snowpark_df.select(
                    F.call_udf(
                        PATCH_UDF_NAME,
                        F.object_construct_keep_null("*"),  # column order not preserved
                        F.lit(F.call_function("build_scoped_file_url", stage, self._snowflake_model_file)),
                        F.lit(method.__name__),
                        F.lit(identifier.get_unescaped_names(snowpark_df.columns)),
                        F.lit(cloudpickle.dumps(args)),
                        F.lit(cloudpickle.dumps(kwargs)),
                    ).as_("RECORD")
                )

                snowpandas_res = snowpark_res.to_snowpark_pandas()
                # res = snowpandas_res["RECORD"].apply(lambda x: snow_series.SnowPandasSeries(x['RES'])).to_frame()
                res = snow_dataframe.SnowparkPandasDataFrame(  # type: ignore[no-untyped-call]
                    snowpandas_res["RECORD"].apply(lambda x: x["RES"]).tolist(),
                    index=snowpandas_res["RECORD"].apply(lambda x: x[INDEX]),
                ).reindex(dataset.index)
                res.columns = [f"x{i}" for i in range(len(res.columns))]
                return res
            else:
                return method(self, dataset)

        patch._patched = True  # type: ignore[attr-defined]
        return patch

    @staticmethod
    def is_patchable(*, module: Any = None, klass: Any = None, method_name: str) -> bool:
        if module is None and klass is None:
            raise ValueError("Both module and klass are None.")

        # Class patch: Whether the class has the method and is not patched.
        if klass is not None and hasattr(klass, method_name) and callable(getattr(klass, method_name)):
            return not hasattr(getattr(klass, method_name), "_patched") or not getattr(klass, method_name)._patched

        # Function patch: Whether the function is not patched.
        if klass is None:
            return not hasattr(getattr(module, method_name), "_patched") or not getattr(module, method_name)._patched
        return False


def _has_snowpandas(*args: Any, **kwargs: Any) -> bool:
    def check(val: Any) -> Any:
        if isinstance(val, snow_dataframe.SnowparkPandasDataFrame) or isinstance(val, snow_series.SnowparkPandasSeries):
            return True
        elif isinstance(val, Iterable):
            if isinstance(val, list) or isinstance(val, tuple) or isinstance(val, set):
                return any(check(item) for item in val)
            elif isinstance(val, dict):
                return any(check(value) for key, value in val.items())
        return False

    if any(check(arg) for arg in args + tuple(kwargs.values())):
        return True
    return False


def _replace_snowpandas_with_queries(*args: Any, **kwargs: Any) -> Tuple[Any, Any]:
    def replace(val: Any) -> Any:
        if isinstance(val, snow_dataframe.SnowparkPandasDataFrame) or isinstance(val, snow_series.SnowparkPandasSeries):
            res = {
                SNOW_PICKLED_OBJ_KEY: stored_procedure_utils.StoredProcedureDefault._try_pickle_snowpark_pandas_objects(
                    val
                ),
                SNOW_TYPE_KEY: TYPE_SNOW_DATAFRAME
                if isinstance(val, snow_dataframe.SnowparkPandasDataFrame)
                else TYPE_SNOW_SERIES,
            }
            if res[SNOW_TYPE_KEY] == TYPE_SNOW_SERIES:
                res[SNOW_SERIES_NAME] = val.name
            return res
        elif isinstance(val, Iterable):
            if isinstance(val, list):
                return [replace(item) for item in val]
            elif isinstance(val, tuple):
                return tuple(replace(item) for item in val)
            elif isinstance(val, set):
                return {replace(item) for item in val}
            elif isinstance(val, dict):
                return {key: replace(value) for key, value in val.items()}
        return val

    args = tuple(replace(arg) for arg in args)
    kwargs = {key: replace(val) for key, val in kwargs.items()}
    return args, kwargs
