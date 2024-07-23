import enum
import inspect
from typing import Any, Callable, Dict, Iterable, Optional, TypedDict

import cloudpickle
import numpy as np
import pandas as pd

from snowflake import snowpark
from snowflake.ml._internal import file_utils
from snowflake.ml._internal.utils import identifier
from snowflake.snowpark import context, functions as F
from snowflake.snowpark._internal import utils
from snowflake.snowpark.modin import pandas as SnowparkPandas

PATCH_SPROC_NAME = f"{utils.random_name_for_temp_object(utils.TempObjectType.PROCEDURE)}_patch_sproc"
PATCH_UDF_NAME = f"{utils.random_name_for_temp_object(utils.TempObjectType.FUNCTION)}_patch_udf"
INDEX = "_INDEX"


class PatchMode(enum.Enum):
    UPDATE = "UPDATE"
    RETURN = "RETURN"


class SnowflakeTable(TypedDict):
    _TABLE_NAME: str


class Patch:
    @staticmethod
    def create_sproc_patch(
        session: Optional[snowpark.Session], mode: PatchMode, method: Callable[..., Any], static: bool = False
    ) -> Callable[..., Any]:
        session = session or context.get_active_session()

        def replace_snowpark_pandas_with_snowflake_table_name(obj: Any) -> Any:
            if isinstance(obj, SnowparkPandas.DataFrame) or isinstance(obj, SnowparkPandas.Series):
                table_name = utils.random_name_for_temp_object(utils.TempObjectType.TABLE)
                obj.to_snowflake(table_name, index=True, index_label=INDEX, table_type="temporary")
                snowflake_table_name_dict: SnowflakeTable = {"_TABLE_NAME": table_name}
                return snowflake_table_name_dict
            # Covering all the iterable objects, including Lists, tuples, dictionaries, and sets
            if isinstance(obj, set):
                return {replace_snowpark_pandas_with_snowflake_table_name(item) for item in obj}
            elif isinstance(obj, list):
                return [replace_snowpark_pandas_with_snowflake_table_name(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: replace_snowpark_pandas_with_snowflake_table_name(value) for key, value in obj.items()}
            elif isinstance(obj, tuple):
                return tuple(replace_snowpark_pandas_with_snowflake_table_name(item) for item in obj)
            else:
                return obj

        if static:

            def patch(*args: Any, **kwargs: Any) -> Any:
                has_snowpark_pandas = _has_snowpark_pandas(*args, **kwargs)
                if has_snowpark_pandas:
                    pickled_args = replace_snowpark_pandas_with_snowflake_table_name(args)
                    pickled_kwargs = replace_snowpark_pandas_with_snowflake_table_name(kwargs)
                    res = cloudpickle.loads(
                        session.call(
                            PATCH_SPROC_NAME,
                            None,
                            cloudpickle.dumps(method),
                            cloudpickle.dumps(pickled_args),
                            cloudpickle.dumps(pickled_kwargs),
                        )
                    )
                else:
                    res = method(*args, **kwargs)

                # TODO: convert numpy to snowpark_pandas
                if has_snowpark_pandas and isinstance(res, pd.DataFrame):
                    res = SnowparkPandas.DataFrame(res)
                return res

        else:

            def patch(self: Any, *args: Any, **kwargs: Any) -> Any:
                stage = session.get_session_stage()
                has_snowpark_pandas = _has_snowpark_pandas(*args, **kwargs)
                if has_snowpark_pandas:
                    pickled_args = replace_snowpark_pandas_with_snowflake_table_name(args)
                    pickled_kwargs = replace_snowpark_pandas_with_snowflake_table_name(kwargs)

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
                            cloudpickle.dumps(pickled_args),
                            cloudpickle.dumps(pickled_kwargs),
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
                    # TODO: convert numpy to snowpark_pandas
                    if has_snowpark_pandas and isinstance(res, pd.DataFrame):
                        res = SnowparkPandas.DataFrame(res)
                    return res
                else:
                    raise ValueError(f"Invalid mode: {mode}")

        patch._patched = True  # type: ignore[attr-defined]
        return patch

    @staticmethod
    def create_udf_patch(session: Optional[snowpark.Session], method: Callable[..., Any]) -> Callable[..., Any]:
        session = session or context.get_active_session()

        def patch(self: Any, dataset: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(dataset, SnowparkPandas.DataFrame):
                stage = session.get_session_stage()
                snowpark_df = dataset.to_snowpark(index_label=INDEX)

                # staged model not exists
                statement_params: Dict[str, Any] = {}
                if (not hasattr(self, "_snowflake_model_file")) or (
                    not file_utils.stage_file_exists(session, stage, self._snowflake_model_file, statement_params)
                ):
                    put_res = file_utils.stage_object(session, self, stage)
                    self._snowflake_model_file = put_res[0].target

                # Constructing the column name key-value pairs for object_construct_keep_null
                column_name_key_value_pairs = []
                for col_name in snowpark_df.columns:
                    column_name_key_value_pairs.append(F.lit(col_name))
                    column_name_key_value_pairs.append(F.col(col_name))

                snowpark_res = snowpark_df.select(
                    F.call_udf(
                        PATCH_UDF_NAME,
                        F.object_construct_keep_null(*column_name_key_value_pairs),
                        F.lit(F.call_function("build_scoped_file_url", stage, self._snowflake_model_file)),
                        F.lit(method.__name__),
                        F.lit(identifier.get_unescaped_names(snowpark_df.columns)),
                        F.lit(cloudpickle.dumps(args)),
                        F.lit(cloudpickle.dumps(kwargs)),
                    ).as_("RECORD")
                )

                snowpark_pandas_res = snowpark_res.to_snowpark_pandas()
                res = SnowparkPandas.DataFrame(
                    data=snowpark_pandas_res["RECORD"].apply(lambda x: x["RES"]).tolist(),
                    index=snowpark_pandas_res["RECORD"].apply(lambda x: x[INDEX]),
                ).sort_index()
                res.columns = [f"x{i}" for i in range(len(res.columns))]
                return res
            else:
                return method(self, dataset)

        patch._patched = True  # type: ignore[attr-defined]
        return patch

    @staticmethod
    def create_native_patch(
        session: Optional[snowpark.Session],
        mode: PatchMode,
        method: Callable[..., Any],
        native_class: Any,
        static: bool = False,
        is_inference: bool = False,
    ) -> Callable[..., Any]:
        session = session or context.get_active_session()

        if static:
            # TODO: implement support for static patch
            patch = Patch.create_sproc_patch(session, mode, method, static)

        else:

            def patch(self: Any, *args: Any, **kwargs: Any) -> Any:
                stage = session.get_session_stage()
                has_snowpandas = _has_snowpark_pandas(*args, **kwargs)
                if has_snowpandas:
                    if args and _is_dataset(args[0]):
                        X = args[0]
                    else:
                        X = kwargs.get("X", kwargs.get("X_test", None))

                    if len(args) > 1 and _is_dataset(args[1]):
                        y = args[1]
                    else:
                        y = kwargs.get("y", None)
                    snowpandas_df = SnowparkPandas.general.concat([X, y], axis=1) if y is not None else X
                    snowpark_df = snowpandas_df.to_snowpark(index_label=INDEX)
                    input_cols = [identifier.get_inferred_name(col) for col in X.columns]
                    output_cols = input_cols
                    label_cols = []
                    if y is not None:
                        if isinstance(y, SnowparkPandas.DataFrame):
                            label_cols = [identifier.get_inferred_name(col) for col in y.columns]
                        elif isinstance(y, SnowparkPandas.Series):
                            label_cols = [identifier.get_inferred_name(y.name)]

                    native_kwargs = {"input_cols": input_cols, "output_cols": output_cols, "drop_input_cols": True}
                    if "label_cols" in inspect.signature(native_class.__init__).parameters:
                        native_kwargs["label_cols"] = label_cols
                    native_kwargs.update(kwargs)
                    for key in ["X", "X_test", "y"]:
                        native_kwargs.pop(key, None)

                    if is_inference:
                        # Fitting: pandas; inference: snowpandas.
                        if (not hasattr(self, "_snowflake_model")) or (self._snowflake_model is None):
                            udf_patch = Patch.create_udf_patch(session, method)
                            return udf_patch(self, X)
                        model = self._snowflake_model
                    else:
                        model = native_class(**native_kwargs)
                        _copy_attributes(self, model)
                    res = getattr(model, method.__name__)(snowpark_df)
                else:
                    res = method(self, *args, **kwargs)

                if mode == PatchMode.UPDATE:
                    self._snowflake_model = None
                    for m in ["to_sklearn", "to_xgboost", "to_lightgbm"]:
                        if hasattr(res, m):
                            self.__dict__.update(getattr(res, m)().__dict__)
                            self._snowflake_model = res
                            break
                    else:
                        self.__dict__.update(res.__dict__)
                    put_res = file_utils.stage_object(session, self, stage)
                    self._snowflake_model_file = put_res[0].target
                    return self
                elif mode == PatchMode.RETURN:
                    # TODO: convert numpy to snowpandas
                    if has_snowpandas and isinstance(res, pd.DataFrame):
                        res = SnowparkPandas.DataFrame(res)
                    elif isinstance(res, snowpark.DataFrame):
                        res = res.sort(INDEX).drop(INDEX).to_snowpark_pandas()
                    return res
                else:
                    raise ValueError(f"Invalid mode: {mode}")

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


def _has_snowpark_pandas(*args: Any, **kwargs: Any) -> bool:
    def check(val: Any) -> Any:
        if isinstance(val, SnowparkPandas.DataFrame) or isinstance(val, SnowparkPandas.Series):
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


def _is_dataset(x: Any) -> bool:
    return isinstance(x, SnowparkPandas.DataFrame) or isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray)


def _copy_attributes(from_obj: object, to_obj: object, include_privates: bool = False) -> None:
    for attr_name in dir(from_obj):
        try:
            if (not include_privates) and attr_name.startswith("_"):
                continue
            if not attr_name.startswith("__") and not callable(getattr(from_obj, attr_name)):
                setattr(to_obj, attr_name, getattr(from_obj, attr_name))
        # undefined property (e.g., OrdinalEncoder.infrequent_categories_)
        except AttributeError:
            continue
