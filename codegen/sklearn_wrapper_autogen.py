# TODO(snandamuri): Move the Autogen class to estimator_autogen_tool.py file and delete this file.
import importlib
import inspect
import os
import sys
import types
from collections import namedtuple
from enum import Enum
from typing import Iterable, List, Optional

from absl import logging

from codegen import sklearn_wrapper_generator as swg

InitRuleInfo = namedtuple("InitRuleInfo", ["init_import_statement", "init_export_statement"])


class GenMode(Enum):
    SRC = "SRC"
    TEST = "TEST"
    SNOWPARK_PANDAS_TEST = "SNOWPARK_PANDAS_TEST"


class AutogenTool:
    """Tool to auto-generate estimator wrappers and integration test for estimator wrappers.

    Args:
        gen_mode: Possible values {GenMode.SRC, GenMode.TEST, GenMode.SNOWPARK_PANDAS_TEST}. Tool generates source code
            for estimator wrappers or integration tests for generated estimator wrappers or snowpark_pandas based on the
            selected mode.
        template_path: Path to file containing estimator wrapper or test template code.
        output_path : Path to the root of the destination folder to write auto-generated code.
        class_list: Allow list of estimator classes. If specified, wrappers or tests will be generated for only
            the classes in the allow list.
    """

    def __init__(
        self,
        template_path: str,
        output_path: str,
        gen_mode: GenMode = GenMode.SRC,
        class_list: Optional[List[str]] = None,
    ) -> None:
        self.gen_mode = gen_mode
        self.template_path = template_path
        self.class_list = class_list
        self.output_path = output_path

        self._validate()

    def _validate(self) -> None:
        if self.template_path is None or self.output_path is None:
            raise AssertionError("template_path and output_path must be specified.")

    @staticmethod
    def get_estimator_class_names(module_name: str) -> List[str]:
        """Inspects the module and computes the list of estimator classes supported AutogenTool.

        Args:
            module_name: Name of the module to inspect.

        Returns:
            List of estimator class names in the given module for which AutogenTool can generate
            wrappers.
        """
        importlib.import_module(module_name)

        module = sys.modules[module_name]
        class_names = []
        for transformer in inspect.getmembers(module):
            if (
                inspect.isclass(transformer[1])
                and swg.WrapperGeneratorFactory.can_generate_wrapper(transformer)
                # Not an imported class
                and transformer[1].__module__.startswith(module_name)
            ):
                class_names.append(transformer[0])
        return class_names

    def generate(self, module_name: str, skip_code_gen: bool = False) -> List[str]:
        """Autogenerate snowflake estimator wrappers for the given list of SKLearn or XGBoost modules.

        Args:
            module_name: Module names to process.
            skip_code_gen: If set to true, this method skips the actual code generation and just returns the list of
                files that would have been generated.

        Returns:
            List of generated files.
        """
        importlib.import_module(module_name)

        module = sys.modules[module_name]
        generators = self._get_wrapper_generators(module=module)

        if self.gen_mode == GenMode.SRC:
            return self._generate_src_files(module_name=module_name, generators=generators, skip_code_gen=skip_code_gen)
        else:
            return self._generate_test_files(
                module_name=module_name, generators=generators, skip_code_gen=skip_code_gen
            )

    def _generate_src_files(
        self, module_name: str, generators: Iterable[swg.WrapperGeneratorBase], skip_code_gen: bool = False
    ) -> List[str]:
        """Autogenerate snowflake estimator wrappers for the given SKLearn or XGBoost module.

        Args:
            module_name: Module name to process.
            generators: Generator objects for estimator classes in the given module.
            skip_code_gen: If set to true, this method skips the actual code generation and just returns the list of
                files that would have been generated.

        Returns:
            List of generated files.
        """

        template = open(self.template_path, encoding="utf-8").read()

        generated_files_list = []
        for generator in generators:
            output_file_name = os.path.join(self.output_path, generator.estimator_file_name)
            generated_files_list.append(output_file_name)
            if skip_code_gen:
                continue

            # Apply generator to template and write.
            wrapped_transform_string = template.format(transform=generator)
            wrapped_transform_string = self._filter_out_todo_lines(wrapped_transform_string)
            wrapped_transform_string = self._add_do_not_modify_warning(wrapped_transform_string, True)

            # Create output src dir if it don't exist already.
            os.makedirs("/".join(output_file_name.split("/")[:-1]), exist_ok=True)

            open(output_file_name, "w", encoding="utf-8").write(wrapped_transform_string)
            logging.info("Wrote file %s", output_file_name)

        return generated_files_list

    def _generate_test_files(
        self, module_name: str, generators: Iterable[swg.WrapperGeneratorBase], skip_code_gen: bool = False
    ) -> List[str]:
        """Autogenerate integ tests for snowflake estimator wrappers or snowpark_pandas for the given SKLearn or XGBoost
        module.

        Args:
            module_name: Module name to process.
            generators: Generator objects for estimator classes in the given module.
            skip_code_gen : If set to true, this method skips the actual code generation and just returns the list of
                files that would have been generated.

        Returns:
            List of generated files.
        """
        test_template = open(self.template_path, encoding="utf-8").read()

        generated_files_list = []
        for generator in generators:
            if self.gen_mode == GenMode.TEST:
                test_output_file_name = os.path.join(self.output_path, generator.estimator_test_file_name)
            else:
                test_output_file_name = os.path.join(self.output_path, generator.snowpark_pandas_test_file_name)
            generated_files_list.append(test_output_file_name)
            if skip_code_gen:
                continue

            # Apply generator to template and write.
            wrapped_transform_string = test_template.format(transform=generator)
            wrapped_transform_string = self._filter_out_todo_lines(wrapped_transform_string)
            wrapped_transform_string = self._add_do_not_modify_warning(wrapped_transform_string, True)

            # Create output test dir if it don't exist already.
            os.makedirs("/".join(test_output_file_name.split("/")[:-1]), exist_ok=True)

            open(test_output_file_name, "w", encoding="utf-8").write(wrapped_transform_string)
            logging.info("Wrote file %s", test_output_file_name)

        return generated_files_list

    def _get_wrapper_generators(self, module: types.ModuleType) -> List[swg.WrapperGeneratorBase]:
        """
        Construct wrapper generators for all the supported estimator classes in the given module.

        Args:
            module: Python module object for the module to be processed.

        Returns:
            List of wrapper generator objects for estimator classes in the given module.
        """
        module_name = dict(inspect.getmembers(module))["__spec__"].name
        generators = []
        for transformer in inspect.getmembers(module):
            if inspect.isclass(transformer[1]):
                if (
                    self.class_list is None or len(self.class_list) == 0 or transformer[0] in self.class_list
                ) and swg.WrapperGeneratorFactory.can_generate_wrapper(transformer):
                    # Read information from input transformer.
                    generator = swg.WrapperGeneratorFactory.read(module_name=module_name, class_object=transformer)
                    generators.append(generator)
        return generators

    def _filter_out_todo_lines(self, generated_code: str) -> str:
        """Removes TODO lines from the generated code.

        Args:
            generated_code: Generated code.

        Returns:
            Code with TODO lines removed.
        """
        return "".join(
            [line for line in generated_code.splitlines(keepends=True) if not line.strip().startswith("# TODO")]
        )

    def _add_do_not_modify_warning(self, generated_code: str, is_src_file: bool) -> str:
        """Prepend a warning statement to dicourage modifying auto generated code.

        Args:
            generated_code: Generated code.
            is_src_file: True if source file, False is it is test file.

        Returns:
            Code with do not modify warning.
        """
        template_name = (
            "sklearn_wrapper_template.py_template" if is_src_file else "transformer_autogen_test_template.py_template"
        )
        comment = (
            "#\n"
            + f"# This code is auto-generated using the {template_name} template.\n"
            + "# Do not modify the auto-generated code(except automatic reformatting by precommit hooks).\n"
            + "#\n"
        )
        return comment + generated_code

    @staticmethod
    def module_root_dir(module_name: str) -> str:
        """Compute root directory for the given module.

        Args:
            module_name: module name.

        Returns:
            Root directory for the given module.
        """
        snowml_module_name = swg.WrapperGeneratorFactory.get_snow_ml_module_name(module_name)
        return os.path.join("/".join(snowml_module_name.split(".")))
