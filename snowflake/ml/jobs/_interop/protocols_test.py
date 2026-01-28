import tempfile
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyarrow as pa
from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml.jobs._interop import protocols as p
from snowflake.ml.jobs._interop.dto_schema import ProtocolInfo


class MyClass:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: Any) -> bool:
        return bool(self.x == other.x and self.y == other.y)


class DummyProtocol(p.SerializationProtocol):
    def __init__(
        self,
        name: str,
        version: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        supported_types: p.Condition = None,
    ) -> None:
        self._protocol_info = ProtocolInfo(
            name=name,
            version=version,
            metadata=metadata,
        )
        self._supported_types = supported_types

    @property
    def supported_types(self) -> p.Condition:
        return self._supported_types

    @property
    def protocol_info(self) -> ProtocolInfo:
        return self._protocol_info

    def save(self, obj: Any, dest_dir: str, session: Optional[Any] = None) -> ProtocolInfo:
        # Simple dummy implementation - just return protocol info with manifest
        return self.protocol_info.with_manifest({"dummy_path": f"{dest_dir}/dummy.pkl"})

    def load(
        self,
        payload_info: ProtocolInfo,
        session: Optional[Any] = None,
        path_transform: Optional[Callable[[str], str]] = None,
    ) -> Any:
        # Simple dummy implementation - return a dummy object
        manifest_value = payload_info.manifest.get("dummy_path") if payload_info.manifest else None
        return {"loaded": True, "from": manifest_value}


class TestProtocols(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        (p.CloudPickleProtocol(), None),  # All types supported
        (p.ArrowTableProtocol(), pa.Table),
        (p.PandasDataFrameProtocol(), pd.DataFrame),
        (p.NumpyArrayProtocol(), np.ndarray),
    )
    def test_supported_types(self, protocol: p.SerializationProtocol, expected_type: Any) -> None:
        """Test that protocols report their supported types correctly."""
        self.assertEqual(protocol.supported_types, expected_type)

    @parameterized.parameters(  # type: ignore[misc]
        # CloudPickle - supports arbitrary Python objects
        (
            p.CloudPickleProtocol(),
            MyClass(1, 2),
        ),
        # Arrow Table
        (
            p.ArrowTableProtocol(),
            pa.table({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            lambda test, expected, actual: (
                test.assertEqual(expected.schema, actual.schema),
                test.assertEqual(expected.to_pydict(), actual.to_pydict()),
            ),
        ),
        # Pandas DataFrame
        (
            p.PandasDataFrameProtocol(),
            pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            lambda test, expected, actual: pd.testing.assert_frame_equal(expected, actual),
        ),
        # Numpy Array
        (
            p.NumpyArrayProtocol(),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
            lambda test, expected, actual: np.testing.assert_array_equal(expected, actual),
        ),
    )
    def test_serialization(
        self,
        protocol: p.SerializationProtocol,
        obj: Any,
        assertion_func: Optional[Callable[[Any, Any, Any], None]] = None,
    ) -> None:
        """Test serialization and deserialization of supported types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test serialization
            protocol_info = protocol.save(obj, temp_dir)
            self.assertIsInstance(protocol_info, ProtocolInfo)
            self.assertIsNotNone(protocol_info.manifest)

            # Check protocol-specific manifest structure
            assert isinstance(protocol_info.manifest, dict)
            if isinstance(protocol, p.CloudPickleProtocol):
                self.assertIn("path", protocol_info.manifest)
            elif isinstance(protocol, (p.ArrowTableProtocol, p.PandasDataFrameProtocol)):
                self.assertIn("paths", protocol_info.manifest)
                self.assertIsInstance(protocol_info.manifest["paths"], list)  # type: ignore[typeddict-item]
                self.assertGreater(len(protocol_info.manifest["paths"]), 0)  # type: ignore[typeddict-item]
            elif isinstance(protocol, p.NumpyArrayProtocol):
                self.assertIn("path", protocol_info.manifest)

            # Test deserialization
            loaded_obj = protocol.load(protocol_info)

            # Use the custom assertion function
            if assertion_func is None:
                self.assertEqual(obj, loaded_obj)
            else:
                assertion_func(self, obj, loaded_obj)

    @parameterized.parameters(  # type: ignore[misc]
        # Arrow Table with wrong type
        (
            p.ArrowTableProtocol(),
            {"not": "a table"},  # dict instead of Arrow Table
        ),
        # Pandas DataFrame with wrong type
        (
            p.PandasDataFrameProtocol(),
            [1, 2, 3],  # list instead of DataFrame
        ),
        # Numpy Array with wrong type
        (
            p.NumpyArrayProtocol(),
            "not an array",  # string instead of ndarray
        ),
    )
    def test_serialization_unsupported_types(
        self,
        protocol: p.SerializationProtocol,
        obj: Any,
    ) -> None:
        """Test that protocols raise appropriate errors for unsupported types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(p.SerializationError):
                protocol.save(obj, temp_dir)


class TestAutoProtocol(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sut = p.AutoProtocol()

    def test_register_protocol(self) -> None:
        proto = p.CloudPickleProtocol()
        self.sut.register_protocol(proto)
        self.assertEqual(len(self.sut._protocols), 1)
        self.assertEqual(self.sut._protocols[0], proto)

    def test_register_protocol_index(self) -> None:
        proto1 = DummyProtocol("proto_1")
        proto2 = DummyProtocol("proto_2")
        proto3 = DummyProtocol("proto_3")
        proto4 = DummyProtocol("proto_4")

        self.sut.register_protocol(proto1, index=0)
        self.assertEqual(self.sut._protocols, [proto1])
        self.sut.register_protocol(proto2, index=1)
        self.assertEqual(self.sut._protocols, [proto1, proto2])
        self.sut.register_protocol(proto3, index=-1)
        self.assertEqual(self.sut._protocols, [proto1, proto2, proto3])
        self.sut.register_protocol(proto4, index=1)
        self.assertEqual(self.sut._protocols, [proto1, proto4, proto2, proto3])

    def test_register_protocol_negative(self) -> None:
        # Test invalid protocol type
        with self.assertRaises(ValueError):
            self.sut.register_protocol("not_a_protocol")  # type: ignore[arg-type]

        # Test invalid index
        proto = DummyProtocol("proto")
        with self.assertRaises(ValueError):
            self.sut.register_protocol(proto, index=-2)

    @parameterized.parameters(  # type: ignore[misc]
        ("use_proto_1", "proto_2"),
        ("some_str", "proto_1"),
        (1.0, "proto_3"),
        (1, "proto_3"),
        (MyClass(1, 2), "proto_0"),
    )
    def test_save_protocol(self, obj: Any, expected: str) -> None:
        self.sut.register_protocol(DummyProtocol("proto_0", supported_types=None))
        self.sut.register_protocol(DummyProtocol("proto_1", supported_types=str))
        self.sut.register_protocol(DummyProtocol("proto_2", supported_types=lambda x: x == "use_proto_1"))
        self.sut.register_protocol(DummyProtocol("proto_3", supported_types=(float, int)))

        with tempfile.TemporaryDirectory() as temp_dir:
            proto_info = self.sut.save(obj, temp_dir)
            self.assertEqual(proto_info.name, expected)

    def test_try_register_protocol_success(self) -> None:
        initial_count = len(self.sut._protocols)
        self.sut.try_register_protocol(DummyProtocol, "test_proto")
        self.assertEqual(len(self.sut._protocols), initial_count + 1)
        self.assertEqual(self.sut._protocols[-1].protocol_info.name, "test_proto")

    def test_try_register_protocol_failure(self) -> None:
        # Create a class that will fail during construction
        class FailingProtocol(p.SerializationProtocol):
            def __init__(self) -> None:
                raise RuntimeError("Construction failed")

            @property
            def supported_types(self) -> p.Condition:
                return None

            @property
            def protocol_info(self) -> ProtocolInfo:
                return ProtocolInfo(name="failing")

            def save(self, obj: Any, dest_dir: str, session: Optional[Any] = None) -> ProtocolInfo:
                return ProtocolInfo(name="failing")

            def load(
                self,
                payload_info: ProtocolInfo,
                session: Optional[Any] = None,
                path_transform: Optional[Callable[[str], str]] = None,
            ) -> Any:
                pass

        initial_count = len(self.sut._protocols)
        # Should not raise an exception, just log a warning
        with self.assertLogs("snowflake.ml.jobs._interop.protocols", level="WARNING"):
            self.sut.try_register_protocol(FailingProtocol)
        self.assertEqual(len(self.sut._protocols), initial_count)


class TestCloudPickleProtocol(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.protocol = p.CloudPickleProtocol()
        self.mock_session = MagicMock(spec=snowpark.Session)

    def test_pack_unpack_arguments_roundtrip(self) -> None:
        """Test roundtrip for _pack_arguments and _unpack_arguments with mixed sessions."""
        original_args = (self.mock_session, 42, {"nested": self.mock_session})
        original_kwargs = {"flag": True, "session_kw": self.mock_session, "deep": [self.mock_session]}

        packed = self.protocol._pack_obj((original_args, original_kwargs))
        args, kwargs = self.protocol._unpack_obj(packed, session=self.mock_session)

        self.assertEqual(args, original_args)
        self.assertEqual(kwargs, original_kwargs)

    def test_pack_unpack_obj_complex(self) -> None:
        """Test _pack_obj and _unpack_obj with complex nested structures and mixed types."""
        obj = {
            "a": [1, 2, {"b": (self.mock_session, 3)}],
            "c": {"d": [self.mock_session, None], "e": "f"},
            "empty": ([], {}, ()),
        }

        packed = self.protocol._pack_obj(obj)
        unpacked = self.protocol._unpack_obj(packed, session=self.mock_session)

        self.assertEqual(unpacked, obj)

    def test_pack_unpack_only_session(self) -> None:
        """Test packing and unpacking when the object is just a session."""
        packed = self.protocol._pack_obj(self.mock_session)
        self.assertEqual(packed, {p.SESSION_KEY_PREFIX: None})

        unpacked = self.protocol._unpack_obj(packed, session=self.mock_session)
        self.assertEqual(unpacked, self.mock_session)


if __name__ == "__main__":
    absltest.main()
