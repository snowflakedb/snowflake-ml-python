from absl.testing.absltest import TestCase, main

import snowflake.ml._internal.utils.uri as uri


class UriTest(TestCase):
    """Testing URI utility functions."""

    def test_snowflake_stage_uris(self) -> None:
        """Tests the handling of Snowflake stage URIs."""

        # Normal operation with valid inputs.
        self.assertTrue(uri.is_snowflake_stage_uri("sfc://SNOWFLAKE_STAGE/content"))
        self.assertTrue(uri.is_snowflake_stage_uri("sfstage://SNOWFLAKE_STAGE/content"))
        self.assertTrue(uri.is_snowflake_stage_uri("sfstage:@SNOWFLAKE_STAGE/content"))

        # Uppercase scheme is still recognized as valid
        self.assertTrue(uri.is_snowflake_stage_uri("SFC://SNOWFLAKE_STAGE/content"))

        # Invalid or non-snowflake schemes.
        self.assertFalse(uri.is_snowflake_stage_uri("sfc_stage://SNOWFLAKE_STAGE/content"))
        self.assertFalse(uri.is_snowflake_stage_uri("http://SNOWFLAKE_STAGE/content"))

        # Extracting the stage name.
        self.assertEqual(
            uri.get_snowflake_stage_path_from_uri("sfc://SNOWFLAKE_STAGE/content"), "SNOWFLAKE_STAGE/content"
        )

        self.assertEqual(
            uri.get_uri_from_snowflake_stage_path("@SNOWFLAKE_STAGE/content"), "sfc://SNOWFLAKE_STAGE/content"
        )

        self.assertEqual(
            uri.get_snowflake_stage_path_from_uri("sfc://SNOWFLAKE_STAGE/content/"), "SNOWFLAKE_STAGE/content"
        )

        self.assertEqual(
            uri.get_uri_from_snowflake_stage_path("@SNOWFLAKE_STAGE/content/"), "sfc://SNOWFLAKE_STAGE/content"
        )

        self.assertEqual(uri.get_snowflake_stage_path_from_uri("sfc://SNOWFLAKE_STAGE"), "SNOWFLAKE_STAGE")

        self.assertEqual(uri.get_uri_from_snowflake_stage_path("@SNOWFLAKE_STAGE"), "sfc://SNOWFLAKE_STAGE")

        self.assertEqual(uri.get_snowflake_stage_path_from_uri("sfc://SNOWFLAKE_STAGE/"), "SNOWFLAKE_STAGE")

        self.assertEqual(uri.get_uri_from_snowflake_stage_path("@SNOWFLAKE_STAGE/"), "sfc://SNOWFLAKE_STAGE")

        self.assertEqual(
            uri.get_uri_from_snowflake_stage_path("@SNOWFLAKE_DB.SNOWFLAKE_SCHEMA.SNOWFLAKE_STAGE/content"),
            "sfc://SNOWFLAKE_DB.SNOWFLAKE_SCHEMA.SNOWFLAKE_STAGE/content",
        )

        self.assertEqual(
            uri.get_uri_from_snowflake_stage_path(
                stage_path='@"SNOWFLAKE_DB"."SNOWFLAKE_SCHEMA".SNOWFLAKE_STAGE/content'
            ),
            'sfc://"SNOWFLAKE_DB"."SNOWFLAKE_SCHEMA".SNOWFLAKE_STAGE/content',
        )

        self.assertEqual(uri.get_snowflake_stage_path_from_uri("sfc://SNOWFLAKE_STAGE"), "SNOWFLAKE_STAGE")

        # No stage path from invalid scheme.
        self.assertEqual(uri.get_snowflake_stage_path_from_uri("unsupported://SNOWFLAKE_STAGE"), None)

        # Assembling URIs
        self.assertEqual(
            uri.get_uri_from_snowflake_stage_path("@SNOWFLAKE_STAGE/content"), "sfc://SNOWFLAKE_STAGE/content"
        )

    def test_non_snowflake_uris(self) -> None:
        """Tests various other non-Snowflake-specific URIs"""

        # Http.
        self.assertTrue(uri.is_http_uri("https://www.snowflake.com/en/"))
        self.assertTrue(uri.is_http_uri("http://www.snowflake.com/en/"))
        self.assertFalse(uri.is_http_uri("udp://www.snowflake.com/en/"))
        self.assertFalse(uri.is_http_uri("/www.snowflake.com/en/"))

        # Local files.
        self.assertTrue(uri.is_local_uri("/dev/null"))
        self.assertTrue(uri.is_local_uri("file://etc/passwd"))
        self.assertFalse(uri.is_local_uri("https://www.snowflake.com/en/"))

    def test_get_uri_scheme(self) -> None:
        """Return scheme of URI."""
        self.assertEqual(uri.get_uri_scheme("s3://my_bucket/"), "s3")

    def test_get_stage_and_path(self) -> None:
        self.assertEqual(uri.get_stage_and_path("@db.schema.stage/a/spec.yaml"), ("@db.schema.stage", "a/spec.yaml"))


if __name__ == "__main__":
    main()
