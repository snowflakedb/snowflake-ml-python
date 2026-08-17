"""
Integration tests for the AI_EMBED SQL function against a user-deployed SPCS service.

Covers:
  - Permission enforcement: roles without USAGE on the service are denied.
  - Table column operations: embedding columns, cosine similarity, nearest-neighbor search.
  - NULL handling: NULL inputs always return NULL and never raise.
"""

import logging
import os
import time

from absl.testing import absltest

import snowflake.snowpark.exceptions
from tests.integ.snowflake.ml.registry.services import registry_aisql_byom_test_base

logger = logging.getLogger(__name__)

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TestAIEmbedEndpointInteg(registry_aisql_byom_test_base.AISQLBYOMTestBase):
    """Integration tests for AI_EMBED against a user-deployed SPCS SentenceTransformer service."""

    _SESSION_PARAMS = {
        "ENABLE_SPCS_SERVICE_FUNCTIONS_IN_AIEMBED": "true",
        "SPCS_SERVICE_FUNCTION_IN_AIEMBED_NAME_KEYWORD": "'encode'",
    }

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._original_cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_cache_dir:
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = cls._original_cache_dir
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        super().tearDownClass()

    # ─── Deployment ───────────────────────────────────────────────────────────

    def _do_deploy(self) -> None:
        """Inner deployment — separated so the sentinel logic stays clean."""
        import sentence_transformers

        model_name = f"model_ai_embed_{self._run_id}"
        version_name = f"ver_{self._run_id}"
        service_name = f"service_ai_embed_{self._run_id}"
        pool_name = self._TEST_GPU_COMPUTE_POOL

        logger.info("Logging SentenceTransformer model %s ...", model_name)
        st_model = sentence_transformers.SentenceTransformer(_EMBED_MODEL)
        mv = self.registry.log_model(
            st_model,
            model_name=model_name,
            version_name=version_name,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

        logger.info("Creating SPCS service %s on pool %s ...", service_name, pool_name)
        mv.create_service(
            service_name=service_name,
            service_compute_pool=pool_name,
            ingress_enabled=True,
            gpu_requests=1,
        )

        # Poll by name rather than via mv.list_services().loc[0] to avoid
        # picking up unrelated services that may also be associated with the model.
        # The service is created in the current session schema (the test DB).
        cur_db = self.session.get_current_database().strip('"')
        cur_schema = self.session.get_current_schema().strip('"')
        fq_service_name = f"{cur_db}.{cur_schema}.{service_name.upper()}"
        logger.info("Waiting for service %s to reach RUNNING ...", fq_service_name)
        deadline = time.time() + 1800
        while time.time() < deadline:
            rows = self.session.sql(f"SHOW SERVICES LIKE '{service_name.upper()}'").collect()
            svc_status = rows[0]["status"] if rows else "PENDING"
            logger.info("Service status: %s", svc_status)
            if svc_status != "PENDING":
                break
            time.sleep(10)

        if svc_status != "RUNNING":
            raise AssertionError(f"Service did not reach RUNNING: status={svc_status!r}")

        TestAIEmbedEndpointInteg._model_fq_name = f"{cur_db}.{cur_schema}.{model_name.upper()}"
        TestAIEmbedEndpointInteg._service_name = fq_service_name
        logger.info("AI_EMBED test service deployed: %s", self._service_name)

        deadline = time.monotonic() + 120
        while True:
            try:
                result = self.session.sql(f"SELECT {fq_service_name}!ENCODE('hello world') AS result").collect()
                value = result[0]["RESULT"]
                logger.info("Direct SQL ENCODE('hello world') result: %s", value)
                if value is not None:
                    break
            except snowflake.snowpark.exceptions.SnowparkSQLException as e:
                logger.warning("Direct SQL ENCODE call failed: %s", e)
            if time.monotonic() >= deadline:
                logger.warning("ENCODE still returning None after 120 s — proceeding anyway")
                try:
                    logs = self.session.sql(
                        f"SELECT timestamp, log FROM TABLE({fq_service_name}!SPCS_GET_LOGS())"
                        " ORDER BY timestamp LIMIT 100"
                    ).collect()
                    for row in logs:
                        logger.info("CONTAINER LOG [%s]: %s", row["TIMESTAMP"], row["LOG"])
                except snowflake.snowpark.exceptions.SnowparkSQLException as log_err:
                    logger.warning("Could not fetch container logs: %s", log_err)
                break
            logger.info("ENCODE returned None, retrying in 10 s ...")
            time.sleep(10)

        # AI_EMBED requires explicit USAGE on the service and on its inference function
        # even for the owning role — ownership alone is not sufficient for SQL-level routing.
        current_role = self.session.get_current_role().strip('"')
        self.session.sql(f"GRANT USAGE ON SERVICE {fq_service_name} TO ROLE {current_role}").collect()
        self.session.sql(
            f"GRANT SERVICE ROLE {fq_service_name}!INFERENCE_SERVICE_FUNCTION_USAGE TO ROLE {current_role}"
        ).collect()

    # ─── Permission tests ─────────────────────────────────────────────────────

    def test_no_service_usage_denies_ai_embed(self) -> None:
        """A role without USAGE on the service cannot call AI_EMBED."""
        service_name = self._service_name
        role = self._aisql_byom_make_limited_role("EMBED_NO_PRIV", service_fqn=self._service_name)
        try:
            with self.assertRaisesRegex(
                Exception, "Insufficient privileges|does not exist or not authorized|invalid argument"
            ):
                self._run_as_role(
                    role,
                    lambda: self.session.sql(f"SELECT AI_EMBED('{service_name}', 'hello')").collect(),
                )
        finally:
            self._db_manager.drop_role(role, if_exists=True)

    def test_service_usage_allows_ai_embed(self) -> None:
        """A role with USAGE on the service can call AI_EMBED and receives a vector."""
        service_name = self._service_name
        admin_role = self.session.get_current_role().strip('"')
        role = self._aisql_byom_make_limited_role("EMBED_USAGE", service_fqn=self._service_name)
        try:
            self.session.sql(f"GRANT USAGE ON SERVICE {service_name} TO ROLE {role}").collect()
            self.session.sql(
                f"GRANT SERVICE ROLE {service_name}!INFERENCE_SERVICE_FUNCTION_USAGE TO ROLE {role}"
            ).collect()
            if self._model_fq_name:
                self.session.sql(f"GRANT USAGE ON MODEL {self._model_fq_name} TO ROLE {role}").collect()

            def _call() -> None:
                # Re-apply session params — some Snowflake parameters are not
                # preserved across USE ROLE in the same session.
                self.session.sql("ALTER SESSION SET ENABLE_SPCS_SERVICE_FUNCTIONS_IN_AIEMBED=true").collect()
                self.session.sql("ALTER SESSION SET SPCS_SERVICE_FUNCTION_IN_AIEMBED_NAME_KEYWORD='encode'").collect()
                result = self.session.sql(f"SELECT AI_EMBED('{service_name}', 'hello world') AS embedding").collect()
                self.assertLen(result, 1)
                self.assertIsNotNone(result[0]["EMBEDDING"])

            self._run_as_role(role, _call)
        finally:
            self.session.use_role(admin_role)
            self._db_manager.drop_role(role, if_exists=True)

    # ─── Table column tests ───────────────────────────────────────────────────

    def test_ai_embed_on_table_column(self) -> None:
        """AI_EMBED produces a non-null vector for every row in a table column."""
        service_name = self._service_name
        self.session.sql("CREATE OR REPLACE TEMPORARY TABLE documents (id INT, content VARCHAR)").collect()
        self.session.sql(
            "INSERT INTO documents VALUES "
            "(1, 'Snowflake is a cloud data platform.'), "
            "(2, 'Embeddings represent text as vectors.'), "
            "(3, 'AI_EMBED generates vector embeddings.')"
        ).collect()

        result = self.session.sql(
            f"""
            SELECT id, AI_EMBED('{service_name}', content) AS embedding
            FROM documents
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 3)
        for row in result:
            self.assertIsNotNone(row["EMBEDDING"], f"Expected non-null embedding for row id={row['ID']}")

    def test_ai_embed_vector_cosine_similarity(self) -> None:
        """VECTOR_COSINE_SIMILARITY of a text with itself is approximately 1.0."""
        service_name = self._service_name
        result = self.session.sql(
            f"""
            SELECT VECTOR_COSINE_SIMILARITY(
                AI_EMBED('{service_name}', 'Snowflake is a data platform'),
                AI_EMBED('{service_name}', 'Snowflake is a data platform')
            ) AS similarity
            """
        ).collect()

        self.assertLen(result, 1)
        similarity = result[0]["SIMILARITY"]
        self.assertIsNotNone(similarity)
        self.assertAlmostEqual(float(similarity), 1.0, delta=0.01)

    def test_ai_embed_nearest_neighbor(self) -> None:
        """The semantically closest row to a query is found via cosine similarity ranking."""
        service_name = self._service_name
        self.session.sql("CREATE OR REPLACE TEMPORARY TABLE articles (id INT, text VARCHAR)").collect()
        self.session.sql(
            "INSERT INTO articles VALUES "
            "(1, 'The stock market fell sharply today due to inflation fears.'), "
            "(2, 'The football team won the championship with a dramatic last-minute goal.'), "
            "(3, 'Scientists discovered a new species of deep-sea fish near hydrothermal vents.')"
        ).collect()

        result = self.session.sql(
            f"""
            SELECT id,
                VECTOR_COSINE_SIMILARITY(
                    AI_EMBED('{service_name}', text),
                    AI_EMBED('{service_name}', 'sports and athletic competition')
                ) AS similarity
            FROM articles
            ORDER BY similarity DESC
            LIMIT 1
            """
        ).collect()

        self.assertLen(result, 1)
        self.assertEqual(result[0]["ID"], 2, "Expected the sports article to be the nearest neighbor")

    # ─── NULL handling ────────────────────────────────────────────────────────

    def test_null_input_returns_null(self) -> None:
        """AI_EMBED with a NULL input returns NULL without raising."""
        service_name = self._service_name
        result = self.session.sql(f"SELECT AI_EMBED('{service_name}', NULL) AS embedding").collect()

        self.assertLen(result, 1)
        self.assertIsNone(result[0]["EMBEDDING"])

    def test_null_rows_in_batch_do_not_throw(self) -> None:
        """NULL rows in a batch return NULL; non-null rows return vectors. No exception is raised."""
        service_name = self._service_name
        self.session.sql("CREATE OR REPLACE TEMPORARY TABLE mixed_nulls (id INT, content VARCHAR)").collect()
        self.session.sql(
            "INSERT INTO mixed_nulls VALUES "
            "(1, 'hello world'), "
            "(2, NULL), "
            "(3, 'embeddings are useful'), "
            "(4, NULL)"
        ).collect()

        result = self.session.sql(
            f"""
            SELECT id, AI_EMBED('{service_name}', content) AS embedding
            FROM mixed_nulls
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 4)
        self.assertIsNotNone(result[0]["EMBEDDING"])  # id=1: non-null
        self.assertIsNone(result[1]["EMBEDDING"])  # id=2: NULL input
        self.assertIsNotNone(result[2]["EMBEDDING"])  # id=3: non-null
        self.assertIsNone(result[3]["EMBEDDING"])  # id=4: NULL input

    # ─── Helper ───────────────────────────────────────────────────────────────

    def _setup_embed_test_table(self) -> None:
        """Create TEMPORARY TABLE embed_test(id INT, text_col VARCHAR, cat VARCHAR).

        Row 2 intentionally has NULL text_col to exercise NULL-propagation paths.
        """
        self.session.sql(
            "CREATE OR REPLACE TEMPORARY TABLE embed_test (id INT, text_col VARCHAR, cat VARCHAR)"
        ).collect()
        self.session.sql(
            "INSERT INTO embed_test VALUES "
            "(1, 'Snowflake is a cloud platform', 'tech'), "
            "(2, NULL, 'empty'), "
            "(3, 'AI and machine learning', 'tech'), "
            "(4, 'Sports championship game', 'sports')"
        ).collect()

    # ─── Group A: expression-type inputs ─────────────────────────────────────

    def test_ai_embed_function_over_column(self) -> None:
        """A2: AI_EMBED accepts a scalar function (LOWER) applied to a column ref."""
        service_name = self._service_name
        self._setup_embed_test_table()

        result = self.session.sql(
            f"""
            SELECT id, AI_EMBED('{service_name}', LOWER(text_col)) AS e
            FROM embed_test
            WHERE text_col IS NOT NULL
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 3)
        for row in result:
            self.assertIsNotNone(row["E"])

    def test_ai_embed_coalesce_fallback(self) -> None:
        """A3: COALESCE as input — NULL rows fall back to literal and produce a non-null vector."""
        service_name = self._service_name
        self._setup_embed_test_table()

        result = self.session.sql(
            f"""
            SELECT id, AI_EMBED('{service_name}', COALESCE(text_col, 'fallback')) AS e
            FROM embed_test
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 4)
        for row in result:
            self.assertIsNotNone(row["E"], f"Expected non-null embedding for id={row['ID']} after COALESCE")

    def test_ai_embed_string_concat(self) -> None:
        """A4: String concatenation (||) as AI_EMBED input."""
        service_name = self._service_name
        self._setup_embed_test_table()

        result = self.session.sql(
            f"""
            SELECT id,
                AI_EMBED('{service_name}', cat || ': ' || COALESCE(text_col, '')) AS e
            FROM embed_test
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 4)
        for row in result:
            self.assertIsNotNone(row["E"])

    def test_ai_embed_case_expression(self) -> None:
        """A5: CASE expression as AI_EMBED input (multi-branch type unification)."""
        service_name = self._service_name
        self._setup_embed_test_table()

        result = self.session.sql(
            f"""
            SELECT id,
                AI_EMBED('{service_name}',
                    CASE WHEN cat = 'tech' THEN text_col ELSE UPPER(COALESCE(text_col, cat)) END
                ) AS e
            FROM embed_test
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 4)
        for row in result:
            self.assertIsNotNone(row["E"])

    def test_ai_embed_cast_input(self) -> None:
        """A6: CAST(integer column AS VARCHAR) as AI_EMBED input."""
        service_name = self._service_name
        self._setup_embed_test_table()

        result = self.session.sql(
            f"""
            SELECT id, AI_EMBED('{service_name}', CAST(id AS VARCHAR)) AS e
            FROM embed_test
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 4)
        for row in result:
            self.assertIsNotNone(row["E"])

    # ─── Group B: query-shape variations ─────────────────────────────────────

    def test_ai_embed_cte_column(self) -> None:
        """B1: AI_EMBED on a column projected through a CTE."""
        service_name = self._service_name
        self._setup_embed_test_table()

        result = self.session.sql(
            f"""
            WITH t AS (SELECT id, text_col AS s FROM embed_test WHERE text_col IS NOT NULL)
            SELECT id, AI_EMBED('{service_name}', s) AS e
            FROM t
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 3)
        for row in result:
            self.assertIsNotNone(row["E"])

    def test_ai_embed_subquery_column(self) -> None:
        """B2: AI_EMBED on a column projected through a subquery."""
        service_name = self._service_name
        self._setup_embed_test_table()

        result = self.session.sql(
            f"""
            SELECT id, AI_EMBED('{service_name}', s) AS e
            FROM (SELECT id, text_col AS s FROM embed_test WHERE text_col IS NOT NULL) sub
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 3)
        for row in result:
            self.assertIsNotNone(row["E"])

    @absltest.skip("Invalid pip requirement : --extra-index-url https://download.pytorch.org/whl/cu124.")
    def test_ai_embed_any_value_aggregate(self) -> None:
        """B6: AI_EMBED inside ANY_VALUE with GROUP BY — one vector per category."""
        service_name = self._service_name
        self._setup_embed_test_table()

        result = self.session.sql(
            f"""
            SELECT cat,
                ANY_VALUE(AI_EMBED('{service_name}', COALESCE(text_col, cat))) AS e
            FROM embed_test
            GROUP BY cat
            ORDER BY cat
            """
        ).collect()

        self.assertLen(result, 3)  # tech, empty, sports
        for row in result:
            self.assertIsNotNone(row["E"])

    def test_ai_embed_window_function(self) -> None:
        """B7: AI_EMBED result used inside LAG window function."""
        service_name = self._service_name
        self._setup_embed_test_table()

        result = self.session.sql(
            f"""
            SELECT id,
                AI_EMBED('{service_name}', COALESCE(text_col, cat)) AS e,
                LAG(AI_EMBED('{service_name}', COALESCE(text_col, cat))) OVER (ORDER BY id) AS prev_e
            FROM embed_test
            ORDER BY id
            """
        ).collect()

        self.assertLen(result, 4)
        self.assertIsNotNone(result[0]["E"])
        self.assertIsNone(result[0]["PREV_E"])  # no preceding row
        self.assertIsNotNone(result[1]["PREV_E"])  # lags row 1

    # ─── Group C: uses of the vector result ──────────────────────────────────

    def test_ai_embed_vector_l2_distance(self) -> None:
        """C2: VECTOR_L2_DISTANCE of a text with itself is 0.0."""
        service_name = self._service_name
        result = self.session.sql(
            f"""
            SELECT VECTOR_L2_DISTANCE(
                AI_EMBED('{service_name}', 'Snowflake is a data platform'),
                AI_EMBED('{service_name}', 'Snowflake is a data platform')
            ) AS dist
            """
        ).collect()

        self.assertLen(result, 1)
        self.assertAlmostEqual(float(result[0]["DIST"]), 0.0, delta=1e-4)

    def test_ai_embed_vector_subscript(self) -> None:
        """C3: Subscripting the embedding vector returns a scalar float.

        VECTOR type does not support direct bracket indexing; cast to ARRAY first.
        """
        service_name = self._service_name
        result = self.session.sql(
            f"SELECT (AI_EMBED('{service_name}', 'hello world')::ARRAY)[0]::FLOAT AS first_dim"
        ).collect()

        self.assertLen(result, 1)
        self.assertIsNotNone(result[0]["FIRST_DIM"])
        float(result[0]["FIRST_DIM"])  # must be numeric

    # ─── Group F: rejection paths ─────────────────────────────────────────────

    def test_ai_embed_prompt_input_rejected(self) -> None:
        """F2: PROMPT() input is rejected when AI_EMBED targets an SPCS service."""
        service_name = self._service_name
        self._setup_embed_test_table()

        with self.assertRaisesRegex(Exception, "(?i)PROMPT|not supported|not yet implemented"):
            self.session.sql(
                f"""
                SELECT AI_EMBED('{service_name}', PROMPT('Embed: {{0}}', text_col))
                FROM embed_test
                WHERE text_col IS NOT NULL
                """
            ).collect()


if __name__ == "__main__":
    absltest.main()
