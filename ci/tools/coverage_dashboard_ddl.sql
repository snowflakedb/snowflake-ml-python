CREATE USER snowml_jenkins_coverage PASSWORD='snowml' MUST_CHANGE_PASSWORD=TRUE;
CREATE DATABASE snowml_coverage_db;
CREATE ROLE snowml_coverage_writer_rl;
CREATE ROLE snowml_coverage_reader_rl;
CREATE WAREHOUSE snowml_coverage_wh WITH warehouse_size = xsmall;

GRANT USAGE ON DATABASE snowml_coverage_db TO ROLE snowml_coverage_reader_rl;
GRANT USAGE ON DATABASE snowml_coverage_db TO ROLE snowml_coverage_writer_rl;
GRANT USAGE ON SCHEMA snowml_coverage_db.public TO ROLE snowml_coverage_reader_rl;
GRANT USAGE ON SCHEMA snowml_coverage_db.public TO ROLE snowml_coverage_writer_rl;
GRANT SELECT ON FUTURE TABLES IN DATABASE snowml_coverage_db TO ROLE snowml_coverage_reader_rl; -- Others can only view the tables
GRANT SELECT, INSERT ON FUTURE TABLES IN DATABASE snowml_coverage_db TO ROLE snowml_coverage_writer_rl; -- Only jenkins can insert the tables

CREATE TABLE IF NOT EXISTS snowml_coverage_db.public.breakdown_coverage
(
    time TIMESTAMP_LTZ, -- jenkins uses UTC time automatically. TIMESTAMP_LTZ internally stores UTC time with a specified precision.
    git_revision VARCHAR(40), -- git revision is 40 character long. record the main branch's git revision to insert
    package VARCHAR, -- each file's package (directory information)
    filename VARCHAR, -- file's name
    covered_lines INT, -- number of lines that is hit (covered in test) in one file
    total_lines INT, -- number of lines in one file
    duration_sec INT -- duration of the overall bazel coverage runtime in seconds
);
GRANT ROLE snowml_coverage_writer_rl TO USER snowml_jenkins_coverage;
GRANT OPERATE, USAGE ON WAREHOUSE snowml_coverage_wh TO ROLE snowml_coverage_writer_rl;
GRANT OPERATE, USAGE ON WAREHOUSE snowml_coverage_wh TO ROLE snowml_coverage_reader_rl;
