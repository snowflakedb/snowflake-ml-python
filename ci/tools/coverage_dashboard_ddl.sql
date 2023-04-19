CREATE TABLE IF NOT EXISTS breakdown_coverage
(
    time TIMESTAMP_LTZ, -- jenkins uses UTC time automatically. TIMESTAMP_LTZ internally stores UTC time with a specified precision.
    git_revision VARCHAR(40), -- git revision is 40 character long. record the main branch's git revision to insert
    package VARCHAR, -- each file's package (directory information)
    filename VARCHAR, -- file's name
    covered_lines INT, -- number of lines that is hit (covered in test) in one file
    total_lines INT, -- number of lines in one file
    duration_sec INT -- duration of the overall bazel coverage runtime in seconds
);
