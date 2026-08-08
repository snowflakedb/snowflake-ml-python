"""Golden DESCRIBE-output fixtures captured from the live JKEW_DB.JKEW_SCHEMA env.

Each ``*.json`` file is the raw ``DESCRIBE ONLINE FEATURE TABLE <name>
TYPE = SPECIFICATION`` payload (decoded from the ``specification`` column).
``test_golden_spec_round_trip.py`` parametrises over them and asserts
byte-for-byte hash equality after export -> reload -> compile.
"""
