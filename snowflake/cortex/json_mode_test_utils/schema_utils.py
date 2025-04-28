"""
Aux file for test purposes, keeping test data for json-mode related tests in one place and organised.
"""

from dataclasses import dataclass
from typing import Any, Union

from pandas.core.interchange.dataframe_protocol import Column

from snowflake.cortex._complete import ConversationMessage, ResponseFormat


@dataclass
class JsonModeTestUtils:
    """Aux object for test purposes to unify testing data"""

    prompt: Union[str, list[ConversationMessage], Column]
    response_format: ResponseFormat
    error_message_sql: str
    error_message_rest: dict[str, Any]
    expected_response: dict[str, Any]


response_format_with_bad_input = JsonModeTestUtils(
    prompt=[{"role": "user", "content": "Please prepare me a data set of 5 ppl and their age"}],
    response_format={
        "type": "xml",
        "schema": {
            "type": "object",
            "properties": {"people": {"type": "string"}},
            "required": ["people"],
        },
    },
    error_message_sql="json mode output validation error: An error occurred while validating the model output%!"
    "(EXTRA string=json mode output validation error: unsupported value for 'responseFormat.type' "
    "field: xml. Only 'json' is supported)",
    error_message_rest={"code": "390142", "message": "Incoming request does not contain a valid payload."},
    expected_response={},
)


# 422
response_format_failing_input_validation = JsonModeTestUtils(
    prompt=[{"role": "user", "content": "Please prepare me a data set of 5 ppl and their age"}],
    response_format={
        "type": "json",
        "schema": {"type": "object", "properties": {"people": {"type": "i_dont_exist"}}, "required": ["people"]},
    },
    error_message_sql=r"input schema validation error: [{\"evaluationPath\":\"/properties/properties\",\"errors\":"
    r"{\"additionalProperties\":\"Additional property 'dataset_name' does not match the schema\"}},"
    r"{\"evaluationPath\":\"/additionalProperties/dataset_name\",\"errors\":{\"$ref\":\"Value does "
    r"not match the reference schema\"}},{\"errors\":{\"properties\":\"Property 'type' does not match"
    r" the schema\"}},{\"evaluationPath\":\"/properties/type\",\"errors\":{\"enum\":\"Value should "
    r"match one of the values specified by the enum\"}}]",
    error_message_rest={
        "message": 'input schema validation error: [{"evaluationPath":"/properties/properties","errors":'
        '{"additionalProperties":"Additional property \'dataset_name\' does not match the schema"}},'
        '{"evaluationPath":"/additionalProperties/dataset_name","errors":{"$ref":"Value does not match the reference '
        'schema"}},{"errors":{"properties":"Property \'type\' does not match the schema"}},{"evaluationPath":"/'
        'properties/type","errors":{"enum":"Value should match one of the values specified by the enum"}}]',
        "request_id": "2a257ab1-1ac2-44d6-9020-5d183e72c9ef",
    },
    expected_response={},
)


response_format_positive = JsonModeTestUtils(
    prompt=[
        {"role": "system", "content": "Age should be between 20 and 30"},
        {"role": "user", "content": "Please prepare me a data set of 5 ppl and their age"},
    ],
    response_format={
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "number"},
                        },
                        "required": ["name", "age"],
                    },
                }
            },
            "required": ["people"],
        },
    },
    error_message_sql="",
    error_message_rest={},
    expected_response={
        "people": [
            {"age": 22, "name": "John Smith"},
            {"age": 27, "name": "Sarah Jones"},
            {"age": 25, "name": "Michael Chen"},
            {"age": 28, "name": "Emma Wilson"},
            {"age": 23, "name": "Lucas Brown"},
        ]
    },
)
