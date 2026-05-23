from snowflake.ml.model._signatures import core

# OpenAI/vLLM structured outputs response_format:
# https://github.com/openai/openai-python/blob/ef00216846515033e4cf73ab3227e91386d958ba/src/openai/types/shared/response_format_json_schema.py#L45
# {type: "json_schema", json_schema: {name, description?, schema: <JSON Schema dict>, strict?}}
# The inner `schema` is arbitrary JSON, modeled as DataType.OBJECT (Snowpark MapType<String, Variant>).
_RESPONSE_FORMAT_FEATURE_SPEC = core.FeatureGroupSpec(
    name="response_format",
    specs=[
        core.FeatureSpec(name="type", dtype=core.DataType.STRING),
        core.FeatureGroupSpec(
            name="json_schema",
            specs=[
                core.FeatureSpec(name="name", dtype=core.DataType.STRING),
                core.FeatureSpec(name="description", dtype=core.DataType.STRING),
                core.FeatureSpec(name="schema", dtype=core.DataType.OBJECT),
                core.FeatureSpec(name="strict", dtype=core.DataType.BOOL),
            ],
        ),
    ],
)

# ParamGroupSpec mirror of _RESPONSE_FORMAT_FEATURE_SPEC for the ParamSpec-based signatures.
_RESPONSE_FORMAT_PARAM_SPEC = core.ParamGroupSpec(
    name="response_format",
    default_value=None,
    specs=[
        core.ParamSpec(name="type", dtype=core.DataType.STRING, default_value="json_schema"),
        core.ParamGroupSpec(
            name="json_schema",
            default_value=None,
            specs=[
                core.ParamSpec(name="name", dtype=core.DataType.STRING, default_value=""),
                core.ParamSpec(name="description", dtype=core.DataType.STRING, default_value=None),
                core.ParamSpec(name="schema", dtype=core.DataType.OBJECT, default_value=None),
                core.ParamSpec(name="strict", dtype=core.DataType.BOOL, default_value=None),
            ],
        ),
    ],
)


_OPENAI_CHAT_SIGNATURE_SPEC = core.ModelSignature(
    inputs=[
        core.FeatureGroupSpec(
            name="messages",
            specs=[
                core.FeatureGroupSpec(
                    name="content",
                    specs=[
                        core.FeatureSpec(name="type", dtype=core.DataType.STRING),
                        # Text prompts
                        core.FeatureSpec(name="text", dtype=core.DataType.STRING),
                        # Image URL prompts
                        core.FeatureGroupSpec(
                            name="image_url",
                            specs=[
                                # Base64 encoded image URL or image URL
                                core.FeatureSpec(name="url", dtype=core.DataType.STRING),
                                # Image detail level (e.g., "low", "high", "auto")
                                core.FeatureSpec(name="detail", dtype=core.DataType.STRING),
                            ],
                        ),
                        # Video URL prompts
                        core.FeatureGroupSpec(
                            name="video_url",
                            specs=[
                                # Base64 encoded video URL
                                core.FeatureSpec(name="url", dtype=core.DataType.STRING),
                            ],
                        ),
                        # Audio prompts
                        core.FeatureGroupSpec(
                            name="input_audio",
                            specs=[
                                core.FeatureSpec(name="data", dtype=core.DataType.STRING),
                                core.FeatureSpec(name="format", dtype=core.DataType.STRING),
                            ],
                        ),
                    ],
                    shape=(-1,),
                ),
                core.FeatureSpec(name="name", dtype=core.DataType.STRING),
                core.FeatureSpec(name="role", dtype=core.DataType.STRING),
                core.FeatureSpec(name="title", dtype=core.DataType.STRING),
            ],
            shape=(-1,),
        ),
        core.FeatureSpec(name="temperature", dtype=core.DataType.DOUBLE),
        core.FeatureSpec(name="max_completion_tokens", dtype=core.DataType.INT64),
        core.FeatureSpec(name="stop", dtype=core.DataType.STRING, shape=(-1,)),
        core.FeatureSpec(name="n", dtype=core.DataType.INT32),
        core.FeatureSpec(name="stream", dtype=core.DataType.BOOL),
        core.FeatureSpec(name="top_p", dtype=core.DataType.DOUBLE),
        core.FeatureSpec(name="frequency_penalty", dtype=core.DataType.DOUBLE),
        core.FeatureSpec(name="presence_penalty", dtype=core.DataType.DOUBLE),
        _RESPONSE_FORMAT_FEATURE_SPEC,
    ],
    outputs=[
        core.FeatureSpec(name="id", dtype=core.DataType.STRING),
        core.FeatureSpec(name="object", dtype=core.DataType.STRING),
        core.FeatureSpec(name="created", dtype=core.DataType.FLOAT),
        core.FeatureSpec(name="model", dtype=core.DataType.STRING),
        core.FeatureGroupSpec(
            name="choices",
            specs=[
                core.FeatureSpec(name="index", dtype=core.DataType.INT32),
                core.FeatureGroupSpec(
                    name="message",
                    specs=[
                        core.FeatureSpec(name="content", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="name", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="role", dtype=core.DataType.STRING),
                    ],
                ),
                core.FeatureSpec(name="logprobs", dtype=core.DataType.STRING),
                core.FeatureSpec(name="finish_reason", dtype=core.DataType.STRING),
            ],
            shape=(-1,),
        ),
        core.FeatureGroupSpec(
            name="usage",
            specs=[
                core.FeatureSpec(name="completion_tokens", dtype=core.DataType.INT32),
                core.FeatureSpec(name="prompt_tokens", dtype=core.DataType.INT32),
                core.FeatureSpec(name="total_tokens", dtype=core.DataType.INT32),
            ],
        ),
    ],
)

_OPENAI_CHAT_SIGNATURE_WITH_PARAMS_SPEC = core.ModelSignature(
    inputs=[
        core.FeatureGroupSpec(
            name="messages",
            specs=[
                core.FeatureGroupSpec(
                    name="content",
                    specs=[
                        core.FeatureSpec(name="type", dtype=core.DataType.STRING),
                        # Text prompts
                        core.FeatureSpec(name="text", dtype=core.DataType.STRING),
                        # Image URL prompts
                        core.FeatureGroupSpec(
                            name="image_url",
                            specs=[
                                # Base64 encoded image URL or image URL
                                core.FeatureSpec(name="url", dtype=core.DataType.STRING),
                                # Image detail level (e.g., "low", "high", "auto")
                                core.FeatureSpec(name="detail", dtype=core.DataType.STRING),
                            ],
                        ),
                        # Video URL prompts
                        core.FeatureGroupSpec(
                            name="video_url",
                            specs=[
                                # Base64 encoded video URL
                                core.FeatureSpec(name="url", dtype=core.DataType.STRING),
                            ],
                        ),
                        # Audio prompts
                        core.FeatureGroupSpec(
                            name="input_audio",
                            specs=[
                                core.FeatureSpec(name="data", dtype=core.DataType.STRING),
                                core.FeatureSpec(name="format", dtype=core.DataType.STRING),
                            ],
                        ),
                    ],
                    shape=(-1,),
                ),
                core.FeatureSpec(name="name", dtype=core.DataType.STRING),
                core.FeatureSpec(name="role", dtype=core.DataType.STRING),
                core.FeatureSpec(name="title", dtype=core.DataType.STRING),
            ],
            shape=(-1,),
        ),
    ],
    outputs=[
        core.FeatureSpec(name="id", dtype=core.DataType.STRING),
        core.FeatureSpec(name="object", dtype=core.DataType.STRING),
        core.FeatureSpec(name="created", dtype=core.DataType.FLOAT),
        core.FeatureSpec(name="model", dtype=core.DataType.STRING),
        core.FeatureGroupSpec(
            name="choices",
            specs=[
                core.FeatureSpec(name="index", dtype=core.DataType.INT32),
                core.FeatureGroupSpec(
                    name="message",
                    specs=[
                        core.FeatureSpec(name="content", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="name", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="role", dtype=core.DataType.STRING),
                    ],
                ),
                core.FeatureSpec(name="logprobs", dtype=core.DataType.STRING),
                core.FeatureSpec(name="finish_reason", dtype=core.DataType.STRING),
            ],
            shape=(-1,),
        ),
        core.FeatureGroupSpec(
            name="usage",
            specs=[
                core.FeatureSpec(name="completion_tokens", dtype=core.DataType.INT32),
                core.FeatureSpec(name="prompt_tokens", dtype=core.DataType.INT32),
                core.FeatureSpec(name="total_tokens", dtype=core.DataType.INT32),
            ],
        ),
    ],
    params=[
        core.ParamSpec(name="temperature", dtype=core.DataType.DOUBLE, default_value=1.0),
        core.ParamSpec(name="max_completion_tokens", dtype=core.DataType.INT64, default_value=250),
        core.ParamSpec(name="stop", dtype=core.DataType.STRING, default_value=[], shape=(-1,)),
        core.ParamSpec(name="n", dtype=core.DataType.INT32, default_value=1),
        core.ParamSpec(name="stream", dtype=core.DataType.BOOL, default_value=False),
        core.ParamSpec(name="top_p", dtype=core.DataType.DOUBLE, default_value=1.0),
        core.ParamSpec(name="frequency_penalty", dtype=core.DataType.DOUBLE, default_value=0.0),
        core.ParamSpec(name="presence_penalty", dtype=core.DataType.DOUBLE, default_value=0.0),
        _RESPONSE_FORMAT_PARAM_SPEC,
    ],
)

_OPENAI_CHAT_SIGNATURE_SPEC_WITH_CONTENT_FORMAT_STRING = core.ModelSignature(
    inputs=[
        core.FeatureGroupSpec(
            name="messages",
            specs=[
                core.FeatureSpec(name="content", dtype=core.DataType.STRING),
                core.FeatureSpec(name="name", dtype=core.DataType.STRING),
                core.FeatureSpec(name="role", dtype=core.DataType.STRING),
                core.FeatureSpec(name="title", dtype=core.DataType.STRING),
            ],
            shape=(-1,),
        ),
        core.FeatureSpec(name="temperature", dtype=core.DataType.DOUBLE),
        core.FeatureSpec(name="max_completion_tokens", dtype=core.DataType.INT64),
        core.FeatureSpec(name="stop", dtype=core.DataType.STRING, shape=(-1,)),
        core.FeatureSpec(name="n", dtype=core.DataType.INT32),
        core.FeatureSpec(name="stream", dtype=core.DataType.BOOL),
        core.FeatureSpec(name="top_p", dtype=core.DataType.DOUBLE),
        core.FeatureSpec(name="frequency_penalty", dtype=core.DataType.DOUBLE),
        core.FeatureSpec(name="presence_penalty", dtype=core.DataType.DOUBLE),
        _RESPONSE_FORMAT_FEATURE_SPEC,
    ],
    outputs=[
        core.FeatureSpec(name="id", dtype=core.DataType.STRING),
        core.FeatureSpec(name="object", dtype=core.DataType.STRING),
        core.FeatureSpec(name="created", dtype=core.DataType.FLOAT),
        core.FeatureSpec(name="model", dtype=core.DataType.STRING),
        core.FeatureGroupSpec(
            name="choices",
            specs=[
                core.FeatureSpec(name="index", dtype=core.DataType.INT32),
                core.FeatureGroupSpec(
                    name="message",
                    specs=[
                        core.FeatureSpec(name="content", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="name", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="role", dtype=core.DataType.STRING),
                    ],
                ),
                core.FeatureSpec(name="logprobs", dtype=core.DataType.STRING),
                core.FeatureSpec(name="finish_reason", dtype=core.DataType.STRING),
            ],
            shape=(-1,),
        ),
        core.FeatureGroupSpec(
            name="usage",
            specs=[
                core.FeatureSpec(name="completion_tokens", dtype=core.DataType.INT32),
                core.FeatureSpec(name="prompt_tokens", dtype=core.DataType.INT32),
                core.FeatureSpec(name="total_tokens", dtype=core.DataType.INT32),
            ],
        ),
    ],
)

_OPENAI_CHAT_SIGNATURE_WITH_PARAMS_SPEC_WITH_CONTENT_FORMAT_STRING = core.ModelSignature(
    inputs=[
        core.FeatureGroupSpec(
            name="messages",
            specs=[
                core.FeatureSpec(name="content", dtype=core.DataType.STRING),
                core.FeatureSpec(name="name", dtype=core.DataType.STRING),
                core.FeatureSpec(name="role", dtype=core.DataType.STRING),
                core.FeatureSpec(name="title", dtype=core.DataType.STRING),
            ],
            shape=(-1,),
        ),
    ],
    outputs=[
        core.FeatureSpec(name="id", dtype=core.DataType.STRING),
        core.FeatureSpec(name="object", dtype=core.DataType.STRING),
        core.FeatureSpec(name="created", dtype=core.DataType.FLOAT),
        core.FeatureSpec(name="model", dtype=core.DataType.STRING),
        core.FeatureGroupSpec(
            name="choices",
            specs=[
                core.FeatureSpec(name="index", dtype=core.DataType.INT32),
                core.FeatureGroupSpec(
                    name="message",
                    specs=[
                        core.FeatureSpec(name="content", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="name", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="role", dtype=core.DataType.STRING),
                    ],
                ),
                core.FeatureSpec(name="logprobs", dtype=core.DataType.STRING),
                core.FeatureSpec(name="finish_reason", dtype=core.DataType.STRING),
            ],
            shape=(-1,),
        ),
        core.FeatureGroupSpec(
            name="usage",
            specs=[
                core.FeatureSpec(name="completion_tokens", dtype=core.DataType.INT32),
                core.FeatureSpec(name="prompt_tokens", dtype=core.DataType.INT32),
                core.FeatureSpec(name="total_tokens", dtype=core.DataType.INT32),
            ],
        ),
    ],
    params=[
        core.ParamSpec(name="temperature", dtype=core.DataType.DOUBLE, default_value=1.0),
        core.ParamSpec(name="max_completion_tokens", dtype=core.DataType.INT64, default_value=250),
        core.ParamSpec(name="stop", dtype=core.DataType.STRING, default_value=[], shape=(-1,)),
        core.ParamSpec(name="n", dtype=core.DataType.INT32, default_value=1),
        core.ParamSpec(name="stream", dtype=core.DataType.BOOL, default_value=False),
        core.ParamSpec(name="top_p", dtype=core.DataType.DOUBLE, default_value=1.0),
        core.ParamSpec(name="frequency_penalty", dtype=core.DataType.DOUBLE, default_value=0.0),
        core.ParamSpec(name="presence_penalty", dtype=core.DataType.DOUBLE, default_value=0.0),
        _RESPONSE_FORMAT_PARAM_SPEC,
    ],
)


# Centralized collection of all OpenAI chat signature specs for easy membership testing
# NEW SIGNATURES SHOULD BE ADDED TO THIS TUPLE
# Note: Using tuple instead of frozenset because ModelSignature is not hashable (has __eq__ but no __hash__)
_OPENAI_CHAT_SIGNATURE_SPECS = (
    _OPENAI_CHAT_SIGNATURE_SPEC,
    _OPENAI_CHAT_SIGNATURE_SPEC_WITH_CONTENT_FORMAT_STRING,
    _OPENAI_CHAT_SIGNATURE_WITH_PARAMS_SPEC,
    _OPENAI_CHAT_SIGNATURE_WITH_PARAMS_SPEC_WITH_CONTENT_FORMAT_STRING,
)

# Refer vLLM documentation: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/#chat-template

# Use this if you prefer to use the content format string instead of the default ChatML template.
# Most models support this.
OPENAI_CHAT_SIGNATURE_WITH_CONTENT_FORMAT_STRING = {"__call__": _OPENAI_CHAT_SIGNATURE_SPEC_WITH_CONTENT_FORMAT_STRING}

# This is the default signature.
# The content format allows vLLM to handler content parts like text, image, video, audio, file, etc.
OPENAI_CHAT_SIGNATURE = {"__call__": _OPENAI_CHAT_SIGNATURE_SPEC}

# Use this signature to leverage ParamSpec with the default ChatML template.
OPENAI_CHAT_WITH_PARAMS_SIGNATURE = {"__call__": _OPENAI_CHAT_SIGNATURE_WITH_PARAMS_SPEC}

# Use this signature to leverage ParamSpec with the content format string.
OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING = {
    "__call__": _OPENAI_CHAT_SIGNATURE_WITH_PARAMS_SPEC_WITH_CONTENT_FORMAT_STRING
}
