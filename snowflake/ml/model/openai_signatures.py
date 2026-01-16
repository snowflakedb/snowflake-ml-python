from snowflake.ml.model._signatures import core

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


# Refer vLLM documentation: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/#chat-template

# Use this if you prefer to use the content format string instead of the default ChatML template.
# Most models support this.
OPENAI_CHAT_SIGNATURE_WITH_CONTENT_FORMAT_STRING = {"__call__": _OPENAI_CHAT_SIGNATURE_SPEC_WITH_CONTENT_FORMAT_STRING}

# This is the default signature.
# The content format allows vLLM to handler content parts like text, image, video, audio, file, etc.
OPENAI_CHAT_SIGNATURE = {"__call__": _OPENAI_CHAT_SIGNATURE_SPEC}
