from snowflake.ml.model._signatures import core

_OPENAI_CHAT_SIGNATURE_SPEC = core.ModelSignature(
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

OPENAI_CHAT_SIGNATURE = {"__call__": _OPENAI_CHAT_SIGNATURE_SPEC}
