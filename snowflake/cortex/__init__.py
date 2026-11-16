# Helper imports are underscore-prefixed so they don't show up as public attributes of snowflake.cortex.
import importlib as _importlib
import pkgutil as _pkgutil
import typing as _typing

# Other distributions can install subpackages under snowflake.cortex, including into a different sys.path entry.
# Only snowflake-ml-python may ship this __init__.py; other distributions ship just their own subfolders.
__path__ = _pkgutil.extend_path(__path__, __name__)

# The public API is imported on first use (PEP 562), so importing snowflake.cortex or one of those subpackages
# does not load Snowpark.
_LAZY_IMPORTS: dict[str, str] = {
    "Candidate": "snowflake.cortex._candidate",
    "ClassifyText": "snowflake.cortex._classify_text",
    "classify_text": "snowflake.cortex._classify_text",
    "Complete": "snowflake.cortex._complete",
    "complete": "snowflake.cortex._complete",
    "CompleteOptions": "snowflake.cortex._complete",
    "ConversationMessage": "snowflake.cortex._complete",
    "EmbedText768": "snowflake.cortex._embed_text_768",
    "embed_text_768": "snowflake.cortex._embed_text_768",
    "EmbedText1024": "snowflake.cortex._embed_text_1024",
    "embed_text_1024": "snowflake.cortex._embed_text_1024",
    "ExtractAnswer": "snowflake.cortex._extract_answer",
    "extract_answer": "snowflake.cortex._extract_answer",
    "evaluate": "snowflake.cortex._evaluate",
    "EvaluationJob": "snowflake.cortex._evaluation_job",
    "JobStatus": "snowflake.cortex._experiment_job",
    "EvaluationMetric": "snowflake.cortex._evaluation_spec",
    "EvaluationResult": "snowflake.cortex._evaluation_result",
    "Finetune": "snowflake.cortex._finetune",
    "FinetuneJob": "snowflake.cortex._finetune",
    "FinetuneStatus": "snowflake.cortex._finetune",
    "OptimizationJob": "snowflake.cortex._optimization_job",
    "OptimizationResult": "snowflake.cortex._optimization_result",
    "optimize_prompt": "snowflake.cortex._optimize_prompt",
    "Sentiment": "snowflake.cortex._sentiment",
    "sentiment": "snowflake.cortex._sentiment",
    "Summarize": "snowflake.cortex._summarize",
    "summarize": "snowflake.cortex._summarize",
    "Translate": "snowflake.cortex._translate",
    "translate": "snowflake.cortex._translate",
}

__all__ = [
    "Candidate",
    "ClassifyText",
    "classify_text",
    "Complete",
    "complete",
    "CompleteOptions",
    "ConversationMessage",
    "EmbedText768",
    "embed_text_768",
    "EmbedText1024",
    "embed_text_1024",
    "ExtractAnswer",
    "extract_answer",
    "evaluate",
    "EvaluationJob",
    "JobStatus",
    "EvaluationMetric",
    "EvaluationResult",
    "Finetune",
    "FinetuneJob",
    "FinetuneStatus",
    "OptimizationJob",
    "OptimizationResult",
    "optimize_prompt",
    "Sentiment",
    "sentiment",
    "Summarize",
    "summarize",
    "Translate",
    "translate",
]

if _typing.TYPE_CHECKING:
    # Static analyzers and IDEs can't see names resolved by __getattr__; keep this in sync with __all__.
    from snowflake.cortex._candidate import Candidate
    from snowflake.cortex._classify_text import ClassifyText, classify_text
    from snowflake.cortex._complete import (
        Complete,
        CompleteOptions,
        ConversationMessage,
        complete,
    )
    from snowflake.cortex._embed_text_768 import EmbedText768, embed_text_768
    from snowflake.cortex._embed_text_1024 import EmbedText1024, embed_text_1024
    from snowflake.cortex._evaluate import evaluate
    from snowflake.cortex._evaluation_job import EvaluationJob
    from snowflake.cortex._evaluation_result import EvaluationResult
    from snowflake.cortex._evaluation_spec import EvaluationMetric
    from snowflake.cortex._experiment_job import JobStatus
    from snowflake.cortex._extract_answer import ExtractAnswer, extract_answer
    from snowflake.cortex._finetune import Finetune, FinetuneJob, FinetuneStatus
    from snowflake.cortex._optimization_job import OptimizationJob
    from snowflake.cortex._optimization_result import OptimizationResult
    from snowflake.cortex._optimize_prompt import optimize_prompt
    from snowflake.cortex._sentiment import Sentiment, sentiment
    from snowflake.cortex._summarize import Summarize, summarize
    from snowflake.cortex._translate import Translate, translate


def __getattr__(name: str) -> _typing.Any:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
