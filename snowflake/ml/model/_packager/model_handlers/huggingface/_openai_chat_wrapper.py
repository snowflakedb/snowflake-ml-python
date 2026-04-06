import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

from packaging import version

if TYPE_CHECKING:
    import torch
    import transformers

logger = logging.getLogger(__name__)


class HuggingFaceOpenAICompatibleModel:
    """
    A class to wrap a Hugging Face text generation model and provide an
    OpenAI-compatible chat completion interface.
    """

    def __init__(self, pipeline: "transformers.Pipeline") -> None:
        """
        Initializes the model and tokenizer.

        Args:
            pipeline (transformers.pipeline): The Hugging Face pipeline to wrap.
        """

        self.pipeline = pipeline
        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer
        self.model_name = self.pipeline.model.name_or_path

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        """
        Applies a chat template to a list of messages.

        Args:
            messages (list[dict]): A list of message dictionaries.

        Returns:
            The formatted prompt string ready for model input.
        """

        final_messages = []
        for message in messages:
            if isinstance(message.get("content", ""), str):
                final_messages.append({"role": message.get("role", "user"), "content": message.get("content", "")})
            else:
                # extract only the text from the content
                # sample data:
                # {
                #     "role": "user",
                #     "content": [
                #         {"type": "text", "text": "Hello, how are you?"}, # extracted
                #         {"type": "image", "image": "https://example.com/image.png"}, # not extracted
                #     ],
                # }
                for content_part in message.get("content", []):
                    if content_part.get("type", "") == "text":
                        final_messages.append(
                            {"role": message.get("role", "user"), "content": content_part.get("text", "")}
                        )
                    # TODO: implement other content types

        # Use the tokenizer's apply_chat_template method.
        # We ensured a template exists in __init__.
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(  # type: ignore[no-any-return]
                final_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback for very old transformers without apply_chat_template
        # Manually apply ChatML-like formatting
        prompt = ""
        for message in final_messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _get_stopping_criteria(self, stop_strings: list[str]) -> "transformers.StoppingCriteriaList":

        import transformers

        class StopStringsStoppingCriteria(transformers.StoppingCriteria):
            def __init__(self, stop_strings: list[str], tokenizer: Any) -> None:
                self.stop_strings = stop_strings
                self.tokenizer = tokenizer

            def __call__(self, input_ids: "torch.Tensor", scores: "torch.Tensor", **kwargs: Any) -> bool:
                # Decode the generated text for each sequence
                for i in range(input_ids.shape[0]):
                    generated_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    # Check if any stop string appears in the generated text
                    for stop_str in self.stop_strings:
                        if stop_str in generated_text:
                            return True
                return False

        return transformers.StoppingCriteriaList([StopStringsStoppingCriteria(stop_strings, self.tokenizer)])

    def generate_chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        stop_strings: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        n: int = 1,
    ) -> dict[str, Any]:
        """
        Generates a chat completion response in an OpenAI-compatible format.

        Args:
            messages (list[dict]): A list of message dictionaries, e.g.,
                                   [{"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": "What is deep learning?"}]
            max_completion_tokens (int): The maximum number of completion tokens to generate.
            stop_strings (list[str]): A list of strings to stop generation.
            temperature (float): The temperature for sampling. 0 means greedy decoding.
            top_p (float): The top-p value for nucleus sampling.
            stream (bool): Whether to stream the generation (not yet supported).
            frequency_penalty (float): The frequency penalty for sampling (maps to repetition_penalty).
            presence_penalty (float): The presence penalty for sampling (not directly supported).
            n (int): The number of samples to generate.

        Returns:
            dict: An OpenAI-compatible dictionary representing the chat completion.
        """
        # Apply chat template to convert messages into a single prompt string
        prompt_text = self._apply_chat_template(messages)

        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        if stream:
            logger.warning(
                "Streaming is not supported using transformers.Pipeline implementation. Ignoring stream=True."
            )
            stream = False

        if presence_penalty is not None:
            logger.warning(
                "Presence penalty is not supported using transformers.Pipeline implementation."
                " Ignoring presence_penalty."
            )
            presence_penalty = None

        import transformers

        transformers_version = version.parse(transformers.__version__)

        # Stop strings are supported in transformers >= 4.43.0
        can_handle_stop_strings = transformers_version >= version.parse("4.43.0")

        # Determine sampling based on temperature (following serve.py logic)
        # Default temperature to 1.0 if not specified
        actual_temperature = temperature if temperature is not None else 1.0
        do_sample = actual_temperature > 0.0

        # Set up generation config following best practices from serve.py
        generation_config = transformers.GenerationConfig(
            max_new_tokens=max_completion_tokens if max_completion_tokens is not None else 1024,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
        )

        # Only set temperature and top_p if sampling is enabled
        if do_sample:
            generation_config.temperature = actual_temperature
            if top_p is not None:
                generation_config.top_p = top_p

        # Handle repetition penalty (mapped from frequency_penalty)
        if frequency_penalty is not None:
            # OpenAI's frequency_penalty is typically in range [-2.0, 2.0]
            # HuggingFace's repetition_penalty is typically > 0, with 1.0 = no penalty
            # We need to convert: frequency_penalty=0 -> repetition_penalty=1.0
            # Higher frequency_penalty should increase repetition_penalty
            generation_config.repetition_penalty = 1.0 + (frequency_penalty if frequency_penalty > 0 else 0)

        # For multiple completions (n > 1), use sampling not beam search
        if n > 1:
            generation_config.num_return_sequences = n
            # Force sampling on for multiple sequences
            if not do_sample:
                logger.warning("Forcing do_sample=True for n>1. Consider setting temperature > 0 for better diversity.")
                generation_config.do_sample = True
                generation_config.temperature = 1.0
        else:
            generation_config.num_return_sequences = 1

        # Handle stop strings if provided
        stopping_criteria = None
        if stop_strings and not can_handle_stop_strings:
            logger.warning("Stop strings are not supported in transformers < 4.41.0. Ignoring stop strings.")

        if stop_strings and can_handle_stop_strings:
            stopping_criteria = self._get_stopping_criteria(stop_strings)
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config,
                # Pass tokenizer for proper handling of stop strings
                tokenizer=self.tokenizer,
                stopping_criteria=stopping_criteria,
            )
        else:
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config,
            )

        # Generate text
        # Handle the case where output might be 1D if n=1
        if output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)

        generated_texts = []
        completion_tokens = 0
        total_tokens = prompt_tokens

        for output_id in output_ids:
            # The output_ids include the input prompt
            # Decode the generated text, excluding the input prompt
            # so we slice to get only new tokens
            generated_tokens = output_id[prompt_tokens:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Trim stop strings from generated text if they appear
            # The stop criteria would stop generating further tokens, so we need to trim the generated text
            if stop_strings and can_handle_stop_strings:
                for stop_str in stop_strings:
                    if stop_str in generated_text:
                        # Find the first occurrence and trim everything from there
                        stop_idx = generated_text.find(stop_str)
                        generated_text = generated_text[:stop_idx]
                        break  # Stop after finding the first stop string

            generated_texts.append(generated_text)

            # Calculate completion tokens
            completion_tokens += len(generated_tokens)
            total_tokens += len(generated_tokens)

        choices = []
        for i, generated_text in enumerate(generated_texts):
            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": generated_text},
                    "logprobs": None,  # Not directly supported in this basic implementation
                    "finish_reason": "stop",  # Assuming stop for simplicity
                }
            )

        # Construct OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        return response
