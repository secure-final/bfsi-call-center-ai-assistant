"""Tier 2: Small language model inference. Optional LoRA adapters."""
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT, load_config
from src.logging_config import get_logger

logger = get_logger(__name__)


def _alpaca_prompt(instruction: str, input_text: str = "", context: str = "") -> str:
    if context:
        return (
            f"Below is an instruction that describes a task, along with context from our knowledge base. "
            f"Write a response that uses only the context when giving specific numbers or policies.\n\n"
            f"### Context:\n{context}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text or 'N/A'}\n\n"
            f"### Response:\n"
        )
    return (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text or 'N/A'}\n\n"
        f"### Response:\n"
    )


class SLMInference:
    """Load base model (and optional PEFT adapters) and generate responses."""

    def __init__(
        self,
        base_model_name: str | None = None,
        adapter_path: Path | str | None = None,
        use_4bit: bool | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ):
        cfg = load_config()
        slm_cfg = cfg.get("slm", {})
        self.base_model_name = base_model_name or slm_cfg.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        adapter = adapter_path or slm_cfg.get("adapter_path")
        self.adapter_path = Path(adapter) if adapter else None
        if self.adapter_path and not self.adapter_path.is_absolute():
            self.adapter_path = PROJECT_ROOT / self.adapter_path
        self.use_4bit = use_4bit if use_4bit is not None else slm_cfg.get("use_4bit", False)
        self.max_new_tokens = max_new_tokens or slm_cfg.get("max_new_tokens", 256)
        self.temperature = temperature if temperature is not None else slm_cfg.get("temperature", 0.3)
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> bool:
        """Load model and tokenizer. Returns True on success, False on failure."""
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading tokenizer: %s", self.base_model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, trust_remote_code=True
            )
            model_kwargs = {"trust_remote_code": True}
            use_4bit = self.use_4bit
            if use_4bit:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available; loading in full precision")
                    use_4bit = False
            logger.info("Loading model: %s (4bit=%s)", self.base_model_name, use_4bit)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, **model_kwargs
            )
            if self.adapter_path and self.adapter_path.exists():
                try:
                    from peft import PeftModel
                    self._model = PeftModel.from_pretrained(
                        self._model, str(self.adapter_path)
                    )
                    self._model = self._model.merge_and_unload()
                    logger.info("Loaded PEFT adapters from %s", self.adapter_path)
                except Exception as e:
                    logger.warning("Could not load adapters from %s: %s", self.adapter_path, e)
            self._model.eval()
            return True
        except Exception as e:
            logger.exception("Failed to load SLM: %s", e)
            return False

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        context: str = "",
    ) -> str:
        """Generate response for the given instruction (and optional input/context). Returns fallback message on failure."""
        fallback = (
            "I could not generate a specific response for that. "
            "Please rephrase your question, or contact our customer care for detailed assistance."
        )
        if not self._load_model():
            return fallback
        try:
            prompt = _alpaca_prompt(instruction, input_text, context)
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            device = (
                self._model.device
                if hasattr(self._model, "device")
                else next(self._model.parameters()).device
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            import torch
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            reply = self._tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            text = reply.strip()
            return text if text else fallback
        except Exception as e:
            logger.exception("SLM generate failed: %s", e)
            return fallback
