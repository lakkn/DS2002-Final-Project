# rag_pipeline/llm.py
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_NAME = "microsoft/Phi-3-mini-4k-instruct"


class LocalLLM:
    def __init__(self, model_name: str = LLM_NAME):
        """
        Loads a local open-source model
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # a little bit of output cleanup if necessary
        if "Answer:" in text:
            text = text.split("Answer:", 1)[1].strip()
        return text.strip()
