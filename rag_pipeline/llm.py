from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_NAME = "microsoft/Phi-3-mini-4k-instruct"

#simple llm setup for the ragbot
class LocalLLM:
    def __init__(self, model_name: str = LLM_NAME, max_input_tokens: int = 1024):
        """
        Loads a local open-source model
        """
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )


        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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
