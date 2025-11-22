import torch
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama

class Model(ABC):
    @abstractmethod
    def answer(self, prompt):
        pass

    @abstractmethod
    def get_name(self):
        pass

class Qwen3_4b_GGUF(Model):
    def __init__(self):
        self.model = Llama.from_pretrained(
            repo_id="unsloth/Qwen3-4B-GGUF",
            filename="Qwen3-4B-Q4_0.gguf"
        )

    def answer(self, prompt):
        prompt += "/no_think\n\n"
        response = self.model.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response['choices'][0]['message']['content']
    
    def get_name(self):
        return "qwen3_4b_gguf"

class Qwen3_4b(Model):
    def __init__(self):
        model_name = "Qwen/Qwen3-4B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def answer(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32000
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        #thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        #print("thinking content:", thinking_content)
        return content

    def get_name(self):
        return "qwen3_4b"