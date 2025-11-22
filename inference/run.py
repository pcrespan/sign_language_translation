import time
from llm import Qwen3_4b

model = Qwen3_4b()

while (1):
    prompt = input("Prompt: ")
    start = time.time()
    print(model.answer(prompt))
    end = time.time()
    duration = end - start
    print(f"Finished in {duration:.2f} seconds.")