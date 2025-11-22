import json
import time
import argparse
import sys
import os
from tqdm import tqdm
from llm import Qwen3_4b, Qwen3_4b_GGUF

BASE_PATH = "../dataset"

def create_prompt(phrase):
    prompt = """Quero que você adapte um texto em português para a Linguagem Brasileira de Sinais (LIBRAS).
    
    ### Exemplos de entrada e saída esperadas:
    
    *Exemplo 1*  
    *Entrada:*
    Eu fui ao banco ontem.
    
    *Saída esperada:*
    EU IR BANCO&DINHEIRO ONTEM
    
    *Exemplo 2*  
    *Entrada:*
    Ela vai à escola amanhã
    
    *Saída esperada:*
    ELA IR ESCOLA AMANHÃ
    
    ### Regra para a sua resposta
    - Retornar apenas a frase adaptada, e nada além disso.
    
    Agora, analise a seguinte frase e responda de acordo com a regra.\n\n  
    """

    prompt += phrase
    return prompt

def load_dataset(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_model_results(path, results):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(results, f)

def run_benchmark(model):
    DATASET_PATH = f"{BASE_PATH}/benchmark.json"
    RESULTS_FILENAME = f"./results/{model.get_name()}_results.json"

    dataset = load_dataset(DATASET_PATH)
    model_answers = {}
    model_answers["answers"] = []
    answer = {}
    start = time.time()

    for phrase in tqdm(dataset, desc="Phrases", unit="phrase"):
        answer["input"] = phrase["input"]
        answer["output"] = phrase["output"]
        output = model.answer(create_prompt(phrase["input"]))
        print(output)
        answer["model_output"] = output
        model_answers["answers"].append(answer)

    end = time.time()
    duration_minutes = (end - start) / 60
    model_answers["duration"] = duration_minutes

    print(f"Finished in {duration_minutes:.2f} minutes.")
    write_model_results(RESULTS_FILENAME, model_answers)

def main():
    models = [Qwen3_4b_GGUF, Qwen3_4b]

    for model in models:
        run_benchmark(model())

if __name__ == '__main__':
    main()