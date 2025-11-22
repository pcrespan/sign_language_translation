import json
import requests

def save_benchmark(benchmark):
    with open('../dataset/benchmark.json', 'w') as f:
        json.dump(benchmark, f)

with open("../dataset/dataset.json", "r") as f:
    dataset = json.load(f)

outputs = []

for item in dataset["benchmark"]:
    text = item["input"]
    url = "https://traducao2.vlibras.gov.br/translate"

    data = {
        "text": text
    }

    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Resposta:", response.text)
        benchmark = {}
        benchmark["input"] = text
        benchmark["output"] = response.text
        outputs.append(benchmark)
    else:
        print(f"Erro: {response.status_code}")

save_benchmark(outputs)