# Preparação e Treinamento de Modelo BART para Geração de Resumos

## Introdução

Este projeto foi desenvolvido com o objetivo de criar um pipeline completo para o treinamento de um modelo BART (Bidirectional and Auto-Regressive Transformers) voltado para a geração de resumos. A partir de um dataset que contém títulos e descrições de produtos, nosso modelo é capaz de gerar resumos automáticos, otimizando processos que exigem grande volume de informação.

## Objetivos

- **Processar e preparar o dataset**: Limpeza, formatação e organização dos dados para alimentar o modelo de maneira eficiente.
- **Configuração do ambiente de desenvolvimento**: Preparar as ferramentas e bibliotecas necessárias.
- **Estruturar os dados de entrada e saída**: Gerar prompts e respostas que servirão como base para o treinamento.
- **Tokenizar o texto**: Transformar o texto em tokens utilizáveis pelo modelo BART.
- **Divisão do dataset**: Separação dos dados para garantir a correta validação e treinamento.
- **Configuração do treinamento**: Definir os parâmetros essenciais para o treinamento otimizado do modelo.
- **Treinamento e utilização do modelo**: Treinar o modelo e utilizá-lo para gerar resumos com base nos dados fornecidos.

## Requisitos

Para rodar este projeto, certifique-se de que seu ambiente de desenvolvimento esteja configurado corretamente. Você precisará das seguintes ferramentas e bibliotecas:

- Python 3.7 ou superior
- Gerenciador de pacotes Pip
- As seguintes bibliotecas Python:
  - `transformers`
  - `datasets`
  - `torch`
  - `pandas`
  - `numpy`

## Instalação

Para instalar todas as dependências necessárias, basta rodar o comando abaixo no seu terminal:

```bash
pip install transformers datasets pandas numpy torch
```

## Preparação dos Dados

### 1. Importação das Bibliotecas

O primeiro passo é importar as bibliotecas que serão utilizadas para carregar e processar os dados.

```python
import json
import random
import torch
import pandas as pd
import re
import numpy as np

from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
from datasets import Dataset
```

### 2. Carregamento e Limpeza dos Dados

Para alimentar o modelo, precisamos de dados limpos e formatados. Neste exemplo, vamos carregar um arquivo JSON que contém os títulos e descrições dos produtos, remover entradas duplicadas e limpar textos inadequados.

```python
# Lista para armazenar dados processados
json_new = []

# Leitura do arquivo JSON
with open('/content/drive/MyDrive/Colab Notebooks/trn.json', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            item = json.loads(line)
            new_item = {
                "title": item["title"],
                "content": item["content"]
            }
            json_new.append(new_item)
        except json.JSONDecodeError as e:
            print(f"Erro ao processar linha: {e}")

# Converter dados em DataFrame para facilitar a manipulação
df = pd.DataFrame(json_new)

# Remover linhas com valores nulos
df = df[(df['title'].str.strip() != '') & (df['content'].str.strip() != '')]

# Remover duplicatas
df = df.drop_duplicates(subset=['title', 'content'])

# Função para limpeza de texto
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.strip().lower())
    return text

# Aplicando a função de limpeza
df['title'] = df['title'].apply(clean_text)
df['content'] = df['content'].apply(clean_text)

# Remover descrições muito curtas
df = df[df['content'].str.split().str.len() > 5]

# Dividindo o dataset em 4 partes para facilitar o processamento
df_split = np.array_split(df, 4)

# Salvando os dados em arquivos JSON separados
for i, df_part in enumerate(df_split, start=1):
    output_filename = f'./json_parte_{i}.json'
    df_part.to_json(output_filename, orient='records', lines=True, force_ascii=False)
    print(f"Parte {i} salva em '{output_filename}'.")
```

### 3. Amostragem dos Dados

Para testar o modelo sem gastar muitos recursos computacionais, utilizamos uma amostra de 10% do dataset processado.

```python
# Carregar e amostrar 10% dos dados para teste
data = []
with open(f'./json_parte_1.json', 'r') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue

# Reduzir o dataset para 10% dos dados originais
data = random.sample(data, int(0.10 * len(data)))
```

### 4. Criação de Prompts e Respostas

O modelo precisa de dados formatados de maneira adequada. Neste caso, criamos prompts com perguntas baseadas nos títulos dos produtos e utilizamos as descrições como respostas.

```python
# Estruturando os dados
processed_data = []

for item in data:
    prompt = "What is the summary of " + item['title'] + "?"
    response = item['content']
    processed_data.append({'prompt': prompt, 'response': response})

processed_data_dict = {
    'prompt': [d['prompt'] for d in processed_data],
    'response': [d['response'] for d in processed_data]
}
```

## Treinamento do Modelo

### 1. Carregamento do Modelo BART

Utilizamos o modelo BART pré-treinado da biblioteca `transformers`. Esse modelo será ajustado aos nossos dados para geração de resumos.

```python
# Carregar o tokenizer e o modelo BART pré-treinado
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
```

### 2. Tokenização dos Dados

Transformamos nossos dados textuais em tokens que o modelo possa compreender.

```python
# Função de tokenização
def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], max_length=256, truncation=True, padding='max_length')
    targets = tokenizer(examples['response'], max_length=256, truncation=True, padding='max_length')
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids']
    }

# Aplicando a tokenização
tokenized_dataset = Dataset.from_dict(processed_data_dict).map(tokenize_function, batched=True)
```

### 3. Divisão do Dataset

Dividimos os dados em treino e validação para monitorar o desempenho do modelo durante o treinamento.

```python
# Dividir o dataset em treino e validação
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)  # 80% treino, 20% validação
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
```

### 4. Configuração dos Argumentos de Treinamento

Definimos os parâmetros de treinamento, como taxa de aprendizado e número de épocas.

```python
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    logging_dir='./logs',
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=True
)
```

### 5. Inicialização e Treinamento do Modelo

Com todos os dados preparados e os parâmetros configurados, podemos iniciar o treinamento.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
```

## Uso do Modelo Treinado

Após o treinamento, podemos utilizar o modelo para gerar resumos de novos produtos.

```python
def generate_summary(title, description):
    prompt = f"What is the summary of {title}?"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Exemplo de uso
summary = generate_summary("Example Product Title", "Example product description.")
print(f"Resumo gerado: {summary}")
```

## Considerações Finais

O projeto demonstrou a preparação de um pipeline completo para a geração de resumos utilizando o modelo BART, desde a limpeza e preparação dos dados até o treinamento e a geração de resumos. Esse processo é altamente escalável e pode ser ajustado para diferentes tipos de dados ou modelos.
