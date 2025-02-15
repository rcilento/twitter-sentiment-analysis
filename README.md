# Twitter Sentiment Analysis and Insights

## Introdução

Este projeto tem como objetivo realizar a análise de sentimento de tweets relacionados à AirFrance e gerar insights acionáveis a partir do feedback dos clientes. Utilizando modelos de linguagem natural (transformers) e técnicas de Recuperação Aumentada por Geração (RAG), o projeto identifica pontos críticos de melhoria e destaca aspectos positivos do serviço da AirFrance.

## Estrutura do Projeto

**data/raw_input/**: Armazena arquivos csv de input ao projeto

**data/clean_data/**: Recebe os arquivos csv após limpeza feita no preprocessamento

**models/**: Recebe os modelo de análise de sentimento vencedor da comparação de modelos.

**outputs/**: Recebe os resultados de saída dos scripts.

**prompts/**: Armazena os prompts para serem executados na RAG.

**scripts/preprocessing/**: Scripts relacionados ao processamento de dados, como limpeza e preparação dos dados.

**scripts/sentiment-analysis/**: Scripts relacionados à escolha e predição de modelos de análise de sentimento.

**scripts/rag/**: Scripts relacionados à Recuperação Aumentada por Geração (RAG).

**scripts/reporting/**: Scripts relacionados à construção de relatórios.

## Instalação

Siga as etapas abaixo para configurar o ambiente e instalar as dependências necessárias:

1. Clone o repositório:
    ```bash
    git clone https://github.com/rcilento/tweets-sentiment-analysis.git
    cd tweets-sentiment-analysis
    ```

2. Crie um ambiente virtual com python 3.9 e ative-o:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure as variáveis de ambiente (adicione suas chaves de API no arquivo `.env`).
- OPENAI_API_KEY

## Uso

### 1. Processamento dos Dados

Execute o script de processamento de dados para limpar e preparar os tweets:

```bash
python scripts/preprocessing/preprocess_raw_input.py
```

### 2. Análise de Sentimento

Compare diferentes modelos de análise de sentimento para selecionar o melhor:

```bash
python scripts/sentiment-analysis/find_best_sentiment_model.py
```

Prediga os sentimentos dos tweets utilizando o melhor modelo:

```bash
python scripts/sentiment-analysis/analyse_tweets_sentiment.py
```

### 3. Construção da Loja de Vetores

Crie a vectorstore a partir dos tweets processados:

```bash
python scripts/rag/create_vectorstore.py
```


### 4. Geração de Insights

Execute o script para gerar insights acionáveis a partir dos tweets:

```bash
python scripts/rag/generate_rag_insights.py
```

### 5. Geração de Relatório

Crie um relatório de sentimentos e insights utilizando Streamlit:

```bash
streamlit run scripts/reporting/create_report.py
```

