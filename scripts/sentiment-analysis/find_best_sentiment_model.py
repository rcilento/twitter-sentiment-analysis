from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
from utils import read_clean_data, analyze_sentiment
from typing import List, Dict, Tuple

def evaluate_model(y_true: List[int], y_pred: List[int]) -> Tuple[float, str]:
    """
    Avalia o desempenho das predições de sentimento.

    Args:
        y_true (List[int]): Lista de rótulos verdadeiros.
        y_pred (List[int]): Lista de rótulos preditos.

    Returns:
        Tuple[float, str]: Acurácia e relatório de classificação.
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"])
    return accuracy, report

def compare_models(texts: List[str], y_true: List[int], models_map: Dict[str, any]) -> Dict[str, any]:
    """
    Compara múltiplos modelos de análise de sentimento.

    Args:
        texts (List[str]): Lista de textos para análise.
        y_true (List[int]): Lista de rótulos verdadeiros.
        models_map (Dict[str, any]): Dicionário de modelos a serem comparados.

    Returns:
        Dict: Dicionário contendo métricas de avaliacao para cada modelo em models_map.
    """
    evaluations = {}
    for model_name, model in models_map.items():
        print(f"Analisando sentimentos com o modelo: {model_name}")
        evaluations[model_name] = {}
        y_pred = analyze_sentiment(model, texts)
        accuracy, report = evaluate_model(y_true, y_pred)
        evaluations[model_name] ['accuracy'] = accuracy
        evaluations[model_name]['report'] = report
        print(f"Accuracy para {model_name}: {accuracy}")
        print(f"Relatório de classificação para {model_name}:\n{report}")
    return evaluations

def find_best_model(evaluations: Dict[str, any]) -> str:
    """
    Encontra o melhor modelo com base na acurácia.

    Args:
        evaluations (List[Tuple[str, float, str]]): Lista de avaliações de modelos.

    Returns:
        str: Nome do melhor modelo.
    """
    best_acc = 0
    best_model = None
    for model_name, metrics in evaluations.items():
        if metrics['accuracy'] > best_acc:
            best_model = model_name
            best_acc = metrics['accuracy']
    return best_model

def save_best_model(models_map: Dict[str, any], best_model: str, save_path: str):
    """
    Salva o melhor modelo no diretório especificado.

    Args:
        models_map (Dict[str, any]): Dicionário de modelos.
        best_model (str): Nome do melhor modelo.
        save_path (str): Caminho para salvar o modelo.
    """
    models_map[best_model].save_pretrained(save_path)

def save_models_metrics(filepath, evaluations):
    for model_name, metrics in evaluations.items():
        report = metrics['report']
        with open(filepath + f'report_{model_name}.txt', 'w', encoding='utf-8') as file:
            file.write(report)

def main():
    models = {
        "twitter-roberta-base-sentiment-latest": pipeline("sentiment-analysis",
                                                          model="cardiffnlp/twitter-roberta-base-sentiment-latest"),
        "bertweet-base-sentiment-analysis": pipeline("sentiment-analysis",
                                                     model="finiteautomata/bertweet-base-sentiment-analysis"),
        "distilbert-base-uncased-finetuned-sst-2-english": pipeline("sentiment-analysis",
                                                                    model="distilbert-base-uncased-finetuned-sst-2-english"),
        "robust-sentiment-analysis": pipeline("sentiment-analysis",
                                              model="tabularisai/robust-sentiment-analysis")
    }
    df = read_clean_data()
    df = df[~df['sentimento_tweet'].isna()]

    tweets = df["texto_tweet"].tolist()
    y = df["sentimento_tweet"]

    results = compare_models(tweets, y, models)
    best_model = find_best_model(results)
    save_best_model(models, best_model, f'models/{best_model}')
    save_models_metrics('outputs/model_metrics/', results)

if __name__ == "__main__":
    main()
