import pandas as pd
import glob
import json
from typing import List, Dict, Any


def read_clean_data() -> pd.DataFrame:
    """
    Lê todos os arquivos CSV do diretório data/clean_data e os concatena em um único DataFrame.

    Returns:
        pd.DataFrame: DataFrame contendo os dados concatenados.
    """
    files = glob.glob("data/clean_data/*.csv")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs)


def analyze_sentiment(model, texts: List[str]) -> List[int]:
    """
    Analisa o sentimento dos textos usando um modelo de linguagem e mapeia as etiquetas para valores numéricos.

    Args:
        model: Modelo de linguagem para análise de sentimento.
        texts (List[str]): Lista de textos para análise.

    Returns:
        List[int]: Lista de predições de sentimento em formato numérico.
    """
    predictions = model(texts)
    label_map = {
        "VERY NEGATIVE": 0, "NEGATIVE": 0, "NEG": 0, "LABEL_0": 0, "LABEL_1": 0,
        "NEUTRAL": 1, "NEU": 1, "LABEL_2": 1,
        "VERY POSITIVE": 2, "POSITIVE": 2, "POS": 2, "LABEL_3": 2, "LABEL_4": 2
    }
    return [label_map.get(pred["label"].upper(), -1) for pred in
            predictions]  # Retorna -1 se a etiqueta não for encontrada


