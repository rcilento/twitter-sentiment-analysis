import os
import pandas as pd
import re


def preprocess_input(in_filename: str, out_filename: str):
    """
    Lê um arquivo de entrada, realiza substituições de caracteres e salva em um novo arquivo.

    Args:
        in_filename (str): Caminho do arquivo de entrada.
        out_filename (str): Caminho do arquivo de saída.
    """
    with open(in_filename, 'r', encoding="utf8") as infile, open(out_filename, 'w', encoding="utf8") as outfile:
        data = infile.read()
        data = re.sub(r'\;+', '', data)
        data = re.sub(r'\"+', '"', data)
        outfile.write(data)


def read_twitter_posts_file(f: str) -> pd.DataFrame:
    """
    Lê um arquivo CSV e retorna um DataFrame.

    Args:
        f (str): Caminho do arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame contendo os dados do arquivo CSV.

    Raises:
        TypeError: Se o formato do arquivo não for suportado.
    """
    if f.endswith('.csv'):
        return pd.read_csv(f, sep=',', doublequote=False, engine='python', on_bad_lines='skip')
    raise TypeError("File format is not supported")


def clean_url(text: str) -> str:
    """
    Remove URLs de um texto.

    Args:
        text (str): Texto a ser limpo.

    Returns:
        str: Texto limpo sem URLs.
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)


def process_files(input_dir: str, output_dir: str):
    """
    Processa todos os arquivos no diretório de entrada, aplicando pré-processamento e limpeza.

    Args:
        input_dir (str): Diretório de entrada contendo os arquivos brutos.
        output_dir (str): Diretório de saída para os arquivos processados.
    """
    raw_inputs = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    for file in raw_inputs:
        raw_file = os.path.join(input_dir, file)
        new_file_path = os.path.join(output_dir, file)
        preprocess_input(in_filename=raw_file, out_filename=new_file_path)
        posts_df = read_twitter_posts_file(new_file_path)
        posts_df['texto_tweet'] = posts_df['texto_tweet'].apply(clean_url)
        if 'sentimento_tweet' in posts_df.columns:
            posts_df = posts_df[~posts_df['sentimento_tweet'].isna()]
        posts_df.to_csv(new_file_path, index=False)


if __name__ == "__main__":
    process_files(input_dir="data/raw_input", output_dir="data/clean_data")