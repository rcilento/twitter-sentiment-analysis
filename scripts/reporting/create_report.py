import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Carrega e limpa os dados do arquivo CSV.

    Args:
        file_path (str): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame contendo os dados limpos.
    """
    nltk.download('stopwords')
    stop = set(stopwords.words('english')).union({'air', 'france', 'airfrance', '@airfrance', '#airfrance'})

    data = pd.read_csv(file_path, index_col=1)
    data = data[~data['texto_tweet'].isna()]
    data['texto_tweet'] = data['texto_tweet'].apply(
        lambda x: ' '.join([word for word in str(x).split() if str(word) not in stop]))

    return data


def generate_sentiment_bar_chart(data: pd.DataFrame, output_path: str):
    """
    Gera gráfico de barras da distribuição de sentimentos.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados.
        output_path (str): Caminho para salvar o gráfico.
    """
    sentiment_counts = data['predicted_sentiment'].value_counts()
    plt.figure(figsize=(10, 5))
    plt.bar(sentiment_counts.index, sentiment_counts.values)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(output_path)
    plt.close()


def generate_wordcloud(data: pd.DataFrame, output_path: str):
    """
    Gera uma nuvem de palavras a partir dos tweets negativos.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados.
        output_path (str): Caminho para salvar a nuvem de palavras.
    """
    text_data = ' '.join(data[data['predicted_sentiment'] == 'Negative']['texto_tweet'].dropna())
    wordcloud = WordCloud(width=800, height=400, max_words=50).generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()


def get_top_words(data: pd.DataFrame) -> pd.Series:
    """
    Obtém as 20 palavras mais frequentes nos tweets negativos.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados.

    Returns:
        pd.Series: Série contendo as palavras mais frequentes.
    """
    text_data = ' '.join(data[data['predicted_sentiment'] == 'Negative']['texto_tweet'].dropna())
    return pd.Series(text_data.split()).value_counts().head(20)


def load_questions(file_path: str) -> list:
    """
    Carrega questões de um arquivo.

    Args:
        file_path (str): Caminho para o arquivo de questões.

    Returns:
        list: Lista de questões.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip().split('\n\n')


def create_report(data_file: str, questions_file: str):
    """
    Cria um relatório com gráficos e insights utilizando Streamlit.

    Args:
        data_file (str): Caminho para o arquivo de dados.
        questions_file (str): Caminho para o arquivo de questões.
    """
    data = load_and_clean_data(data_file)
    row_count = len(data)

    generate_sentiment_bar_chart(data, 'outputs/images/sentiment_bar_chart.png')
    generate_wordcloud(data, 'outputs/images/wordcloud.png')
    top_words = get_top_words(data)

    questions = []
    for i in range(1, 3):
        with open(f'outputs/rag_answers/answer-{i}.txt', 'r') as file:
            questions.append(file.read())

    st.title('Customers Sentiment Report')
    st.write(f'Total Analysed Tweets: {row_count}')

    st.subheader('Sentiment Distribution')
    st.image('outputs/images/sentiment_bar_chart.png')

    st.subheader('Negative Tweets Wordcloud')
    st.image('outputs/images/wordcloud.png')

    st.subheader('Top 20 Most Frequent Words in Negative Tweets')
    st.table(top_words)

    st.subheader('AI QA - Extracting Insights')
    for question in questions:
        st.subheader('Question :thinking_face:', divider='blue')
        st.markdown(question)
        st.text('\n')


if __name__ == "__main__":
    create_report('data/tweets_with_sentiment.csv', 'prompts/tweets_insights_questions.txt')
