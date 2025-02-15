from transformers import pipeline
from utils import analyze_sentiment, read_clean_data


def load_sentiment_model():
    model_name = 'models/bertweet-base-sentiment-analysis'
    model = pipeline(task='sentiment-analysis', model=model_name)
    return model

def generate_sentiment_tweets_file():
    model = load_sentiment_model()
    df = read_clean_data()
    y_pred = analyze_sentiment(model, df['texto_tweet'].tolist())
    df['predicted_sentiment'] = y_pred
    df['predicted_sentiment'] = df['predicted_sentiment'].mask(
        df['predicted_sentiment'] == 0, 'Negative').mask(
        df['predicted_sentiment'] == 1, 'Neutral').mask(
        df['predicted_sentiment'] == 2, 'Positive')
    df.to_csv('data/tweets_with_sentiment.csv')


if __name__ == "__main__":
    generate_sentiment_tweets_file()