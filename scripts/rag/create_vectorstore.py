from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import pandas as pd

def create_vectorstore(csv_path: str) -> FAISS:
    """
    Cria um vectorstore a partir de um arquivo CSV contendo tweets.

    Args:
        csv_path (str): Caminho para o arquivo CSV.

    Returns:
        FAISS: Loja de vetores construída.
    """
    df = pd.read_csv(csv_path)
    data = [
        Document(page_content=doc["texto_tweet"],
                 metadata={"language": doc["idioma_tweet"], "sentiment": doc["predicted_sentiment"]})
        for _, doc in df.iterrows()
    ]
    return FAISS.from_documents(data, OpenAIEmbeddings(model="text-embedding-3-small"))

def save_vectorstore(vectorstore: FAISS, save_path: str):
    """
    Salva a vectorstore localmente.

    Args:
        vectorstore (FAISS): Loja de vetores a ser salva.
        save_path (str): Caminho para salvar a loja de vetores.
    """
    vectorstore.save_local(save_path)

def main():
    load_dotenv()
    vectorstore = create_vectorstore('data/tweets_with_sentiment.csv')
    save_vectorstore(vectorstore, ".vectorstore")

if __name__ == "__main__":
    main()
