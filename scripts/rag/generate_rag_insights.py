from datetime import datetime
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from typing import List, Dict
import json

def load_prompt(prompt_file: str) -> str:
    """
    Carrega um prompt de um arquivo.

    Args:
        prompt_file (str): Nome do arquivo de prompt.

    Returns:
        str: Conteúdo do prompt.
    """
    with open("prompts/" + prompt_file, "r") as file:
        return file.read()

def load_questions(file_path: str) -> List[str]:
    """
    Carrega questões de um arquivo.

    Args:
        file_path (str): Caminho para o arquivo de questões.

    Returns:
        List[str]: Lista de questões.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip().split('\n\n')

def write_file(file_path: str, content: str):
    """
    Escreve conteúdo em um arquivo.

    Args:
        file_path (str): Caminho para o arquivo.
        content (str): Conteúdo a ser escrito.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def generate_rag(vectorstore: FAISS, prompt_template: str) -> any:
    """
    Gera uma cadeia de recuperação aumentada por geração (RAG) usando a loja de vetores e um template de prompt.

    Args:
        vectorstore (FAISS): Loja de vetores.
        prompt_template (str): Template de prompt.

    Returns:
        any: Cadeia RAG gerada.
    """
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50})
    llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'input'])
    documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, documents_chain)

def evaluate_rag_response(question: str, answer: str, context: str) -> Dict[str, str]:
    """
    Avalia a resposta da RAG utilizando prompts de avaliação.

    Args:
        question (str): Questão feita.
        answer (str): Resposta gerada.
        context (str): Contexto da questão.

    Returns:
        Dict[str, str]: Métricas de avaliação.
    """
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    context_relevance_prompt = load_prompt("context_relevance_prompt.txt").format(context=context, question=question)
    groundness_prompt = load_prompt("groundness_prompt.txt").format(context=context, answer=answer)
    answer_relevance_prompt = load_prompt("answer_relevance_prompt.txt").format(answer=answer, question=question)
    return {
        'context_relevance': llm.predict(context_relevance_prompt),
        'groundness': llm.predict(groundness_prompt),
        'answer_relevance': llm.predict(answer_relevance_prompt)
    }

def is_metrics_above_threshold(metrics: Dict[str, str], threshold: float) -> bool:
    """
    Verifica se as métricas estão acima de um determinado limite.

    Args:
        metrics (Dict[str, str]): Métricas de avaliação.
        threshold (float): Limite de comparação.

    Returns:
        bool: Verdadeiro se todas as métricas estiverem acima do limite, falso caso contrário.
    """
    for value in metrics.values():
        if not value.isdecimal() or threshold > int(value):
            return False
    return True

def load_vectorstore() -> FAISS:
    """
    Carrega a loja de vetores localmente.

    Returns:
        FAISS: Loja de vetores carregada.
    """
    return FAISS.load_local(r".vectorstore", embeddings=OpenAIEmbeddings(model="text-embedding-3-small"), allow_dangerous_deserialization=True)

def log_to_json(timestamp: str, question: str, full_prompt: str, context: str, answer: str, log_file: str,
                metrics: Dict):
    """
    Registra uma entrada de log em formato JSON em um arquivo especificado.

    Args:
        timestamp (str): Timestamp da entrada de log.
        question (str): Pergunta feita.
        full_prompt (str): Prompt completo utilizado.
        context (str): Contexto da pergunta.
        answer (str): Resposta gerada.
        log_file (str): Caminho para o arquivo de log.
        metrics (Dict): Métricas avaliadas para resposta.
    """
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "full_prompt": full_prompt,
        "context": context,
        "answer": answer,
        "metrics": metrics
    }

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            log_data = json.load(file)
    except FileNotFoundError:
        log_data = []

    log_data.append(log_entry)

    with open(log_file, 'w', encoding='utf-8') as file:
        json.dump(log_data, file, ensure_ascii=False, indent=4)

def main():
    load_dotenv()
    vectorstore = load_vectorstore()
    prompt_template = load_prompt('tweets_insights_prompt.txt')
    queries = load_questions('prompts/tweets_insights_questions.txt')

    rag = generate_rag(vectorstore, prompt_template)

    for i, query in enumerate(queries):
        timestamp = datetime.now().isoformat()
        response = rag.invoke({"input": query})
        answer = response["answer"]
        context = str(response.get("context", ""))
        full_prompt = prompt_template.format(context=context, input=query)
        eval_metrics = evaluate_rag_response(question=query, answer=answer, context=context)
        if not is_metrics_above_threshold(eval_metrics, 0.6):
            answer = "Sorry, no answer was found to this question"
        print(query)
        print(answer)
        print(eval_metrics)
        log_to_json(log_file='outputs/rag_answers/rag_answers.json', timestamp=timestamp, question=query, full_prompt=full_prompt, context=context,
                    answer=answer, metrics=eval_metrics)
        write_file(f'outputs/rag_answers/answer-{i+1}.txt', query + "\n\n\n" + answer)

if __name__ == "__main__":
    main()
