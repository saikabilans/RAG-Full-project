import os
import json
import pickle
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "policies"
OUTPUT_DIR = BASE_DIR / "output" / "task1"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def load_documents():
    documents = []

    for file in sorted(DATA_DIR.glob("*.txt")):
        documents.append({
            "source": file.name,
            "page": 1,
            "text": file.read_text(encoding="utf-8")
        })

    return documents


def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_chunks():
    chunks = []

    for document in load_documents():
        parts = chunk_text(document["text"])

        for index, part in enumerate(parts):
            chunks.append({
                "id": f"{document['source']}_{index}",
                "source": document["source"],
                "page": document["page"],
                "text": part
            })

    return chunks


def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(embeddings)


def save_index(chunks, embeddings):
    np.save(
        MODEL_DIR / "embeddings.npy",
        embeddings
    )

    with open(
        MODEL_DIR / "chunks.pkl",
        "wb"
    ) as file:
        pickle.dump(chunks, file)


def retrieve(question, chunks, embeddings, k=5):
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ question_embedding
    indexes = np.argsort(scores)[::-1][:k]

    return [
        {
            **chunks[index],
            "score": float(scores[index])
        }
        for index in indexes
    ]


def count_tokens(text):
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def build_context(results, max_tokens=350):
    selected = []
    total_tokens = 0

    for result in results:
        tokens = count_tokens(result["text"])

        if total_tokens + tokens > max_tokens:
            break

        selected.append(result)
        total_tokens += tokens

    return selected


def ask_llm(question, context):
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    if not api_key:
        return "OPENROUTER_API_KEY is missing in .env"

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    prompt = f"""
Answer ONLY using the context.

If the answer is not available in the context,
reply exactly:

I don't have information on that.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def main():

    chunks = create_chunks()
    embeddings = create_embeddings(chunks)

    save_index(chunks, embeddings)

    print(f"Documents: {len(load_documents())}")
    print(f"Chunks: {len(chunks)}")
    print("Embeddings saved to models/")

    questions = [
        "How many casual leaves are available per year?",
        "Where should planned leave requests be submitted?",
        "What are the standard working hours?",
        "Who must approve work from home?",
        "What is overtime?",
        "What is required for travel expense claims?",
        "Who won the football World Cup?",
        "What is today's weather in Chennai?",
        "What is the current price of gold?",
        "What is the price of Bitcoin today?"
    ]

    results = []

    for k in [2, 5, 10]:
        print(f"\n========== k={k} ==========")

        for question in questions:
            retrieved = retrieve(
                question,
                chunks,
                embeddings,
                k
            )

            top_score = retrieved[0]["score"]

            print("\nQuestion:", question)
            print("Score:", round(top_score, 4))

            if top_score < 0.35:
                answer = "I don't have information on that."
                llm_called = False
            else:
                selected = build_context(retrieved)
                context = "\n\n".join(
                    f"[Source: {item['source']}, Page {item['page']}]\n"
                    f"{item['text']}"
                    for item in selected
                )

                answer = ask_llm(
                    question,
                    context
                )
                llm_called = True

            print("Answer:", answer)

            results.append({
                "k": k,
                "question": question,
                "answer": answer,
                "score": top_score,
                "llm_called": llm_called
            })

    with open(
        OUTPUT_DIR / "task1_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    report = f"""
TASK 1 REPORT
=============

Documents: {len(load_documents())}
Chunks: {len(chunks)}
Embedding model: {EMBEDDING_MODEL}
Tested k values: 2, 5, 10
Fallback threshold: 0.35
Context budget: 350 tokens

Output:
output/task1/task1_results.json

Models:
models/embeddings.npy
models/chunks.pkl
"""

    (REPORT_DIR / "task1_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    print("\nTASK 1 COMPLETED")
    print("Output: output/task1/task1_results.json")
    print("Report: reports/task1_report.txt")


if __name__ == "__main__":
    main()