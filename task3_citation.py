import os
import json
import pickle
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "task3"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_index():
    embeddings = np.load(
        MODEL_DIR / "embeddings.npy"
    )

    with open(
        MODEL_DIR / "chunks.pkl",
        "rb"
    ) as file:
        chunks = pickle.load(file)

    return chunks, embeddings


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


def ask_llm(question, results):
    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    model = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    if not api_key:
        return "OPENROUTER_API_KEY is missing in .env"

    context = "\n\n".join(
        f"[Source: {item['source']}, Page {item['page']}]\n"
        f"{item['text']}"
        for item in results
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    prompt = f"""
You are a company policy assistant.

Answer ONLY using the context.

Every factual statement must include a citation:

[Source: filename.txt, Page 1]

If the answer is not supported by the context,
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


def validate_citations(answer, results):
    pattern = (
        r"\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]"
    )

    citations = re.findall(
        pattern,
        answer
    )

    valid = {
        (
            item["source"],
            str(item["page"])
        )
        for item in results
    }

    invalid = [
        citation
        for citation in citations
        if citation not in valid
    ]

    return citations, invalid


def main():
    chunks, embeddings = load_index()

    questions = [
        "How many casual leaves are available per year?",
        "What are the standard working hours?",
        "Who must approve work from home?",
        "What is overtime?",
        "What is required for travel expense claims?",
        "Can employees claim personal expenses?",
        "Who won the football World Cup?",
        "What is today's weather in Chennai?"
    ]

    results = []

    for question in questions:
        print("\nQuestion:", question)

        retrieved = retrieve(
            question,
            chunks,
            embeddings
        )

        if (
            not retrieved
            or retrieved[0]["score"] < 0.35
        ):
            answer = (
                "I don't have information on that."
            )
        else:
            answer = ask_llm(
                question,
                retrieved
            )

        citations, invalid = validate_citations(
            answer,
            retrieved
        )

        print("Answer:", answer)
        print("Citations:", citations)

        if invalid:
            print(
                "Invalid citations:",
                invalid
            )
        else:
            print("Citation validation: PASS")

        results.append({
            "question": question,
            "answer": answer,
            "citations": citations,
            "invalid_citations": invalid
        })

    with open(
        OUTPUT_DIR / "task3_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    report = """
TASK 3 REPORT
=============

Citation format:
[Source: filename.txt, Page 1]

Citation validation was performed
against retrieved source chunks.

Output:
output/task3/task3_results.json
"""

    (REPORT_DIR / "task3_report.txt").write_text(
        report,
        encoding="utf-8"
    )
    print("Output: output/task3/task3_results.json")
    print("Report: reports/task3_report.txt")


if __name__ == "__main__":
    main()