import json
import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

GOLDEN_FILE = BASE_DIR / "data" / "golden.jsonl"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output" / "task4"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_golden():
    data = []

    with open(
        GOLDEN_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            if line.strip():
                data.append(
                    json.loads(line)
                )

    return data


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
Answer ONLY using the context.

If the answer is not available
in the context, reply exactly:

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
    golden = load_golden()

    print(
        f"Golden questions: {len(golden)}"
    )

    chunks, embeddings = load_index()

    results = []
    retrieval_success = 0

    for number, item in enumerate(
        golden,
        start=1
    ):
        question = item["question"]

        print(
            f"\n---------- Question {number} ----------"
        )

        retrieved = retrieve(
            question,
            chunks,
            embeddings
        )

        top_score = (
            retrieved[0]["score"]
            if retrieved
            else 0
        )

        if top_score < 0.35:
            answer = (
                "I don't have information on that."
            )
        else:
            answer = ask_llm(
                question,
                retrieved
            )

        expected_source = item.get(
            "source",
            ""
        )

        retrieved_sources = [
            result["source"]
            for result in retrieved
        ]

        if expected_source:
            success = (
                expected_source
                in retrieved_sources
            )
        else:
            success = (
                top_score < 0.35
            )

        if success:
            retrieval_success += 1

        print("Question:", question)
        print(
            "Reference:",
            item["answer"]
        )
        print(
            "Generated:",
            answer
        )
        print(
            "Top score:",
            round(top_score, 4)
        )

        results.append({
            "question": question,
            "reference_answer": item["answer"],
            "generated_answer": answer,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "retrieval_success": success
        })

    accuracy = (
        retrieval_success / len(golden)
        if golden
        else 0
    )

    with open(
        OUTPUT_DIR / "task4_results.json",
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
TASK 4 REPORT
=============

Golden dataset:
data/golden.jsonl

Total questions:
{len(golden)}

Retrieval successful:
{retrieval_success}

Retrieval accuracy:
{accuracy:.2%}

Output:
output/task4/task4_results.json
"""

    (REPORT_DIR / "task4_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    print(
        f"Retrieval accuracy: {accuracy:.2%}"
    )
    print(
        "Output: output/task4/task4_results.json"
    )
    print(
        "Report: reports/task4_report.txt"
    )


if __name__ == "__main__":
    main()
