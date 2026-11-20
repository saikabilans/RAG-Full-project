import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "task2"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
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


def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def bm25_search(question, chunks, bm25, k=10):
    scores = bm25.get_scores(
        tokenize(question)
    )

    indexes = np.argsort(scores)[::-1][:k]

    return [
        {
            **chunks[index],
            "score": float(scores[index])
        }
        for index in indexes
    ]


def vector_search(question, chunks, embeddings, k=10):
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


def reciprocal_rank_fusion(
    bm25_results,
    vector_results
):
    scores = {}
    items = {}

    for rank, item in enumerate(
        bm25_results,
        start=1
    ):
        item_id = item["id"]

        scores[item_id] = (
            scores.get(item_id, 0)
            + 1 / (60 + rank)
        )

        items[item_id] = item

    for rank, item in enumerate(
        vector_results,
        start=1
    ):
        item_id = item["id"]

        scores[item_id] = (
            scores.get(item_id, 0)
            + 1 / (60 + rank)
        )

        items[item_id] = item

    ordered = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        {
            **items[item_id],
            "rrf_score": scores[item_id]
        }
        for item_id in ordered
    ]


def main():

    chunks, embeddings = load_index()

    bm25 = BM25Okapi(
        [
            tokenize(chunk["text"])
            for chunk in chunks
        ]
    )

    questions = [
        "How many casual leaves are available?",
        "Where should leave requests be submitted?",
        "What are the normal working hours?",
        "Who approves work from home?",
        "When is work considered overtime?",
        "What is needed for travel expenses?",
        "Can personal expenses be reimbursed?",
        "How should payroll problems be reported?"
    ]

    results = []

    for question in questions:
        print("\nQuestion:", question)

        bm25_results = bm25_search(
            question,
            chunks,
            bm25
        )

        vector_results = vector_search(
            question,
            chunks,
            embeddings
        )

        hybrid_results = reciprocal_rank_fusion(
            bm25_results,
            vector_results
        )

        try:
            reranker = CrossEncoder(
                RERANK_MODEL
            )

            pairs = [
                (
                    question,
                    item["text"]
                )
                for item in hybrid_results[:10]
            ]

            scores = reranker.predict(pairs)

            reranked = []

            for item, score in zip(
                hybrid_results[:10],
                scores
            ):
                reranked.append({
                    **item,
                    "rerank_score": float(score)
                })

            reranked.sort(
                key=lambda x: x["rerank_score"],
                reverse=True
            )

        except Exception as error:
            print(
                "Reranker error:",
                error
            )

            reranked = hybrid_results[:5]

        print(
            "BM25:",
            bm25_results[0]["source"]
        )

        print(
            "Vector:",
            vector_results[0]["source"]
        )

        print(
            "Hybrid:",
            hybrid_results[0]["source"]
        )

        print(
            "Reranked:",
            reranked[0]["source"]
        )

        results.append({
            "question": question,
            "bm25_top": bm25_results[0]["source"],
            "vector_top": vector_results[0]["source"],
            "hybrid_top": hybrid_results[0]["source"],
            "reranked_top": reranked[0]["source"]
        })

    with open(
        OUTPUT_DIR / "task2_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2
        )

    report = """
TASK 2 REPORT
=============

Methods:
1. BM25
2. Dense vector retrieval
3. Reciprocal Rank Fusion
4. Cross-encoder reranking

Output:
output/task2/task2_results.json
"""

    (REPORT_DIR / "task2_report.txt").write_text(
        report,
        encoding="utf-8"
    )
    print("Output: output/task2/task2_results.json")
    print("Report: reports/task2_report.txt")


if __name__ == "__main__":
    main()