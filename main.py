from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from task1_rag_pipeline import main as run_task1
from task2_advanced_retrieval import main as run_task2
from task3_citation import main as run_task3
from task4_ragas import main as run_task4


app = FastAPI(
    title="RAG Full Project API",
    description="Week 7 RAG Assessment",
    version="1.0.0"
)


class TaskRequest(BaseModel):
    task: int


@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/run/task1")
def task1():
    try:
        run_task1()

        return {
            "status": "success",
            "task": "task1",
            "output": "output/task1/"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/run/task2")
def task2():
    try:
        run_task2()

        return {
            "status": "success",
            "task": "task2",
            "output": "output/task2/"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/run/task3")
def task3():
    try:
        run_task3()

        return {
            "status": "success",
            "task": "task3",
            "output": "output/task3/"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/run/task4")
def task4():
    try:
        run_task4()

        return {
            "status": "success",
            "task": "task4",
            "output": "output/task4/"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/run/all")
def run_all():

    try:

        run_task1()
        run_task2()
        run_task3()
        run_task4()

        return {
            "status": "success",
            "message": "All tasks completed",
            "outputs": [
                "output/task1/",
                "output/task2/",
                "output/task3/",
                "output/task4/"
            ]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )