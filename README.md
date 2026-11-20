## Project Overview


This project is a Policy Q&A Assistant using **Retrieval-Augmented Generation (RAG)**.


The system retrieves information from company policy documents and generates answers only from the retrieved context.


It also includes fallback handling, advanced retrieval, citations, hallucination control, and RAGAS evaluation.


## Tasks


### Task 1 - RAG Pipeline


Load at least 8 documents
Parse and chunk the documents
Create embeddings
Store the chunks with metadata and stable IDs
Implement question answering
Add a fallback when relevant information is not found
Implement context token budget
Test answerable and unanswerable questions
### Task 2 - Advanced Retrieval


Implement BM25 retrieval
Implement vector retrieval
Compare BM25 and vector search using Hit@3
Implement Reciprocal Rank Fusion (RRF)
Implement hybrid search
Add cross-encoder reranking
Implement query rewriting or HyDE
Add metadata filtering
### Task 3 - Citations and Hallucination Control


Add source and page information to the context
Generate answers with citations
Validate citations
Compare different prompt versions
Check hallucinated answers
Check faithfulness
Provide an /ask API endpoint
### Task 4 - Golden Dataset and RAGAS


Create golden.jsonl
Add 15 questions:
  - 8 simple questions
  - 4 multi-chunk questions
  - 3 unanswerable questions
Include answer, source, and page
Run RAGAS evaluation
Check faithfulness
Check answer relevancy
Check context precision
Check context recall
Analyze the 3 worst questions
Apply one improvement
Compare before and after results
## Project Structure
text
week7_rag_assessment/
│
├── data/
│   ├── documents/
│   └── golden.jsonl
│
├── output/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
│
├── models/
│
├── task1.py
├── task2.py
├── task3.py
├── task4.py
├── main.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md


## How to Run


Create and activate the virtual environment:
bash
python -m venv venv


Windows:
bash
venv\Scripts\activate


Install the required packages:
bash
pip install -r requirements.txt


Run Task 1:
bash
python task1.py


Run Task 2:
bash
python task2.py


Run Task 3:
bash
python task3.py


Run Task 4:
bash
python task4.py


Run the API:
bash
python main.py


## Environment Variables


Create a .env file and keep API keys inside it.


Example:
env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=your_model_here


Do not upload .env or API keys to GitHub.


## Expected Result


The system should answer questions using the available policy documents.


For questions that cannot be answered from the documents, it should respond:
text
I don't have information on that.


The system should also provide valid citations and reduce hallucinated answers.


## Git


Commit the project regularly with clear commit messages.


Example:
bash
git add .
git commit -m "completed task 1"
git push


Do not commit:
text
.env
API keys
passwords


## Conclusion


This project demonstrates a complete RAG-based Policy Q&A Assistant with advanced retrieval, citation validation, hallucination control, and RAGAS evaluation.