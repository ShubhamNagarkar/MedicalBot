# ⚕️ MedicalBot: Medical Question Answering System

This repository contains the development of **MedicalBot**, a question-answering system designed to respond to user queries about medical diseases using information from a medical dataset.

## 🎯 Problem Statement

The primary goal is to develop an effective medical question-answering system. Utilizing a provided medical dataset (initially based on MedQuad, potentially augmented), the system aims to accurately understand and answer user questions primarily focused on various medical diseases, conditions, and related concepts. The challenge lies in interpreting natural language queries and retrieving/generating factually correct and relevant information from the source data.

## ✨ Approach: Retrieval-Augmented Generation (RAG)

This project implements a **Retrieval-Augmented Generation (RAG)** approach. This combines the strengths of:

1.  **Retrieval:** Efficiently searching and retrieving relevant passages from the medical dataset based on the user's query.
2.  **Generation:** Utilizing a Large Language Model (LLM) to synthesize a coherent and contextually appropriate answer based on the retrieved information.

This approach helps to ground the LLM's responses in factual data from the knowledge base, reducing hallucinations and improving the accuracy of the answers.

## 🛠️ Key Components & Workflow

1.  **Data Preprocessing (`data-analysis.ipynb`):** The initial dataset (`medical_dataset.csv`) is loaded, explored, and cleaned. This involves structuring the Q&A pairs appropriately, handling inconsistencies, and saving the processed data (`clean_questions_to_answers_dataset_v1.json`).
2.  **RAG Experiment (`RAG_retrieval_analysis.ipynb`):** This notebook contains benchmarking experiments concerning Chunking, Indexing, Retrieval logic, Ranking and Retrieval Testing.
3.  **Inference and Evaluation (`RAGMedicalBot.ipynb`):** The system's performance is evaluated using appropriate metrics for generation (e.g., ROUGE, BERTScore).
4.  **Utilities (`utils.py`):** Helper functions support various tasks throughout the notebooks.


