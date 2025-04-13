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


Okay, this is excellent detail about your experimental process and findings. Here's how you can integrate this information into the README, likely as a new section detailing the implementation choices and rationale.

Add this section, perhaps after the ## 📁 Repository Structure section:

Markdown

## 🛠️ Implementation Details & Experiments

This section details key technical decisions and findings from experiments conducted during the development of MedicalBot's RAG pipeline, primarily documented in `RAG_retrieval_analysis.ipynb`.

### 1. Embedding Model Selection

Generating high-quality embeddings for both user queries and document chunks is critical for retrieval accuracy. Several models were evaluated, with the top performers being:

* `abhinand/MedEmbed-small-v0.1`: Part of the MedEmbed family, fine-tuned specifically on medical/clinical data. Its base model is `BGE-small-en-v1.5`.
* `all-MiniLM-L6-v2`: A popular, efficient sentence transformer model.

**Findings:**
* Both models demonstrated strong performance in understanding query/context similarity.
* `abhinand/MedEmbed-small-v0.1` showed a slight edge, likely due to its domain-specific fine-tuning, enhancing understanding of medical terminology.
* While `all-MiniLM-L6-v2` is generally faster, `MedEmbed-small` (based on `BGE-small-en-v1.5`) provides a good balance of accuracy and performance.

**Decision:** The **`abhinand/MedEmbed-small-v0.1`** model was selected for the final MedicalBot implementation due to its superior performance attributed to its medical domain knowledge.

### 2. Chunking Strategy

The method of splitting the source documents into smaller chunks significantly impacts retrieval quality and downstream processing. Experiments were conducted with various chunk sizes and overlaps, including:

* Chunk Size: 500 chars, Overlap: 150 chars
* Chunk Size: 300 chars, Overlap: 100 chars
* Chunk Size: 1000 chars, Overlap: 300 chars

**Decision:** A **chunk size of 500 characters with an overlap of 150 characters** was chosen.

**Rationale:**
* **Resource Constraints:** Smaller chunks are less demanding on memory and compute resources during embedding and retrieval.
* **Information Density:** Source answers often contained extraneous information; smaller chunks help isolate relevant facts.
* **LLM Context Limits:** Smaller retrieved chunks allow more top-K results, system prompts, and user queries to fit within the limited context windows (e.g., 1024-2048 tokens) of many LLMs.

### 3. Re-Ranking Implementation

Initial retrieval based solely on vector similarity can be improved by re-ranking the candidate chunks before passing them to the LLM.

**Findings:**
* Experiments showed that implementing a re-ranking step **significantly improved** retrieval performance compared to using raw similarity scores alone.

**Decision:** A custom, multi-faceted re-ranking strategy was implemented.

**Logic:** The final score for ranking chunks is a weighted combination of the following similarity scores:
1.  Similarity between the **user query** and the **question** associated with the retrieved chunk (metadata).
2.  Similarity between the **user query** and the **questions** retrieved from a separate questions-only index (providing broader question context).
3.  Similarity between the **user query** and the **content of the retrieved chunk** itself.

**Rationale:** This approach explicitly leverages the valuable metadata (the original questions associated with answers/chunks) and considers the interplay between the query, chunk content, and related questions, leading to more contextually relevant results for this specific Q&A dataset.

### 4. Vector Index Choice

The vector index stores the embeddings for fast retrieval.

**Decision:** **FAISS (Facebook AI Similarity Search)** was chosen as the vector index.

**Rationale:**
* **Performance:** FAISS is highly optimized for speed and efficiency in similarity searching.
* **Ease of Use:** It is relatively easy to integrate, operates in-memory, and allows straightforward saving/loading of the index, making it ideal for rapid prototyping, POCs, and MVPs.
* **Scalability:** While FAISS is suitable for this stage, alternative vector databases like Milvus or Pinecone are noted as potential options for future scaling if required (offering managed hosting, advanced features, etc.).
