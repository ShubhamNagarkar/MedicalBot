# ⚕️ MedicalBot: Medical Question Answering System

This repository contains the development of **MedicalBot**, a question-answering system designed to respond to user queries about medical diseases using information from a medical dataset.

## 🎯 Problem Statement

The primary goal is to develop an effective medical question-answering system. Utilizing a provided medical dataset (initially based on MedQuad, potentially augmented), the system aims to accurately understand and answer user questions primarily focused on various medical diseases, conditions, and related concepts. The challenge lies in interpreting natural language queries and retrieving/generating factually correct and relevant information from the source data.

## ✨ Approach: Retrieval-Augmented Generation (RAG)

This project implements a **Retrieval-Augmented Generation (RAG)** approach. This combines the strengths of:

1.  **Retrieval:** Efficiently searching and retrieving relevant passages from the medical dataset based on the user's query.
2.  **Generation:** Utilizing a Large Language Model (LLM) to synthesize a coherent and contextually appropriate answer based on the retrieved information.

This approach helps to ground the LLM's responses in factual data from the knowledge base, reducing hallucinations and improving the accuracy of the answers.

## 📁 Repository Structure:

1.  **Data Preprocessing (`data-analysis.ipynb`):** The initial dataset (`medical_dataset.csv`) is loaded, explored, and cleaned. This involves structuring the Q&A pairs appropriately, handling inconsistencies, and saving the processed data (`clean_questions_to_answers_dataset_v1.json`).
2.  **RAG Experiment (`RAG_retrieval_analysis.ipynb`):** This notebook contains benchmarking experiments concerning Chunking, Indexing, Retrieval logic, Ranking and Retrieval Testing.
3.  **Inference and Evaluation (`RAGMedicalBot.ipynb`):** The system's performance is evaluated using appropriate metrics for generation (e.g., ROUGE, BERTScore).
4.  **Utilities (`utils.py`):** Helper functions support various tasks throughout the notebooks.


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

## 📊 Evaluation

The performance of the MedicalBot RAG system was evaluated using both automated metrics (ROUGE and BERTScore) and manual inspection of the generated answers. Detailed evaluation procedures and results can be found in the `RAGMedicalBot.ipynb` notebook.

### ROUGE Metrics (Lexical Overlap)

ROUGE scores measure the overlap between the generated answers and the reference answers from the dataset based on n-grams and sequence alignment.

* **`ROUGE-1` (Unigram Overlap): 0.41** - Indicates that approximately 41% of the individual words (unigrams) in the reference answers were also present in the generated answers.
* **`ROUGE-2` (Bigram Overlap): 0.28** - Shows a lower overlap for word pairs (bigrams), suggesting that while individual terms might be present, exact phrasing is less common.
* **`ROUGE-L` (Longest Common Subsequence): 0.325** - Implies moderate structural alignment; the system often reorders or paraphrases content rather than replicating exact sequences from the source.

**ROUGE Conclusion:** The moderate ROUGE scores suggest that the model does not simply copy text verbatim. It retains key terms (reflected in ROUGE-1) but often paraphrases or restructures sentences (lower ROUGE-2 and ROUGE-L). Manual checks confirmed the model was well-grounded by the retrieved passages with minimal hallucination. The degree of paraphrasing could potentially be tuned via LLM temperature settings in future refinements.

### BERTScore Metrics (Semantic Similarity)

BERTScore compares the semantic similarity between generated and reference answers using contextual embeddings, providing insight beyond exact word matches.

* **`Precision`: 0.859** - Indicates that most of the information in the generated answers is semantically relevant to the reference answers, suggesting low hallucination of irrelevant content.
* **`Recall`: 0.873** - Shows that the generated answers capture a large portion of the relevant semantic content present in the reference answers, indicating completeness.
* **`F1-Score`: 0.866** - The high F1 score demonstrates strong overall semantic similarity; the generated answers are semantically very close to the reference answers, even when the wording differs significantly.

**BERTScore Conclusion:** The high BERTScore results strongly suggest that the LLM effectively understands and conveys the correct medical facts derived from the provided context passages. High recall points to informative answers, while high precision indicates relevance and focus.

### Overall Conclusion

Combining the automated metrics with manual inspection, it's evident that the RAG system performs well for the Medical Q&A task.
* The system produces **factually correct and relevant answers**, drawing accurately from the retrieved context.
* It tends to **paraphrase or reformat information** rather than copying exact sentences, which is desirable behavior for a helpful assistant.
* While lexical similarity (ROUGE) is moderate due to this paraphrasing, **semantic similarity (BERTScore) is high**, confirming that the core meaning and essential medical information are accurately conveyed.

This indicates the chosen RAG approach successfully leverages the provided dataset to generate trustworthy and understandable answers to medical queries.

## 🔮 Future Scope

While the current MedicalBot implementation provides a solid baseline, especially considering potential memory and single-GPU constraints, there are numerous avenues for enhancing each component of the RAG architecture:

### Embedding Models

* **Domain-Specific Models:** Explore larger embedding models specifically pre-trained or fine-tuned on extensive medical and clinical corpora. These could offer more nuanced understanding of medical terminologies compared to general-purpose models.
* **Increased Dimensionality:** Experiment with models offering higher embedding dimensions (beyond the 384 used) which might capture more semantic detail, potentially at the cost of increased computational requirements.

### RAG Pipeline Enhancements

* **LLM-Powered Data Filtering:** Implement a pre-processing step where an LLM filters the source dataset to remove answers lacking substantial contextual information relative to their questions. This could lead to a cleaner, more relevant vector index.
* **Topic Modeling / Intent Classification:**
    * Train a classifier to categorize user queries into predefined medical topics (e.g., Glaucoma, Cancer, Diabetes).
    * Use predicted topics to filter retrieval results (semantic routing) or as features in the re-ranking step.
    * Leverage topic classification to detect Out-of-Domain (OOD) queries.
* **Query Rewriting/Expansion:**
    * Utilize LLMs to rewrite or expand short/ambiguous user queries, potentially adding relevant medical context or synonyms.
    * Augment queries with conversation history for contextually relevant follow-up answers.
* **Advanced Indexing & Retrieval:**
    * Explore managed or more scalable vector databases (e.g., `Milvus`, `Pinecone`, `Weaviate`) for larger datasets and production environments.
    * Implement **Contextual RAG:** Use an LLM to generate concise summaries or contextual descriptions for each chunk and prepend this context before indexing to potentially improve retrieval relevance.
    * Adopt **Hybrid Search:** Combine dense vector retrieval (like cosine similarity) with sparse retrieval methods (like `BM25`) to leverage both semantic meaning and keyword matching.
* **Advanced Re-Ranking:**
    * Employ more sophisticated re-ranking models, possibly using cross-encoders or even LLMs prompted to assess the relevance of retrieved chunks to the query.
    * Investigate **Agentic RAG** approaches where multiple steps or "agents" collaborate on retrieval, synthesis, and verification, if latency permits.

### LLM Improvements (Generator)

* **Larger Context Windows:** Utilize LLMs capable of handling larger input context windows, allowing more retrieved information to be considered when generating answers.
* **Medical Domain LLMs:** Experiment with LLMs specifically fine-tuned for medical dialogue or text generation (e.g., Med-PaLM variants, Meditron).
* **Model Scale:** Explore larger LLMs (e.g., 7B+ parameters) which generally exhibit better reasoning and generation capabilities, potentially leading to improved ROUGE and BERT scores.
* **Fine-tuning (LoRA):** If a high-quality, curated medical Q&A dataset is available, consider fine-tuning an open-source base LLM using techniques like LoRA (Low-Rank Adaptation) to enhance its factual grounding and response style for the medical domain.
* **LLM-as-Judge:** Employ powerful LLMs to evaluate the generated responses for factual consistency against retrieved sources, relevance to the query, and overall quality.

### Scalability & Feasibility Considerations

* **Latency:** Multiple LLM calls (e.g., for filtering, query expansion, re-ranking, generation, judging) can introduce significant latency, which needs careful management in real-time systems.
* **Hosting Costs:** Deploying multiple large LLMs and maintaining scalable vector databases incurs substantial infrastructure and computational costs.
* **Cost Optimization:** Techniques like LoRA fine-tuning on open-source models might offer a more cost-effective alternative to relying solely on proprietary, paid LLM APIs, especially under high user traffic scenarios.
