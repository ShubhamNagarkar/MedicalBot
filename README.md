# MedicalBot

**Problem Statement** <br>
To develop a medical question-answering system utilizing the MedQuad Dataset. The goal is to create a model that can effectively answer user queries related to medical diseases.

**Structure of the Repository** <br>
- **data**: this folder contains the original dataset 'medical_dataset.csv'. Additionally, it contains a processed and cleaner version of the original dataset 'clean_questions_to_answers_dataset_v1.json'
- **data-analysis.ipynb**: This notebook contains basic data exploration and analysis on the medical_dataset.csv. The output of this notebook is the processed and clean dataset.
- **RAG_retrieval_analysis.ipynb**: This notebook contains benchmarking experiments with respect to chunking, indexing, retrieval logic and retrieval testing.
- **RAGMedicalBot.ipynb**: This notebook contains the main code to interface with the LLM Chat Bot. It also contains the LLM evaluation section.
- **utils.py**: This file contains file reading/writing utility functions.




