## LLM Zero to Hero  

These are my workshop notes and exercises for the course I'm currently taking - [Building LLM Applications for Data Scientists and Software Engineers](https://maven.com/hugo-stefan/building-llm-apps-ds-and-swe-from-first-principles). Build LLM-powered software reliably & from first principles. Learn the GenAI software development lifecycle: agents, evals, iteration & more.

---

## **Working Locally**

### **1. Clone the Repository**
```bash
git clone https://github.com/jaeyow/llm-zero-to-hero.git
```

### **2. Set Up Python Environment**
```bash
pyenv versions
pyenv install 3.12.8
pyenv local 3.12.8
python -m venv .venv
source .venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements_llama_index.txt  # LlamaIndex RAG
pip install -r requirements_vanilla.txt     # Vanilla RAG
```

---

## **Workshop 1**

### **A) RAG with LlamaIndex**
Run the following command:
```bash
python 01-rag-llamaindex.py
```

#### **My Notes**
- **VectorStoreIndex**: Creates an in-memory vector store using **LlamaIndex's (LI)** default embedding model (defaults to **OpenAI's embedding model**).
- **Query Engine**: Converts user queries into embedding vectors, retrieves similar chunks, and synthesizes responses using **OpenAI's LLM**.
- **Default Behavior**: LI uses default embedding models, LLMs, and prompts unless explicitly overridden.
- **Experiment**: Unset the OpenAI API key to observe app behavior—it prompts for the key since it relies on OpenAI for embeddings and synthesis.
- **Interaction**: Uses a simple **Gradio wrapper** for user interaction.
- **Prompt Analysis**: `query_engine.get_prompts()` reveals multiple prompts, making it unclear which one is used.

#### **Issues with LlamaIndex**
- Defaults to OpenAI models without transparency.
- Unclear prompts sent to the LLM.
- Difficult to understand internal workings, making improvements challenging.

**Summary**: LlamaIndex simplifies RAG creation but introduces too much abstraction, obscuring the process.

---

### **B) Vanilla RAG**
Run the following command:
```bash
python 02-rag-vanilla.py
```

#### **Homework**
Remove abstractions introduced by libraries like LlamaIndex to fully understand the RAG process.

#### **Steps**
1. **Manual RAG Implementation**:
    - Use the OpenAI API directly.
    - Generate embeddings for user queries and documents.
    - Write custom prompt instructions for the LLM.
    - Use the OpenAI LLM for synthesis.

2. **Log Prompts and Queries in SQLite**:
    - Store user queries, prompts, chunk sizes, and LLM responses in a SQLite database.

3. **Experiment with Prompts**:
    - Analyze how different prompts affect LLM results.

#### **My Notes**
- Removing LlamaIndex provides full control over the RAG process and prompts.
- Explicitly use the **OpenAI embedding model** and **OpenAI LLM**.
- Log additional details (e.g., queries, prompts, chunk sizes) in SQLite for better analysis.
- Enables experimentation and iteration on prompts for improved results.

---

## **Deploying to Modal**

### **A) LlamaIndex RAG**
See [Llama Index RAG in Modal](./workshop_1/modal/modal_rag_llamaindex/README.md).

### **B) Vanilla RAG**
See [Vanilla RAG in Modal](./workshop_1/modal/modal_rag_vanilla/README.md).

---

## **Workshop 2**

Workshop 2 focuses on **LLM APIs and Prompt Engineering**, exploring parameters like **Temperature** and **Top-p (nucleus sampling)** and their impact on LLM results. It also involves iterating on the application from Workshop 1 by removing LlamaIndex and using the OpenAI API directly, providing full control over prompts and RAG orchestration.

---

## **Workshop 3**

Workshop 3 introduces **evaluations** for the application, now free from LlamaIndex. The goal is to perform error analysis and improve the system.

### **Synthetic Data for Evaluation**

#### **Exercise**
1. **Create a Gold Set**:
    - Use `data/questions.json` to generate a list of questions.
    - Run `01_generate_gold_set.py` to create `data/gold_set.jsonl`.
    - Review and curate the answers for accuracy.

2. **Generate Predictions**:
    - Run `02_run_predictions.py` to create `data/predictions.jsonl`.

3. **Evaluate Predictions**:
    - Run `03_run_evals.py` to compare predictions against the Gold Set.
    - Generate `data/evaluation_report.json`.

4. **Display Metrics**:
    - Metrics are displayed in STDIO and saved as CSV and HTML reports.

#### **Sample Metrics**
```plaintext
===== Overall Evaluation Metrics =====
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Accuracy  |  0.7333 |
+-----------+---------+
| Precision |  1      |
+-----------+---------+
| Recall    |  0.7333 |
+-----------+---------+
| F1 Score  |  0.8462 |
+-----------+---------+
```
- Metrics saved to:  
  - `/Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/overall_metrics.csv`
  - `/Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/overall_metrics.html`

```plaintext
===== Metrics by Question Type =====
+---------------------------------------------------------------------+------------+---------+
| Question                                                            |   Accuracy |   Count |
+=====================================================================+============+=========+
| What degrees or academic qualifications does the candidate hold?    |     1      |       6 |
+---------------------------------------------------------------------+------------+---------+
| List any certifications or professional training mentioned.         |     1      |       6 |
+---------------------------------------------------------------------+------------+---------+
| What is the candidate's most recent job title?                      |     0.8333 |       6 |
+---------------------------------------------------------------------+------------+---------+
| Does the candidate have any leadership or management experience?    |     0.8333 |       6 |
+---------------------------------------------------------------------+------------+---------+
| Does the candidate have experience working with cloud technologies? |     0.8333 |       6 |
+---------------------------------------------------------------------+------------+---------+
| List the programming languages mentioned in the resume.             |     0.6667 |       6 |
+---------------------------------------------------------------------+------------+---------+
| What is the total number of years of professional experience?       |     0.6667 |       6 |
+---------------------------------------------------------------------+------------+---------+
| Has the candidate worked in any Fortune 500 companies?              |     0.6667 |       6 |
+---------------------------------------------------------------------+------------+---------+
| What industries has the candidate worked in?                        |     0.5    |       6 |
+---------------------------------------------------------------------+------------+---------+
| What projects or achievements are highlighted in the resume?        |     0.3333 |       6 |
+---------------------------------------------------------------------+------------+---------+
```
- Question metrics saved to:  
  - `/Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/question_metrics.csv`

- Plots saved to:  
  - `/Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/question_accuracy.png`  
  - `/Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/correct_vs_incorrect.png`

---

## **Using Evaluation Metrics to Improve Your RAG System**

### **Analyzing Current Results**
- **Overall Accuracy**: 73.33%
- **Precision**: 1.0
- **F1 Score**: 84.62%
- **Performance Variation**: Significant differences across question types.

### **Improvement Strategy**
1. **Focus on Low-Performing Question Types**:
    - **Projects/Achievements**: 33.33% (complex information extraction).
    - **Industries Worked In**: 50% (categorization/inference).
    - **Years of Experience**: 66.67% (timeline extraction).
    - **Programming Languages**: 66.67% (list extraction).

2. **Error Analysis and System Improvements**:
    - **Chunking Strategy**:
        - Optimize chunk sizes or use semantic chunking.
        - Avoid splitting related information.
    - **Prompt Engineering**:
        - Customize prompts for specific question types.
        - Add explicit instructions for calculations or list extractions.
    - **Retrieval Improvements**:
        - Increase retrieved chunks for complex questions.
        - Implement hybrid retrieval (keyword + semantic).
    - **Post-Processing Enhancements**:
        - Add structured extraction for lists.
        - Validate numerical answers with rule-based checks.
