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
- **VectorStoreIndex**: Creates an in-memory vector store using **LlamaIndex's** default embedding model (defaults to **OpenAI's embedding model**).
- **Query Engine**: Converts user queries into embedding vectors, retrieves similar chunks, and synthesises responses using **OpenAI's LLM**.
- **Default Behavior**: LlamaIndex uses default embedding models, LLMs, and prompts unless explicitly overridden.
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
    - Analyse how different prompts affect LLM results.

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
    - Use `data/questions.json` as the list of questions for this exercise.
    - Run `01_generate_gold_set.py` to create `data/gold_set.jsonl`. This script asks the LLM to generate answers to the questions in `data/questions.json`.
    - Review and curate the answers for accuracy.

2. **Generate Predictions**:
    - Use the `rag_pipeline` function from Workshop 1 to generate predictions.
    - Run `02_run_predictions.py` to create `data/predictions.jsonl`.

3. **Evaluate Predictions**:
    - Merge predictions and the gold set, and use [Instructor](https://python.useinstructor.com/) for structured evaluation.
    - Run `03_run_evals.py` to compare predictions against the Gold Set.
    - Generate `data/evaluation_report.json`.
    - Use metrics from SKlearn to calculate accuracy, precision, recall, and F1 score.
    - run `04_display_metrics.py` to visualize the evaluation report.

4. **Cost and Latency Evaluation**:
    - Finally , run `05_cost_latency_analysis.py` to run the script to evaluate cost and latency.
    - Track token usage for cost analysis:
        - Input tokens (prompt + context)
        - Output tokens (generated response)
        - Total tokens used per query
    - Measure latency metrics:
        - Retrieval time
        - LLM inference time
        - Total request-to-response time
    - Generate cost efficiency metrics:
        - Cost per correct answer
        - Token efficiency (accuracy per 1K tokens)

#### **Eval Metrics Output**
- **Overall Accuracy**: 71.67%
- **Precision**: 1.0
- **Recall**: 71.67%
- **F1 Score**: 83.5%

#### **Error Analysis Key Insights**
- **Most Common Error Type**: Hallucination
- **Most Problematic Question**: 'What projects or achievements are highlighted in the resume?'

---

## **Using Evaluation Metrics to Improve Your RAG System**

### **Analysing Current Results**
- **Overall Accuracy**: 73.33%
- **Precision**: 1.0
- **F1 Score**: 84.62%
- **Performance Variation**: Significant differences across question types.

### **Improvement Strategy**

1. **Target Low-Performing Question Types**:
    - **Projects/Achievements**: 33.33% accuracy (requires advanced information extraction).
    - **Industries Worked In**: 50% accuracy (involves categorization and inference).
    - **Years of Experience**: 66.67% accuracy (timeline-based extraction).
    - **Programming Languages**: 66.67% accuracy (list-based extraction).

2. **Refine System Components**:
    - **Chunking Strategy**:
        - Experiment with dynamic chunk sizes based on semantic boundaries.
        - Ensure related information remains within the same chunk to improve context.
    - **Prompt Engineering**:
        - Design tailored prompts for each question type, emphasizing clarity and specificity.
        - Include explicit instructions for tasks like calculations, list generation, or timeline extraction.
    - **Retrieval Enhancements**:
        - Increase the number of retrieved chunks for questions requiring detailed context.
        - Combine keyword-based and semantic retrieval methods for better coverage.
    - **Post-Processing Improvements**:
        - Implement structured extraction techniques for lists and categorical data.
        - Validate numerical outputs using rule-based or heuristic checks to ensure consistency.

3. **Iterative Testing and Evaluation**:
    - Continuously test changes against the evaluation metrics (accuracy, precision, recall, F1 score).
    - Use error analysis insights to guide further refinements and prioritize impactful adjustments.

4. **Cost and Latency Optimization**:
    - Monitor token usage and latency metrics to balance performance with efficiency.
    - Focus on reducing retrieval and inference times without compromising accuracy.

By addressing these areas systematically, the RAG system can achieve higher accuracy and reliability across all question types.
