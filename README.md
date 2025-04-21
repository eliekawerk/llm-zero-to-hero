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
    - Run [01_generate_gold_set.py](./workshop_3/01_generate_gold_set.py) to create `data/gold_set.jsonl`. This script asks the LLM to generate answers to the questions in `data/questions.json`.
    - Review and curate the answers for accuracy.

2. **Generate Predictions**:
    - Use the `rag_pipeline` function from Workshop 1 to generate predictions.
    - Run [02_run_predictions.py](./workshop_3/02_run_predictions.py) to create `data/predictions.jsonl`.

3. **Evaluate Predictions**:
    - Merge predictions and the gold set, and use [Instructor](https://python.useinstructor.com/) for structured evaluation.
    - Run [03_run_evals.py](./workshop_3/03_run_evals.py) to compare predictions against the Gold Set.
    - Generate `data/evaluation_report.json`.
    - Use metrics from SKLearn to calculate accuracy, precision, recall, and F1 score.
    - run [04_display_metrics.py](./workshop_3/04_display_metrics.py) to visualise the evaluation report.

4. **Cost and Latency Evaluation**:
    - Finally , run [05_cost_latency_analysis.py](./workshop_3/05_cost_latency_analysis.py) to run the script to evaluate cost and latency.
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

```
(.venv) (base)  ~/Dev/llm-zero-to-hero/workshop_3   feature/start-evaluation ±  python 04_display_metrics.py

===== Overall Evaluation Metrics =====
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Accuracy  |  0.7167 |
+-----------+---------+
| Precision |  1      |
+-----------+---------+
| Recall    |  0.7167 |
+-----------+---------+
| F1 Score  |  0.835  |
+-----------+---------+
Overall metrics saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/overall_metrics.csv
Overall metrics HTML report saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/overall_metrics.html

===== Metrics by Question Type =====
+---------------------------------------------------------------------+------------+---------+
| Question                                                            |   Accuracy |   Count |
+=====================================================================+============+=========+
| Has the candidate worked in any Fortune 500 companies?              |     1      |       6 |
+---------------------------------------------------------------------+------------+---------+
| List any certifications or professional training mentioned.         |     1      |       6 |
+---------------------------------------------------------------------+------------+---------+
| What is the candidate's most recent job title?                      |     0.8333 |       6 |
+---------------------------------------------------------------------+------------+---------+
| List the programming languages mentioned in the resume.             |     0.6667 |       6 |
+---------------------------------------------------------------------+------------+---------+
| What degrees or academic qualifications does the candidate hold?    |     0.6667 |       6 |
+---------------------------------------------------------------------+------------+---------+
| What is the total number of years of professional experience?       |     0.6667 |       6 |
+---------------------------------------------------------------------+------------+---------+
| What industries has the candidate worked in?                        |     0.6667 |       6 |
+---------------------------------------------------------------------+------------+---------+
| Does the candidate have experience working with cloud technologies? |     0.6667 |       6 |
+---------------------------------------------------------------------+------------+---------+
| Does the candidate have any leadership or management experience?    |     0.5    |       6 |
+---------------------------------------------------------------------+------------+---------+
| What projects or achievements are highlighted in the resume?        |     0.5    |       6 |
+---------------------------------------------------------------------+------------+---------+
Question metrics saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/question_metrics.csv

===== Detailed Evaluation Results =====
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Question                                                            | Resume                     | Gold Answer                                           | Predicted                                             | Correct   |
+=====================================================================+============================+=======================================================+=======================================================+===========+
| What is the candidate's most recent job title?                      | emma-roberts.pdf           | Platform Developer                                    | The candidate's most recent job title is "Platform... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List the programming languages mentioned in the resume.             | emma-roberts.pdf           | Python, Java, JavaScript                              | The programming languages mentioned in the resume ... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have any leadership or management experience?    | emma-roberts.pdf           | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What degrees or academic qualifications does the candidate hold?    | emma-roberts.pdf           | Bachelor of Science in Computer Science from the U... | Emma Roberts holds a Bachelor of Science in Comput... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the total number of years of professional experience?       | emma-roberts.pdf           | 1.5 years                                             | Emma Roberts has over 1.5 years of professional ex... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Has the candidate worked in any Fortune 500 companies?              | emma-roberts.pdf           | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What industries has the candidate worked in?                        | emma-roberts.pdf           | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have experience working with cloud technologies? | emma-roberts.pdf           | Yes, the candidate has experience working with clo... | Yes, the candidate has experience working with clo... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List any certifications or professional training mentioned.         | emma-roberts.pdf           | - Python for Everybody Specialization - Coursera      | The certifications mentioned in the context are:      | ✓         |
|                                                                     |                            | -...                                                  |                                                       |           |
|                                                                     |                            |                                                       | ...                                                   |           |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What projects or achievements are highlighted in the resume?        | emma-roberts.pdf           | Highlighted projects and achievements in the resum... | Emma Roberts' resume highlights the following proj... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the candidate's most recent job title?                      | jane-smith.pdf             | Senior Full Stack Developer                           | The candidate's most recent job title is Senior Fu... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List the programming languages mentioned in the resume.             | jane-smith.pdf             | JavaScript, TypeScript, Python, Java.                 | The programming languages mentioned in the resume ... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have any leadership or management experience?    | jane-smith.pdf             | Yes, the candidate has leadership experience as th... | Yes, the candidate, Jane Smith, has leadership and... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What degrees or academic qualifications does the candidate hold?    | jane-smith.pdf             | Jane Smith holds a Master of Science in Computer S... | Jane Smith holds the following academic qualificat... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the total number of years of professional experience?       | jane-smith.pdf             | Jane Smith has over 11 years of professional exper... | Jane Smith has a total of over 11 years of profess... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Has the candidate worked in any Fortune 500 companies?              | jane-smith.pdf             | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What industries has the candidate worked in?                        | jane-smith.pdf             | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have experience working with cloud technologies? | jane-smith.pdf             | Yes, the candidate has experience working with clo... | Yes, the candidate has experience working with clo... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List any certifications or professional training mentioned.         | jane-smith.pdf             | AWS Certified Solutions Architect – Associate - Am... | The certifications mentioned in the context are:      | ✓         |
|                                                                     |                            |                                                       |                                                       |           |
|                                                                     |                            |                                                       | ...                                                   |           |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What projects or achievements are highlighted in the resume?        | jane-smith.pdf             | The resume highlights the following projects and a... | Jane Smith's resume highlights the following proje... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the candidate's most recent job title?                      | john-doe.pdf               | Frontend Developer at XYZ Tech Solutions, January ... | The candidate's most recent job title is "Frontend... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List the programming languages mentioned in the resume.             | john-doe.pdf               | HTML, CSS, JavaScript, TypeScript                     | The programming languages mentioned in the resume ... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have any leadership or management experience?    | john-doe.pdf               | The candidate has led the frontend development of ... | I don't have enough information to answer that bas... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What degrees or academic qualifications does the candidate hold?    | john-doe.pdf               | Bachelor of Science in Computer Science from Unive... | John Doe holds a Bachelor of Science in Computer S... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the total number of years of professional experience?       | john-doe.pdf               | 1.5 years                                             | Based on the provided context, John Doe has a tota... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Has the candidate worked in any Fortune 500 companies?              | john-doe.pdf               | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What industries has the candidate worked in?                        | john-doe.pdf               | The candidate has worked in the technology and dig... | I don't have enough information to answer that bas... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have experience working with cloud technologies? | john-doe.pdf               | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List any certifications or professional training mentioned.         | john-doe.pdf               | Front-End Web Developer Nanodegree - Udacity          | Front-End Web Developer Nanodegree - Udacity          | ✓         |
|                                                                     |                            | JavaS...                                              | Jav...                                                |           |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What projects or achievements are highlighted in the resume?        | john-doe.pdf               | The resume highlights the following projects and a... | The resume highlights the following projects and a... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the candidate's most recent job title?                      | alex-thompson.pdf          | Backend Developer at Tech Innovations Ltd.            | The candidate's most recent job title is "Backend ... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List the programming languages mentioned in the resume.             | alex-thompson.pdf          | Python, Java, JavaScript, SQL                         | The programming languages mentioned in the resume ... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have any leadership or management experience?    | alex-thompson.pdf          | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What degrees or academic qualifications does the candidate hold?    | alex-thompson.pdf          | Bachelor of Science in Computer Science from Unive... | Bachelor of Science in Computer Science from the U... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the total number of years of professional experience?       | alex-thompson.pdf          | 1.5 years                                             | Alex Thompson has over 1.5 years of professional e... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Has the candidate worked in any Fortune 500 companies?              | alex-thompson.pdf          | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What industries has the candidate worked in?                        | alex-thompson.pdf          | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have experience working with cloud technologies? | alex-thompson.pdf          | Yes, the candidate has experience working with clo... | Yes, the candidate has experience working with clo... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List any certifications or professional training mentioned.         | alex-thompson.pdf          | - Python for Everybody Specialization - Coursera      | Python for Everybody Specialization - Coursera        | ✓         |
|                                                                     |                            | -...                                                  | Jav...                                                |           |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What projects or achievements are highlighted in the resume?        | alex-thompson.pdf          | Highlighted projects and achievements in the resum... | The projects and achievements highlighted in the r... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the candidate's most recent job title?                      | michael-johnson.pdf        | Principal Full Stack Developer                        | The candidate's most recent job title is "Principa... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List the programming languages mentioned in the resume.             | michael-johnson.pdf        | JavaScript, TypeScript, Python, Java                  | The programming languages mentioned in the resume ... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have any leadership or management experience?    | michael-johnson.pdf        | Yes, the candidate has leadership and management e... | Yes, the candidate, Michael Johnson, has leadershi... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What degrees or academic qualifications does the candidate hold?    | michael-johnson.pdf        | The candidate holds a Master of Science in Compute... | Michael Johnson holds a Master of Science in Compu... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the total number of years of professional experience?       | michael-johnson.pdf        | Over 16 years.                                        | The total number of years of professional experien... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Has the candidate worked in any Fortune 500 companies?              | michael-johnson.pdf        | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What industries has the candidate worked in?                        | michael-johnson.pdf        | The candidate has worked in the Financial, Healthc... | The candidate has worked in the financial analytic... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have experience working with cloud technologies? | michael-johnson.pdf        | Yes, the candidate has experience with cloud techn... | Yes, the candidate, Michael Johnson, has experienc... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List any certifications or professional training mentioned.         | michael-johnson.pdf        | AWS Certified Solutions Architect – Professional, ... | - AWS Certified Solutions Architect – Professional... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What projects or achievements are highlighted in the resume?        | michael-johnson.pdf        | The resume highlights the following projects and a... | The projects and achievements highlighted in the r... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the candidate's most recent job title?                      | JO Reyes CV April 2025.pdf | AI/ML Technical Specialist                            | The candidate's most recent job title is "AI/ML Te... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List the programming languages mentioned in the resume.             | JO Reyes CV April 2025.pdf | Python, TypeScript                                    | The programming languages mentioned in the resume ... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have any leadership or management experience?    | JO Reyes CV April 2025.pdf | Yes, the candidate has leadership and management e... | Yes, the candidate has leadership and management e... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What degrees or academic qualifications does the candidate hold?    | JO Reyes CV April 2025.pdf | - Graduate Certificate of Data Science, University... | The candidate holds the following academic qualifi... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What is the total number of years of professional experience?       | JO Reyes CV April 2025.pdf | Over 10 years.                                        | I don't have enough information to answer that bas... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Has the candidate worked in any Fortune 500 companies?              | JO Reyes CV April 2025.pdf | Not mentioned in the resume.                          | I don't have enough information to answer that bas... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What industries has the candidate worked in?                        | JO Reyes CV April 2025.pdf | The candidate has worked in the industries of digi... | The candidate has worked in the following industri... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| Does the candidate have experience working with cloud technologies? | JO Reyes CV April 2025.pdf | Yes, the candidate has over a decade of cloud-nati... | Yes, the candidate has experience working with clo... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| List any certifications or professional training mentioned.         | JO Reyes CV April 2025.pdf | - Graduate Certificate of Data Science, University... | - Graduate Certificate of Data Science, University... | ✓         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+
| What projects or achievements are highlighted in the resume?        | JO Reyes CV April 2025.pdf | The resume highlights the following projects and a... | The resume highlights several projects and achieve... | ✗         |
+---------------------------------------------------------------------+----------------------------+-------------------------------------------------------+-------------------------------------------------------+-----------+

Plot saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/question_accuracy.png
Plot saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/correct_vs_incorrect.png

===== Error Analysis Key Insights =====
- Most common error type: missing_information (9 occurrences)
- Most common cause: retrieval_failure (12 occurrences)
- Most problematic question: 'What projects or achievements are highlighted in the resume?' (3 failures)

===== Error Types =====
+-----------------------+---------+
| Error Type            |   Count |
+=======================+=========+
| missing_information   |       9 |
+-----------------------+---------+
| hallucination         |       4 |
+-----------------------+---------+
| incorrect_information |       4 |
+-----------------------+---------+

===== Error Causes =====
+---------------------+---------+
| Likely Cause        |   Count |
+=====================+=========+
| retrieval_failure   |      12 |
+---------------------+---------+
| llm_reasoning_error |       4 |
+---------------------+---------+
| other               |       1 |
+---------------------+---------+
Error analysis data saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs
Detailed error analysis HTML report saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/error_analysis.html

Metrics display completed successfully.
```

```
(.venv) (base)  ~/Dev/llm-zero-to-hero/workshop_3   feature/start-evaluation ±  python 05_cost_latency_analysis.py

===== Cost and Token Usage Summary =====
+--------------------------+-----------------+
| Metric                   | Value           |
+==========================+=================+
| Total queries            | 60              |
+--------------------------+-----------------+
| Correct answers          | 43 (71.67%)     |
+--------------------------+-----------------+
| Total tokens used        | 91,255          |
+--------------------------+-----------------+
| - Input tokens           | 88,232 (96.7%)  |
+--------------------------+-----------------+
| - Output tokens          | 3,023 (3.3%)    |
+--------------------------+-----------------+
| Total cost               | $0.9730         |
+--------------------------+-----------------+
| - Input cost             | $0.8823 (90.7%) |
+--------------------------+-----------------+
| - Output cost            | $0.0907 (9.3%)  |
+--------------------------+-----------------+
| Average tokens per query | 1520.9          |
+--------------------------+-----------------+
| Average cost per query   | $0.0162         |
+--------------------------+-----------------+
| Cost per correct answer  | $0.0226         |
+--------------------------+-----------------+
| Accuracy per 1K tokens   | 0.4712          |
+--------------------------+-----------------+
Cost summary saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/cost_summary.csv

===== Latency Summary =====
+---------------------+---------------------+
| Metric              | Average (seconds)   |
+=====================+=====================+
| Total response time | 2.23                |
+---------------------+---------------------+
| Retrieval time      | 1.22 (54.6%)        |
+---------------------+---------------------+
| LLM inference time  | 0.92 (41.5%)        |
+---------------------+---------------------+
Latency summary saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/latency_summary.csv

===== Efficiency Metrics by Question Type =====
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| Question                                         | Accuracy   |   Avg. Tokens | Avg. Cost   | Avg. Time   |   Accuracy per 1K tokens |
+==================================================+============+===============+=============+=============+==========================+
| Has the candidate worked in any Fortune 500 c... | 100.00%    |        1486.8 | $0.0152     | 1.88s       |                   0.6899 |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| List any certifications or professional train... | 100.00%    |        1514.5 | $0.0161     | 2.50s       |                   0.6788 |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| What is the candidate's most recent job title... | 83.33%     |        1491.3 | $0.0153     | 2.08s       |                   0.5592 |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| Does the candidate have experience working wi... | 66.67%     |        1502   | $0.0156     | 2.12s       |                   0.4854 |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| List the programming languages mentioned in t... | 66.67%     |        1490.5 | $0.0153     | 1.98s       |                   0.4645 |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| What degrees or academic qualifications does ... | 66.67%     |        1502.5 | $0.0157     | 1.93s       |                   0.4587 |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| What industries has the candidate worked in?     | 66.67%     |        1489.5 | $0.0153     | 1.95s       |                   0.4736 |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| What is the total number of years of professi... | 66.67%     |        1505.2 | $0.0157     | 1.98s       |                   0.497  |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| Does the candidate have any leadership or man... | 50.00%     |        1507.3 | $0.0158     | 2.07s       |                   0.3307 |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
| What projects or achievements are highlighted... | 50.00%     |        1719.5 | $0.0222     | 3.79s       |                   0.304  |
+--------------------------------------------------+------------+---------------+-------------+-------------+--------------------------+
Question efficiency metrics saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/question_efficiency.csv

Visualizations saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs

===== Cost Optimization Recommendations =====
Token Efficiency Improvements:
- Overall token efficiency is low (0.4712 accuracy per 1K tokens)
  Consider optimizing prompts to reduce token usage while maintaining accuracy
- Most token-efficient question: 'Has the candidate worked in any Fortune 500 companies?'
  Efficiency: 0.6899 accuracy per 1K tokens
- Least token-efficient question: 'What projects or achievements are highlighted in the resume?'
  Efficiency: 0.3040 accuracy per 1K tokens
  Consider revising the retrieval or prompt approach for this question type

Cost Optimization Strategies:
- Input tokens account for a high percentage of token usage
  Consider:
  1. Reducing context size by improving chunk relevance
  2. Using more efficient chunking strategies
  3. Implementing a filtering step to remove less relevant chunks
- Most expensive question type: 'What projects or achievements are highlighted in the resume?'
  Average cost: $0.0222 per query
  Note: This question type also has relatively low accuracy (50.00%)
  Focus optimization efforts here for maximum cost-benefit impact

Performance Optimization Opportunities:
- Focus on improving: 'Does the candidate have any leadership or management experience?'
  Current accuracy: 50.00%

Potential Impact:
- Current cost per correct answer: $0.0226
- By improving accuracy to 86.00%, cost could reduce to $0.0189 per correct answer
- Potential savings: $0.0038 per correct answer (16.7% reduction)

Recommendations saved to /Users/josereyes/Dev/llm-zero-to-hero/workshop_3/outputs/cost_optimization_recommendations.txt

Cost and latency analysis completed successfully.
```

#### **Error Analysis Key Insights**
- **Most Common Error Type**: Missing Information
- **Most Problematic Question**: 'What projects or achievements are highlighted in the resume?'

---

## **Using Evaluation Metrics to Improve Your RAG System**

### **Analysing Current Results**
- **Overall Accuracy**: 71.67%
- **Precision**: 1.0
- **F1 Score**: 83.5%
- **Performance Variation**: Significant differences across question types.

### **Improvement Strategy**

1. **Target Low-Performing Question Types**:
    - **Projects/Achievements**: 33.33% accuracy (requires advanced information extraction).
    - **Industries Worked In**: 50% accuracy (involves categorisation and inference).
    - **Years of Experience**: 66.67% accuracy (timeline-based extraction).
    - **Programming Languages**: 66.67% accuracy (list-based extraction).

2. **Refine System Components**:
    - **Chunking Strategy**:
        - Experiment with dynamic chunk sizes based on semantic boundaries.
        - Ensure related information remains within the same chunk to improve context.
    - **Prompt Engineering**:
        - Design tailored prompts for each question type, emphasising clarity and specificity.
        - Include explicit instructions for tasks like calculations, list generation, or timeline extraction.
    - **Retrieval Enhancements**:
        - Increase the number of retrieved chunks for questions requiring detailed context.
        - Combine keyword-based and semantic retrieval methods for better coverage.
    - **Post-Processing Improvements**:
        - Implement structured extraction techniques for lists and categorical data.
        - Validate numerical outputs using rule-based or heuristic checks to ensure consistency.

3. **Iterative Testing and Evaluation**:
    - Continuously test changes against the evaluation metrics (accuracy, precision, recall, F1 score).
    - Use error analysis insights to guide further refinements and prioritise impactful adjustments.

4. **Cost and Latency Optimisation**:
    - Monitor token usage and latency metrics to balance performance with efficiency.
    - Focus on reducing retrieval and inference times without compromising accuracy.

By addressing these areas systematically, the RAG system can achieve higher accuracy and reliability across all question types.
