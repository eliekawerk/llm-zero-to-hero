## LLM Zero to Hero  

These are my workshop notes and exercises for the course I'm currently taking - [Building LLM Applications for Data Scientists and Software Engineers](https://maven.com/hugo-stefan/building-llm-apps-ds-and-swe-from-first-principles). Build LLM-powered software reliably & from first principles. Learn the GenAI software development lifecycle: agents, evals, iteration & more.

---

## **Working Locally :**  
1. Clone the repository:  
    ```bash
    git clone https://github.com/jaeyow/llm-zero-to-hero.git
    ```  

    ```bash
    pyenv versions
    pyenv install 3.12.8
    pyenv local 3.12.8
    python -m venv .venv
    source .venv/bin/activate
    ```
2. Install dependencies:  
    ```bash
    pip install -r requirements_llama_index.txt # LlamaIndex RAG
    pip install -r requirements_vanilla.txt # Vanilla RAG
    ```  

## **Workshop 1**

### **A) RAG with LlamaIndex**
```
python 01-rag-llamaindex.py
```
**My notes:**

- The **VectorStoreIndex** creates a vector store in-memory, and uses **LlamaIndex's (LI)** default embedding model which is whatever **OpenAI's default embedding model** is.
- the resulting query engine is basically LlamaIndex's implementation of a simple RAG system
- we don't see it, but what is happening under the covers is the LI converts the user query to an embedding vector, and searches the most similar chunks from this in-memory vector store.
- These similar chunks are then sent to the LLM, in this case it is using the **OpenAI's default LLM** for synthesis, and the LLM replies with the nice natural language response. LI has default embedding models and LLMs, if you don't specify.
- LI just assumes that you always want to use OpenAI. But of course these can be overridden if required.
- so it is indeed a **RAG**, just for a single document and using many defaults, (**even the prompts are default prompts**), and it does many things under the covers that we don't see.
- as an experiment, try unsetting OpenAI's API key and run the app. You'll see that it will ask for an OpenAI API key... As it needs an embedding model and LLM to do the RAG.
- to be able to interact with the application this app uses a simple Gradio wrapper
- I tried to extract the actual prompt that LI uses, with a call to `query_engine.get_prompts()`, and even though I can see the prompts being used, it lists several prompts and its not clear which one is being used.
- Yes, I catch myself saying "Fuck you, show me the prompts", [Hamel's article](https://hamel.dev/blog/posts/prompt/) captures exactly what I'm thinking. 

**What's the problem here?**

LlamaIndex is a great LLM abstraction, don't get me wrong. In fact, it is one of my favourite LLM Libraries and often use it my applications. However, this abstraction has a few issues I can see:
- Defaults to using OpenAI embedding model and LLM under the covers. It is not clear until you look at the errors and logs. 
- Unclear what prompts its sending to the LLM. It can be updated, but unclear how.
- Difficult to understand exactly what is happening in the app, so it is difficult to improve it.

Basically, we were able to create a RAG system, that works, but with little understanding what it actually does. Too much magic is going on. I don't like it. 

### **B) Vanilla RAG** 
```
python 02-rag-vanilla.py
```
The homework for the first workshop is to remove the abstractions that is added by these libraries with the purpose of fully understanding what is going on.

This activity will help you understand the importance of owning the prompts in your system. Have full control over the prompts and the data that is being sent to the LLM. This will enable you to improve the results of your system, and also understand what is going on under the covers.

**1. Remove LlamaIndex and perform the RAG manually**
- Use the OpenAI API directly
- Use the OpenAI embedding model to create the embeddings of both the user query and the document
- Write your own prompt instructions to the LLM
- Use the OpenAI LLM to perform the synthesis

**2. Log the prompts and user queries in the SQLite database**
- Use the SQLite database to store the user queries and the prompts that are being sent to the LLM, as well as the chunk size and the response of the LLM

**3. Play with different prompts and see how it affects the results**
- Understand how the prompts affect the results of the LLM

**My notes:**
Now that we have ripped out LlamaIndex, we can see what is going on under the covers. We are using the OpenAI API directly, and we have full control over the prompts that are being sent to the LLM. We can also log the prompts and user queries in the SQLite database, which will help us understand how the prompts affect the results of the LLM.
- We are explicitly using the **OpenAI embedding model** to create the embeddings of both the user query and the document.
- We are explicitly using the **OpenAI LLM** to perform the synthesis.
- We have also logged additional information in the SQLite database, such as the user query, the prompts that was sent to the LLM, chunk size, and the response of the LLM. This will help us understand how the prompts affect the results of the LLM. 
- Before, with LlamaIndex, the steps that took for the whole RAG process was hidden under the covers. Now we can see exactly what is going on, and we have full control over the prompts that are being sent to the LLM. 
- Now we are ready to play and experiment and iterate on the prompts.
