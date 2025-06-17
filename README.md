# Hate Speech Detection Assistant

## Overview

The Hate Speech Detection Assistant is a web-based tool designed to help interns and developers build a GenAI-powered hate speech detection system. The application classifies user input, retrieves related policies, explains moderation decisions, and suggests actions—all within a simple web interface.

---

## Features

- **Text Classification:**  
  Classifies user-submitted text as Hate, Toxic, Offensive, Neutral, or Ambiguous using OpenAI models.

- **Policy Retrieval:**  
  Uses Hybrid RAG (Retrieval-Augmented Generation) to fetch relevant policy/guideline documents from a knowledge base of `.txt` files.

- **Reasoning & Explanation:**  
  Explains the classification decision using LLMs and retrieved policy context.

- **Action Recommendation:**  
  Suggests moderation actions such as remove, warn, flag, or allow.

- **Streamlit UI:**  
  Clean, interactive web interface for input and results.

- **Error Handling:**  
  Graceful error messages and robust backend logic.

---

## Core Components

- **HateSpeechDetectionAgent:**  
  Uses OpenAI to classify input and provide a short explanation.

- **HybridRetrieverAgent:**  
  Retrieves the most relevant `.txt` policy files using sentence-transformers and FAISS/Qdrant.

- **PolicyReasoningAgent:**  
  Combines classification and retrieved docs to justify the decision via OpenAI.

- **ActionRecommenderAgent:**  
  Maps the classification to an action (ban, warn, flag, allow).

- **ErrorHandlerAgent:**  
  Handles errors gracefully and informs the user.

---

## UI

- Built with Streamlit.
- **Input:** Textarea for user content.
- **Output:**  
  - Classification label  
  - Reason  
  - Retrieved policy snippets  
  - Recommended action

---

## Data Requirements

- Place at least **5 `.txt` files** in the `data/policy_docs/` folder.
- Each file should contain moderation guidelines, hate speech policies, or legal references.

**Example: `reddit_policy.txt`**
```
Reddit does not allow content that promotes hate based on identity or vulnerability, including race, ethnicity, religion, gender, sexual orientation, or disability.

Content that incites violence, dehumanizes individuals, or promotes segregation may be removed and result in a user ban.
```
You may use public documents or mock policy text (e.g., Meta policy, Indian Penal Code, Google policy, US laws).

---

## Completion Criteria

- The app detects and classifies text input.
- Retrieves and displays relevant policy context.
- Explains why the content was flagged.
- Suggests an appropriate moderation action.
- Runs without errors on a Windows machine.

---

## Bonus Features

- Audio input (using Whisper)
- Export results to CSV
- REST API version with FastAPI

---

## Development & Code Quality

- Clean, well-documented code (Python docstrings, Pydantic validation)
- Code formatted with [black](https://black.readthedocs.io/en/stable/)
- No extra files like `__pycache__` in git
- README, `requirements.txt`, and `Pipfile` included
- API documentation (Swagger/OpenAPI)
- Unit tests with >80% coverage
- Dockerized app with a compose file

---

## Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd hate-speech-detection
```

### 2. Add your policy documents
Place at least 5 `.txt` files in `data/policy_docs/`.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run locally (Streamlit)
```bash
streamlit run app.py
```

### 5. Run API (FastAPI)
```bash
uvicorn main:app --reload
```

### 6. Run with Docker Compose
```bash
docker-compose up --build
```
- FastAPI API: [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- Qdrant: [http://localhost:6333](http://localhost:6333)

---

## Project Structure

```
.
├── app.py
├── main.py
├── requirements.txt
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── docker-compose.yml
├── data/
│   └── policy_docs/
│       ├── reddit_policy.txt
│       ├── meta_policy.txt
│       └── ...
├── utils/
│   ├── embeddings.py
│   └── qdrant_handler.py
├── agents/
│   ├── hate_speech_agent.py
│   ├── retriever_agent.py
│   ├── reasoning_agent.py
│   ├── action_agent.py
│   └── error_handler.py
└── ...
```

---

## Testing

- Run all unit tests:
```bash
pytest --cov=.
```
- Ensure coverage is >80%.

---

## Documentation

- API documentation available at `/docs` when FastAPI is running.
- Each method and class is documented with Python docstrings.

---

## License

This project is for educational and demonstration purposes.