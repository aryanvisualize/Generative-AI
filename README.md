# Generative AI Experiments

A small collection of LangChain experiments using hosted and local language models, embeddings, and Streamlit interfaces.

## Features

- CineSage: extract structured movie information from free-form text.
- Chatbot: chat with an AI agent in angry, funny, or sad mode.
- Chat model examples for Google Gemini, Hugging Face, and a local Hugging Face pipeline.
- Google Gemini embedding example.

## Requirements

- Python 3.10 or newer
- API credentials for the provider used by the example you want to run
- A working C++ build toolchain may be required by some local Hugging Face dependencies.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a local environment file from the template and add your own credentials:

```powershell
Copy-Item .env.examples .env
```

`.env` is ignored by Git. Never commit real API keys or tokens.

## Run the applications

Launch the movie information extractor:

```powershell
streamlit run CineSage/UICore.py
```

Launch the mode-based chatbot:

```powershell
streamlit run chatbot/chatbot.py
```

## Run the examples

The remaining scripts can be run directly after configuring the matching provider credential:

```powershell
python chatmodels/chat.py
python chatmodels/huggingface.py
python chatmodels/localmodel.py
python embeddingmodels/embedding.py
python embeddingmodels/huggingface_embedding.py
python CineSage/core.py
```

The local model example downloads model files on its first run and can require substantial disk space and memory.

## Project layout

```text
chatbot/             Streamlit conversational agent
chatmodels/          Hosted and local chat model examples
CineSage/            Movie information extraction app and core logic
embeddingmodels/     Embedding model examples
requirements.txt     Python dependencies
.env.examples        Environment variable template
```

## Environment variables

Set only the credentials required by the example you are running:

| Variable | Used by |
| --- | --- |
| `MISTRAL_API_KEY` | CineSage and the chatbot |
| `GOOGLE_API_KEY` | Gemini chat and embedding examples |
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face hosted model example |
