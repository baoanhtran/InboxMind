# InboxMind — AI Gmail Analysis Agent

An intelligent email triage and analysis agent built with **LangChain**, **LangGraph**, and **GPT-4o**.

## Architecture

```
fetch_emails → triage_classifier → [conditional routing]
    ├── attachment_processor   (PDFs, images, xlsx)
    ├── reply_drafter          (GPT-4o draft → GmailCreateDraft)
    └── summarizer             (key info + action items)
         └──→ result_aggregator → human_review → apply_gmail_actions → report_writer
```

## Features

- Fetches unread Gmail messages via the Gmail API
- Classifies each email by intent: `has_attachment`, `needs_reply`, `info_only`
- Extracts content from PDF, Excel, and image attachments (GPT-4o Vision)
- Drafts replies saved to Gmail Drafts (never auto-sends)
- Human-in-the-loop checkpoint before any action is finalised
- Labels each email by intent and marks as read after human approval
- Durable state via LangGraph checkpointing (survives restarts)
- Final structured JSON + Markdown report

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Google credentials

- Go to [Google Cloud Console](https://console.cloud.google.com)
- Create a project, enable the Gmail API
- Create OAuth 2.0 credentials (Desktop app)
- Download `credentials.json` and place it in the project root
- On first run, a browser window opens for consent; `token.json` is created automatically

### 3. Environment variables

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY
```

### 4. Run

```bash
python main.py
```

For the FastAPI server:
```bash
uvicorn api:app --reload
```

## Project Structure

```
inboxmind/
├── agent/
│   ├── state.py          # LangGraph shared state schema
│   ├── graph.py          # StateGraph definition (nodes + edges)
│   └── nodes.py          # All node functions
├── tools/
│   ├── gmail_client.py   # Gmail API wrapper + auth
│   └── attachment.py     # PDF / xlsx / image extractors
├── utils/
│   └── prompts.py        # All LLM prompt templates
├── main.py               # CLI entry point
├── api.py                # FastAPI REST endpoint
├── requirements.txt
└── .env.example
```
