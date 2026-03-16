"""
utils/prompts.py

All GPT-4o prompt templates used by InboxMind nodes.
Keeping prompts centralised makes tuning and A/B testing easy.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ---------------------------------------------------------------------------
# triage_classifier
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM = """You are an expert email triage assistant. 
Analyse the email and return a JSON object with these exact keys:
  "intent"    : one of "has_attachment" | "needs_reply" | "info_only"
  "priority"  : one of "high" | "medium" | "low"
  "summary"   : a single sentence describing the email
  "reasoning" : one sentence explaining your intent classification

Rules:
- "has_attachment" if the email has files/documents that need analysis
- "needs_reply" if the sender is waiting for a response
- "info_only" for newsletters, notifications, FYI emails, confirmations
- Priority "high" = deadline within 48 h, financial, or urgent language

Return ONLY valid JSON, no markdown fences."""

TRIAGE_HUMAN = """From: {sender}
Subject: {subject}
Date: {date}
Has attachments: {has_attachments}

Body:
{body}"""

triage_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(TRIAGE_SYSTEM),
    HumanMessagePromptTemplate.from_template(TRIAGE_HUMAN),
])

# ---------------------------------------------------------------------------
# attachment_processor
# ---------------------------------------------------------------------------

ATTACHMENT_SYSTEM = """You are a document analyst. 
Given extracted text from a file, return a JSON object with:
  "key_findings" : list of up to 5 bullet-point strings (the most important facts)
  "summary"      : 2-3 sentence summary of the document

Focus on numbers, dates, names, decisions, and action items.
Return ONLY valid JSON, no markdown fences."""

ATTACHMENT_HUMAN = """Filename: {filename}
File type: {mime_type}

Extracted content:
{extracted_text}"""

attachment_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(ATTACHMENT_SYSTEM),
    HumanMessagePromptTemplate.from_template(ATTACHMENT_HUMAN),
])

# ---------------------------------------------------------------------------
# reply_drafter
# ---------------------------------------------------------------------------

REPLY_SYSTEM = """You are a professional email assistant helping draft replies.
Write a polite, concise, and contextually appropriate reply.
Match the formality level of the original email.
Do not make up facts — if you don't know something, use a placeholder like [YOUR ANSWER HERE].
Return ONLY the email body text, no subject line, no metadata."""

REPLY_HUMAN = """Original email from {sender}:
Subject: {subject}

{body}

Write a reply:"""

reply_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(REPLY_SYSTEM),
    HumanMessagePromptTemplate.from_template(REPLY_HUMAN),
])

# ---------------------------------------------------------------------------
# summarizer
# ---------------------------------------------------------------------------

SUMMARIZER_SYSTEM = """You are an executive assistant extracting structured information from emails.
Return a JSON object with:
  "key_points"   : list of up to 5 concise bullet strings
  "action_items" : list of action items (things someone must DO), empty list if none
  "deadline"     : ISO date string if a deadline is mentioned, else null

Return ONLY valid JSON, no markdown fences."""

SUMMARIZER_HUMAN = """From: {sender}
Subject: {subject}

{body}"""

summarizer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SUMMARIZER_SYSTEM),
    HumanMessagePromptTemplate.from_template(SUMMARIZER_HUMAN),
])

# ---------------------------------------------------------------------------
# report_writer
# ---------------------------------------------------------------------------

REPORT_SYSTEM = """You are a productivity assistant writing a clean Markdown inbox report.
Use the structured data provided to produce a readable, well-organised report.
Include:
1. An executive summary (2-3 sentences)
2. A table of all emails with: Subject | Sender | Priority | Intent
3. Sections for high-priority items, drafts created, and attachment findings
4. A consolidated action items list

Use standard Markdown. Be concise."""

REPORT_HUMAN = """Inbox analysis data:
{aggregated_json}

Write the Markdown report:"""

report_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(REPORT_SYSTEM),
    HumanMessagePromptTemplate.from_template(REPORT_HUMAN),
])
