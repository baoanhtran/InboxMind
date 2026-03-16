"""
agent/nodes.py

Every node function in the InboxMind LangGraph workflow.

Each function signature follows the LangGraph convention:
    def node_name(state: InboxMindState) -> dict
The returned dict is a partial state update — keys not present are unchanged.

Nodes are pure functions: they do not mutate the input state.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import (
    InboxMindState,
    ClassifiedEmail,
    AttachmentResult,
    DraftResult,
    EmailSummary,
)
from tools.gmail_client import (
    build_gmail_service,
    fetch_emails,
    download_attachment,
    create_draft,
)
from tools.attachment import extract_attachment_text
from utils.prompts import (
    triage_prompt,
    attachment_prompt,
    reply_prompt,
    summarizer_prompt,
    report_prompt,
)


# ---------------------------------------------------------------------------
# Shared LLM instance (created once, reused across nodes)
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        api_key=os.environ["OPENAI_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Node 1 — fetch_emails
# ---------------------------------------------------------------------------

def fetch_emails_node(state: InboxMindState) -> dict:
    """
    Uses Gmail API to search for emails matching the query in state.
    Populates state['emails'] with raw EmailRecord objects.
    """
    print(f"[fetch_emails] Fetching up to {state['max_emails']} emails: '{state['gmail_query']}'")

    service = build_gmail_service()
    emails = fetch_emails(
        service=service,
        query=state["gmail_query"],
        max_results=state["max_emails"],
    )

    print(f"[fetch_emails] Found {len(emails)} emails.")
    return {"emails": emails, "errors": []}


# ---------------------------------------------------------------------------
# Node 2 — triage_classifier
# ---------------------------------------------------------------------------

def triage_classifier_node(state: InboxMindState) -> dict:
    """
    Classifies every fetched email using GPT-4o.
    Each email gets an intent, priority, summary, and reasoning.
    """
    llm = _get_llm(temperature=0.1)
    chain = triage_prompt | llm

    classified: list[ClassifiedEmail] = []
    new_errors: list[str] = []

    for email in state["emails"]:
        print(f"[triage_classifier] Classifying: '{email['subject']}'")
        try:
            response = chain.invoke({
                "sender":          email["sender"],
                "subject":         email["subject"],
                "date":            email["date"],
                "has_attachments": str(email["has_attachments"]),
                "body":            email["body"][:3000],
            })

            parsed = json.loads(response.content)
            classified.append(ClassifiedEmail(
                email_id=email["id"],
                intent=parsed.get("intent", "info_only"),
                priority=parsed.get("priority", "low"),
                summary=parsed.get("summary", ""),
                reasoning=parsed.get("reasoning", ""),
            ))

        except Exception as exc:
            new_errors.append(f"triage_classifier failed for {email['id']}: {exc}")
            # Fallback classification
            classified.append(ClassifiedEmail(
                email_id=email["id"],
                intent="info_only",
                priority="low",
                summary=email["snippet"],
                reasoning="Fallback due to classification error.",
            ))

    # Build the processing batch (IDs of all emails)
    batch = [e["id"] for e in state["emails"]]

    return {
        "classified": classified,
        "current_batch": batch,
        "errors": new_errors,
    }


# ---------------------------------------------------------------------------
# Node 3a — attachment_processor
# ---------------------------------------------------------------------------

def attachment_processor_node(state: InboxMindState) -> dict:
    """
    Downloads and extracts text from all email attachments.
    Handles PDFs, Excel, images (GPT-4o Vision), and plain text.
    Then uses GPT-4o to identify key findings.
    """
    llm = _get_llm(temperature=0.1)
    chain = attachment_prompt | llm
    service = build_gmail_service()

    results: list[AttachmentResult] = []
    new_errors: list[str] = []

    # Only process emails classified as has_attachment
    attachment_email_ids = {
        c["email_id"] for c in state["classified"]
        if c["intent"] == "has_attachment"
    }
    target_emails = [e for e in state["emails"] if e["id"] in attachment_email_ids]

    for email in target_emails:
        for att in email["attachments"]:
            att_id    = att["id"]
            filename  = att["filename"]
            mime_type = att["mime_type"]
            size_kb   = att["size"] // 1024

            print(f"[attachment_processor] '{filename}' ({mime_type}, {size_kb} KB) from '{email['subject']}'")
            try:
                raw_bytes = download_attachment(service, email["id"], att_id)

                extracted = extract_attachment_text(
                    raw_bytes=raw_bytes,
                    filename=filename,
                    mime_type=mime_type,
                    llm=llm,
                )

                # Ask GPT-4o to extract key findings from the content
                response = chain.invoke({
                    "filename":       filename,
                    "mime_type":      mime_type,
                    "extracted_text": extracted[:6000],
                })
                parsed = json.loads(response.content)

                results.append(AttachmentResult(
                    email_id=email["id"],
                    filename=filename,
                    mime_type=mime_type,
                    extracted_text=extracted[:2000],
                    key_findings=parsed.get("key_findings", []),
                ))
                print(f"[attachment_processor] Extracted {len(parsed.get('key_findings', []))} findings from '{filename}'")

            except Exception as exc:
                new_errors.append(f"attachment_processor failed for '{filename}' ({att_id}): {exc}")

    return {"attachment_results": results, "errors": new_errors}


# ---------------------------------------------------------------------------
# Node 3b — reply_drafter
# ---------------------------------------------------------------------------

def reply_drafter_node(state: InboxMindState) -> dict:
    """
    Drafts GPT-4o replies for emails classified as 'needs_reply'.
    Saves each draft to Gmail Drafts (never auto-sends).
    """
    llm = _get_llm(temperature=0.4)
    chain = reply_prompt | llm
    service = build_gmail_service()

    drafts: list[DraftResult] = []
    new_errors: list[str] = []

    reply_email_ids = {
        c["email_id"] for c in state["classified"]
        if c["intent"] == "needs_reply"
    }
    target_emails = [e for e in state["emails"] if e["id"] in reply_email_ids]

    for email in target_emails:
        print(f"[reply_drafter] Drafting reply to: '{email['subject']}'")
        try:
            response = chain.invoke({
                "sender":  email["sender"],
                "subject": email["subject"],
                "body":    email["body"][:3000],
            })
            draft_body = response.content

            # Extract sender email address (handles "Name <email@>" format)
            sender_email = email["sender"]
            if "<" in sender_email and ">" in sender_email:
                sender_email = sender_email.split("<")[1].rstrip(">")

            draft_id = create_draft(
                service=service,
                to=sender_email,
                subject=email["subject"],
                body=draft_body,
            )

            drafts.append(DraftResult(
                email_id=email["id"],
                draft_id=draft_id,
                draft_subject=email["subject"],
                draft_body=draft_body,
                recipient=sender_email,
            ))
            print(f"[reply_drafter] Draft saved: {draft_id}")

        except Exception as exc:
            new_errors.append(f"reply_drafter failed for {email['id']}: {exc}")

    return {"draft_results": drafts, "errors": new_errors}


# ---------------------------------------------------------------------------
# Node 3c — summarizer
# ---------------------------------------------------------------------------

def summarizer_node(state: InboxMindState) -> dict:
    """
    Produces structured summaries (key points + action items) for
    emails classified as 'info_only' (and any others not handled by
    attachment or reply branches).
    """
    llm = _get_llm(temperature=0.1)
    chain = summarizer_prompt | llm

    summaries: list[EmailSummary] = []
    new_errors: list[str] = []

    # Summarise all emails regardless of intent (different branches may have
    # already handled drafts/attachments, but a summary is always useful)
    for email in state["emails"]:
        print(f"[summarizer] Summarising: '{email['subject']}'")
        try:
            response = chain.invoke({
                "sender":  email["sender"],
                "subject": email["subject"],
                "body":    email["body"][:3000],
            })
            parsed = json.loads(response.content)

            summaries.append(EmailSummary(
                email_id=email["id"],
                key_points=parsed.get("key_points", []),
                action_items=parsed.get("action_items", []),
                deadline=parsed.get("deadline"),
            ))

        except Exception as exc:
            new_errors.append(f"summarizer failed for {email['id']}: {exc}")

    return {"summaries": summaries, "errors": new_errors}


# ---------------------------------------------------------------------------
# Node 4 — result_aggregator
# ---------------------------------------------------------------------------

def result_aggregator_node(state: InboxMindState) -> dict:
    """
    Merges all branch outputs into a single structured dict that the
    report_writer and human_review nodes can easily consume.
    """
    print("[result_aggregator] Merging outputs...")

    # Build lookup maps for fast join
    classified_map  = {c["email_id"]: c for c in state.get("classified", [])}
    summary_map     = {s["email_id"]: s for s in state.get("summaries", [])}
    attachment_map: dict[str, list] = {}
    for att in state.get("attachment_results", []):
        attachment_map.setdefault(att["email_id"], []).append(att)
    draft_map = {d["email_id"]: d for d in state.get("draft_results", [])}

    aggregated: list[dict[str, Any]] = []
    for email in state["emails"]:
        eid = email["id"]
        aggregated.append({
            "email": {
                "id":      eid,
                "subject": email["subject"],
                "sender":  email["sender"],
                "date":    email["date"],
                "snippet": email["snippet"],
            },
            "classification": classified_map.get(eid),
            "summary":        summary_map.get(eid),
            "attachments":    attachment_map.get(eid, []),
            "draft":          draft_map.get(eid),
        })

    output = {
        "processed_at":    datetime.utcnow().isoformat() + "Z",
        "total_emails":    len(state["emails"]),
        "total_drafts":    len(state.get("draft_results", [])),
        "total_attachments": len(state.get("attachment_results", [])),
        "emails":          aggregated,
        "all_errors":      state.get("errors", []),
    }

    return {"aggregated_output": output}


# ---------------------------------------------------------------------------
# Node 5 — human_review  (Human-in-the-Loop checkpoint)
# ---------------------------------------------------------------------------

def human_review_node(state: InboxMindState) -> dict:
    """
    Presents the aggregated output to the human operator via the CLI.
    The operator can approve, request a retry, or skip.

    In production this node is used with LangGraph's interrupt() mechanism:
    the graph pauses here and resumes when the human sends a decision via
    the API (see api.py). In this CLI version, we read from stdin.
    """
    from langgraph.types import interrupt

    agg = state.get("aggregated_output", {})

    summary_lines = [
        f"\n{'='*60}",
        "  InboxMind — Human Review Checkpoint",
        f"{'='*60}",
        f"  Emails processed : {agg.get('total_emails', 0)}",
        f"  Drafts created   : {agg.get('total_drafts', 0)}",
        f"  Attachments read : {agg.get('total_attachments', 0)}",
        f"  Errors           : {len(agg.get('all_errors', []))}",
        f"{'='*60}",
    ]

    for item in agg.get("emails", []):
        cl = item.get("classification") or {}
        summary_lines.append(
            f"  [{cl.get('priority','?').upper():6}] [{cl.get('intent','?'):15}] "
            f"{item['email']['subject'][:50]} — {item['email']['sender'][:30]}"
        )
        if item.get("draft"):
            summary_lines.append(f"           → Draft saved: {item['draft']['draft_id']}")
        for att in item.get("attachments", []):
            summary_lines.append(f"           → Attachment: {att['filename']} ({len(att['key_findings'])} findings)")

    print("\n".join(summary_lines))

    # LangGraph's interrupt() suspends execution and returns a value to the caller.
    # When the graph is resumed (via .invoke() with a Command), the returned value
    # becomes the result of the interrupt() call here.
    decision = interrupt({
        "message": "Review the inbox analysis above. Approve to generate the final report.",
        "aggregated_output": agg,
    })

    # decision is expected to be {"approved": True/False, "feedback": "..."}
    approved  = decision.get("approved", True) if isinstance(decision, dict) else True
    feedback  = decision.get("feedback", "") if isinstance(decision, dict) else ""

    return {
        "human_decision": {"approved": approved, "feedback": feedback},
        "retry_count": state.get("retry_count", 0) + (0 if approved else 1),
    }


# ---------------------------------------------------------------------------
# Node 6 — report_writer
# ---------------------------------------------------------------------------

def report_writer_node(state: InboxMindState) -> dict:
    """
    Uses GPT-4o to write a clean Markdown report from the aggregated output.
    Saves it to a file and returns it in state.
    """
    llm = _get_llm(temperature=0.3)
    chain = report_prompt | llm

    print("[report_writer] Generating final report...")

    agg_json = json.dumps(state["aggregated_output"], indent=2, default=str)

    # Add human feedback to the prompt if any
    feedback = ""
    if state.get("human_decision") and state["human_decision"].get("feedback"):
        feedback = f"\n\nHuman reviewer notes: {state['human_decision']['feedback']}"

    response = chain.invoke({"aggregated_json": agg_json + feedback})
    report = response.content

    # Save to disk
    output_path = os.environ.get("REPORT_OUTPUT_PATH", "./report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[report_writer] Report saved to {output_path}")
    return {"final_report": report, "processing_complete": True}