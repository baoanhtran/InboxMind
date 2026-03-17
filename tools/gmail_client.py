"""
tools/gmail_client.py

Wraps LangChain's GmailToolkit to provide a clean Python API for
fetching emails and creating drafts. Authentication is handled via
the standard OAuth2 flow with a local credentials.json file.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import build_resource_service

from agent.state import EmailRecord


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

SCOPES = ["https://mail.google.com/"]

def build_gmail_service():
    """
    Builds and returns the Google API resource object.
    Uses standard google-auth-oauthlib OAuth2 flow.
    On first run, opens a browser for consent and saves token.json.
    Subsequent runs refresh/reuse token.json silently.
    """
    import os
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_toolkit() -> GmailToolkit:
    """Returns a fully authenticated GmailToolkit instance."""
    api_resource = build_gmail_service()
    return GmailToolkit(api_resource=build_resource_service(api_resource))


# ---------------------------------------------------------------------------
# Email fetching
# ---------------------------------------------------------------------------

def fetch_emails(
    service,
    query: str = "is:unread",
    max_results: int = 20,
) -> list[EmailRecord]:
    """
    Search Gmail and return a list of EmailRecord dicts.

    Args:
        service:     The Google API resource from build_gmail_service()
        query:       Gmail search query string (same syntax as the Gmail search box)
        max_results: Maximum number of emails to fetch

    Returns:
        List of EmailRecord TypedDicts
    """
    # 1. Search for matching message IDs
    result = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )
    messages = result.get("messages", [])

    records: list[EmailRecord] = []
    for msg_ref in messages:
        msg_id = msg_ref["id"]
        try:
            record = fetch_single_email(service, msg_id)
            records.append(record)
        except Exception as exc:
            # Non-fatal: skip problematic emails and continue
            print(f"[gmail_client] Warning: could not fetch email {msg_id}: {exc}")

    return records


def fetch_single_email(service, msg_id: str) -> EmailRecord:
    """Fetch a single email by ID and parse it into an EmailRecord."""
    msg = (
        service.users()
        .messages()
        .get(userId="me", id=msg_id, format="full")
        .execute()
    )

    headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
    subject = headers.get("Subject", "(no subject)")
    sender  = headers.get("From", "(unknown sender)")
    date    = headers.get("Date", "")

    body, attachments = _parse_payload(msg["payload"])

    return EmailRecord(
        id=msg_id,
        thread_id=msg.get("threadId", ""),
        subject=subject,
        sender=sender,
        date=date,
        body=body[:8000],           # cap to stay within token limits
        snippet=msg.get("snippet", ""),
        has_attachments=bool(attachments),
        attachments=attachments,
    )


def _parse_payload(payload: dict) -> tuple[str, list[dict]]:
    """
    Recursively extract plain-text body and full attachment metadata
    from a Gmail message payload.

    Each attachment entry is a dict with keys:
        id        — Gmail attachmentId (used to download)
        filename  — original filename (e.g. "report.pdf")
        mime_type — MIME type (e.g. "application/pdf")
        size      — size in bytes
    """
    body_text = ""
    attachments: list[dict] = []

    mime_type = payload.get("mimeType", "")
    parts = payload.get("parts", [])

    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            body_text = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    elif mime_type.startswith("multipart/"):
        for part in parts:
            sub_body, sub_attachments = _parse_payload(part)
            if sub_body and not body_text:
                body_text = sub_body
            attachments.extend(sub_attachments)

    # Detect attachment parts: they have a filename and an attachmentId
    filename  = payload.get("filename", "")
    body_data = payload.get("body", {})
    att_id    = body_data.get("attachmentId", "")

    if filename and att_id:
        attachments.append({
            "id":        att_id,
            "filename":  filename,
            "mime_type": mime_type or "application/octet-stream",
            "size":      body_data.get("size", 0),
        })

    return body_text, attachments


# ---------------------------------------------------------------------------
# Attachment downloading
# ---------------------------------------------------------------------------

def download_attachment(service, msg_id: str, attachment_id: str) -> bytes:
    """Download a Gmail attachment and return raw bytes."""
    attachment = (
        service.users()
        .messages()
        .attachments()
        .get(userId="me", messageId=msg_id, id=attachment_id)
        .execute()
    )
    data = attachment.get("data", "")
    return base64.urlsafe_b64decode(data + "==")


# ---------------------------------------------------------------------------
# Draft creation
# ---------------------------------------------------------------------------

def create_draft(service, to: str, subject: str, body: str) -> str:
    """
    Save an email draft to Gmail Drafts and return the draft ID.
    The draft is NEVER automatically sent.
    """
    import email as email_lib
    from email.mime.text import MIMEText

    message = MIMEText(body)
    message["to"]      = to
    message["subject"] = f"Re: {subject}" if not subject.startswith("Re:") else subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    draft = (
        service.users()
        .drafts()
        .create(userId="me", body={"message": {"raw": raw}})
        .execute()
    )
    return draft["id"]


# ---------------------------------------------------------------------------
# Label management & mark-as-read
# ---------------------------------------------------------------------------

_label_cache: dict[str, str] = {}


def ensure_label_exists(service, label_name: str) -> str:
    """
    Return the Gmail label ID for *label_name*, creating the label if it
    does not already exist.  Results are cached for the process lifetime.
    """
    if label_name in _label_cache:
        return _label_cache[label_name]

    results = service.users().labels().list(userId="me").execute()
    for label in results.get("labels", []):
        if label["name"].lower() == label_name.lower():
            _label_cache[label_name] = label["id"]
            return label["id"]

    # Label doesn't exist yet — create it
    created = (
        service.users()
        .labels()
        .create(
            userId="me",
            body={
                "name": label_name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            },
        )
        .execute()
    )
    _label_cache[label_name] = created["id"]
    return created["id"]


def label_and_mark_read(service, msg_id: str, label_id: str) -> None:
    """
    Apply a label and mark a message as read in a single API call.
    Removing the ``UNREAD`` label is how Gmail marks a message as read.
    """
    service.users().messages().modify(
        userId="me",
        id=msg_id,
        body={
            "addLabelIds": [label_id],
            "removeLabelIds": ["UNREAD"],
        },
    ).execute()