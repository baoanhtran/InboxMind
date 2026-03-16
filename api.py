"""
api.py

FastAPI REST server for InboxMind.

Exposes the LangGraph agent as an HTTP API, enabling integration with
frontends, schedulers, or other services.

Endpoints:
    POST /runs            — Start a new analysis run
    GET  /runs/{run_id}   — Get the current state of a run
    POST /runs/{run_id}/resume — Resume a HITL-paused run with a human decision
    GET  /runs/{run_id}/report — Get the final Markdown report

Usage:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from langgraph.types import Command

load_dotenv()

app = FastAPI(
    title="InboxMind API",
    description="AI Gmail analysis agent powered by LangChain + LangGraph + GPT-4o",
    version="1.0.0",
)

# Lazy-load the graph (avoids slow import on module load for CLI use)
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        from agent.graph import build_graph_with_sqlite
        _graph = build_graph_with_sqlite(db_path="inboxmind_api.db")
    return _graph


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class StartRunRequest(BaseModel):
    gmail_query: str = "is:unread newer_than:1d"
    max_emails: int = 10
    thread_id: Optional[str] = None    # auto-generated if not provided


class StartRunResponse(BaseModel):
    run_id: str
    thread_id: str
    status: str
    message: str


class RunStatusResponse(BaseModel):
    run_id: str
    thread_id: str
    status: str                        # "running" | "paused_for_review" | "complete" | "error"
    emails_processed: int
    drafts_created: int
    attachments_processed: int
    errors: list[str]
    paused_at: Optional[str]
    next_nodes: list[str]


class ResumeRequest(BaseModel):
    approved: bool = True
    feedback: Optional[str] = ""


class ResumeResponse(BaseModel):
    run_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# In-memory run registry (maps run_id → thread_id)
# In production, use a database.
# ---------------------------------------------------------------------------
_run_registry: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/runs", response_model=StartRunResponse, status_code=201)
async def start_run(request: StartRunRequest):
    """
    Start a new InboxMind analysis run.
    The graph will run until it hits the human_review checkpoint,
    then pause and wait for a /resume call.
    """
    graph = get_graph()

    run_id    = str(uuid.uuid4())
    thread_id = request.thread_id or f"run-{run_id[:8]}"
    _run_registry[run_id] = thread_id

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages":            [],
        "gmail_query":         request.gmail_query,
        "max_emails":          request.max_emails,
        "emails":              [],
        "classified":          [],
        "current_batch":       [],
        "attachment_results":  [],
        "draft_results":       [],
        "summaries":           [],
        "aggregated_output":   {},
        "human_decision":      None,
        "final_report":        None,
        "errors":              [],
        "retry_count":         0,
        "processing_complete": False,
    }

    # Run until the HITL interrupt (non-blocking via background task in production)
    # For simplicity here we run synchronously; use BackgroundTasks for production
    try:
        graph.invoke(initial_state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph execution error: {exc}")

    # Check current state
    snapshot = graph.get_state(config)
    is_paused = bool(snapshot.next)

    return StartRunResponse(
        run_id=run_id,
        thread_id=thread_id,
        status="paused_for_review" if is_paused else "complete",
        message=(
            "Run paused for human review. Call POST /runs/{run_id}/resume to continue."
            if is_paused else
            "Run completed successfully."
        ),
    )


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Get the current status and summary of a run."""
    thread_id = _run_registry.get(run_id)
    if not thread_id:
        raise HTTPException(status_code=404, detail="Run not found")

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    if not snapshot:
        raise HTTPException(status_code=404, detail="No state found for this run")

    values = snapshot.values
    is_paused = bool(snapshot.next)
    is_complete = values.get("processing_complete", False)

    if is_complete:
        status = "complete"
    elif is_paused:
        status = "paused_for_review"
    else:
        status = "running"

    return RunStatusResponse(
        run_id=run_id,
        thread_id=thread_id,
        status=status,
        emails_processed=len(values.get("emails", [])),
        drafts_created=len(values.get("draft_results", [])),
        attachments_processed=len(values.get("attachment_results", [])),
        errors=values.get("errors", []),
        paused_at=str(snapshot.next) if is_paused else None,
        next_nodes=list(snapshot.next) if snapshot.next else [],
    )


@app.post("/runs/{run_id}/resume", response_model=ResumeResponse)
async def resume_run(run_id: str, request: ResumeRequest):
    """
    Resume a run paused at the human_review checkpoint.
    Provide your decision: approved=true to generate the report,
    or approved=false with feedback to re-run the triage.
    """
    thread_id = _run_registry.get(run_id)
    if not thread_id:
        raise HTTPException(status_code=404, detail="Run not found")

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=400, detail="This run is not paused — nothing to resume")

    try:
        graph.invoke(
            Command(resume={"approved": request.approved, "feedback": request.feedback}),
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Resume error: {exc}")

    return ResumeResponse(
        run_id=run_id,
        status="complete" if request.approved else "retrying",
        message=(
            "Report generated. Call GET /runs/{run_id}/report to retrieve it."
            if request.approved else
            "Re-running triage with your feedback."
        ),
    )


@app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
async def get_report(run_id: str):
    """
    Retrieve the final Markdown report for a completed run.
    Returns plain text (Markdown).
    """
    thread_id = _run_registry.get(run_id)
    if not thread_id:
        raise HTTPException(status_code=404, detail="Run not found")

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    if not snapshot:
        raise HTTPException(status_code=404, detail="No state found for this run")

    report = snapshot.values.get("final_report")
    if not report:
        raise HTTPException(
            status_code=404,
            detail="Report not yet generated. Ensure the run is complete.",
        )

    return report


@app.get("/health")
async def health():
    return {"status": "ok", "service": "InboxMind API"}
