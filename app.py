"""Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()
st.set_page_config(page_title="HITL PR Review", layout="wide")


def _init_state() -> None:
    defaults = {
        "thread_id": None,
        "pr_url": "",
        "interrupt_payload": None,
        "final": None,
        "checkpoint_id": None,
        "reviewer_id": os.environ.get("GITHUB_USER") or os.environ.get("REVIEWER_ID") or "",
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


async def list_recent_sessions(limit: int = 10) -> list[dict[str, Any]]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level
                                WHEN 'high' THEN 3
                                WHEN 'med' THEN 2
                                ELSE 1
                            END)
                        WHEN 3 THEN 'high'
                        WHEN 2 THEN 'med'
                        ELSE 'low'
                   END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def calibration_summary() -> dict[str, Any]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT COUNT(*) AS reviewed,
                   AVG(confidence) AS avg_confidence,
                   SUM(CASE WHEN decision = 'approve' THEN 1 ELSE 0 END) AS approvals,
                   SUM(CASE WHEN decision = 'reject' THEN 1 ELSE 0 END) AS rejections,
                   SUM(CASE WHEN decision = 'edit' THEN 1 ELSE 0 END) AS edits
              FROM audit_events
             WHERE action = 'human_approval'
               AND decision IN ('approve', 'reject', 'edit')
            """
        ) as cur:
            human = dict(await cur.fetchone())
        async with conn.execute(
            """
            SELECT COUNT(*) AS total,
                   AVG(confidence) AS avg_confidence
              FROM audit_events
             WHERE action IN ('auto_approve', 'commit')
               AND decision IN ('auto', 'approve', 'edit', 'escalate')
            """
        ) as cur:
            posted = dict(await cur.fetchone())
    return {"human": human, "posted": posted}


async def checkpoint_history(thread_id: str) -> list[dict[str, Any]]:
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        items: list[dict[str, Any]] = []
        async for snap in graph.aget_state_history(cfg):
            values = snap.values or {}
            analysis = values.get("analysis")
            confidence = getattr(analysis, "confidence", None)
            checkpoint_id = snap.config.get("configurable", {}).get("checkpoint_id")
            if not checkpoint_id:
                continue
            items.append({
                "checkpoint_id": checkpoint_id,
                "created_at": getattr(snap, "created_at", None),
                "next": ", ".join(snap.next or ()) or "done",
                "decision": values.get("decision") or "-",
                "confidence": confidence,
            })
    return items


async def run_graph(
    pr_url: str,
    thread_id: str,
    *,
    resume_value: Any | None = None,
    checkpoint_id: str | None = None,
) -> dict:
    """Invoke or resume the graph once."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        configurable = {"thread_id": thread_id}
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id
        cfg = {"configurable": configurable}

        if resume_value is None:
            return await graph.ainvoke(
                {
                    "pr_url": pr_url,
                    "thread_id": thread_id,
                    "require_final_approval": True,
                },
                cfg,
            )
        return await graph.ainvoke(Command(resume=resume_value), cfg)


def _run(coro):
    return asyncio.run(coro)


def render_approval_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Approval requested - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.warning("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` - {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_area("Feedback", key=f"approval_feedback_{st.session_state.thread_id}", height=90)
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Strong escalation - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    with st.form(f"escalation_{st.session_state.thread_id}"):
        answers = {
            question: st.text_area(question, key=f"q_{idx}_{st.session_state.thread_id}", height=90)
            for idx, question in enumerate(payload["questions"])
        }
        submitted = st.form_submit_button("Submit answers", type="primary")
        if submitted:
            return answers
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action", "?")
    comment_url = final.get("posted_comment_url")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"{action} - comment posted")
        if comment_url:
            st.link_button("View comment on GitHub", comment_url)
        else:
            st.link_button("Open PR", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("Rejected - no comment posted")
    else:
        st.info(f"final_action = {action}")

    if final.get("posted_comment_body"):
        with st.expander("Posted comment"):
            st.markdown(final["posted_comment_body"])
    st.caption(
        f"thread_id = {st.session_state.thread_id} | "
        f"replay: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Reviewer")
        reviewer = st.text_input("GitHub user", value=st.session_state.reviewer_id)
        st.session_state.reviewer_id = reviewer.strip()
        if st.session_state.reviewer_id:
            os.environ["GITHUB_USER"] = st.session_state.reviewer_id

        st.header("Recent sessions")
        try:
            rows = _run(list_recent_sessions())
        except Exception as exc:
            rows = []
            st.caption(f"Audit DB unavailable: {exc}")
        for row in rows:
            label = f"{row['worst_risk']} | {row['thread_id'][:8]} | {row['events']} events"
            if st.button(label, key=f"session_{row['thread_id']}", use_container_width=True):
                st.session_state.thread_id = row["thread_id"]
                st.session_state.pr_url = row["pr_url"]
                st.session_state.interrupt_payload = None
                st.session_state.final = None
                st.session_state.checkpoint_id = None
                st.rerun()
            st.caption(row["pr_url"])

        st.header("Calibration")
        try:
            summary = _run(calibration_summary())
            human = summary["human"]
            posted = summary["posted"]
            approvals = human.get("approvals") or 0
            reviewed = human.get("reviewed") or 0
            approval_rate = approvals / reviewed if reviewed else 0
            st.metric("Human approval rate", f"{approval_rate:.0%}", f"{reviewed} decisions")
            avg_conf = human.get("avg_confidence")
            st.metric("Avg HITL confidence", f"{avg_conf:.0%}" if avg_conf else "n/a")
            posted_conf = posted.get("avg_confidence")
            st.metric("Avg posted confidence", f"{posted_conf:.0%}" if posted_conf else "n/a")
        except Exception as exc:
            st.caption(f"Calibration unavailable: {exc}")

        if st.session_state.thread_id:
            st.header("Checkpoints")
            try:
                history = _run(checkpoint_history(st.session_state.thread_id))
            except Exception as exc:
                history = []
                st.caption(f"History unavailable: {exc}")
            if history:
                labels = {
                    f"{item['checkpoint_id'][:8]} | {item['next']} | {item['decision']}": item["checkpoint_id"]
                    for item in history
                }
                selected = st.selectbox("State history", options=list(labels.keys()))
                if st.button("Use checkpoint", use_container_width=True):
                    st.session_state.checkpoint_id = labels[selected]
                    st.toast("Checkpoint selected for the next resume.")


_init_state()
render_sidebar()

st.title("HITL PR Review Agent")

with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")

if submitted and pr_url:
    st.session_state.pr_url = pr_url.strip()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.checkpoint_id = None
    st.session_state.last_error = None

    try:
        with st.spinner("Reviewing PR..."):
            result = _run(run_graph(st.session_state.pr_url, st.session_state.thread_id))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.final = result
    except Exception as exc:
        st.session_state.last_error = str(exc)

if st.session_state.last_error:
    st.error(st.session_state.last_error)

payload = st.session_state.interrupt_payload
if payload is not None:
    answer = (
        render_approval_card(payload)
        if payload["kind"] == "approval_request"
        else render_escalation_card(payload)
    )
    if answer is not None:
        checkpoint_id = st.session_state.checkpoint_id
        st.session_state.checkpoint_id = None
        try:
            with st.spinner("Resuming review..."):
                result = _run(run_graph(
                    st.session_state.pr_url,
                    st.session_state.thread_id,
                    resume_value=answer,
                    checkpoint_id=checkpoint_id,
                ))
            if "__interrupt__" in result:
                st.session_state.interrupt_payload = result["__interrupt__"][0].value
            else:
                st.session_state.interrupt_payload = None
                st.session_state.final = result
            st.rerun()
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.rerun()

if st.session_state.final is not None:
    render_final(st.session_state.final)
