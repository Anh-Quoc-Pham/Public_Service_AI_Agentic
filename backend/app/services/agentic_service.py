import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


class AgenticService:
    """Planner-executor orchestration service for agentic query handling."""

    def __init__(self, rag_service: RAGService, llm_service: LLMService):
        self.rag_service = rag_service
        self.llm_service = llm_service
        self.is_initialized = False
        self.sessions: Dict[str, Dict[str, Any]] = {}

        env_max_steps = self._safe_int(os.getenv("AGENT_MAX_STEPS", "4"), default=4)
        self.default_max_steps = max(1, min(8, env_max_steps))
        self.trace_enabled = os.getenv("AGENT_TRACE_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    async def initialize(self):
        """Initialize agentic orchestration state."""
        self.is_initialized = True
        logger.info(
            "Agentic service initialized (max_steps=%s, trace_enabled=%s)",
            self.default_max_steps,
            self.trace_enabled,
        )

    async def process_query(
        self,
        query: str,
        user_context: Optional[Any] = None,
        session_id: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a plan-execute-answer loop and return grounded response payload."""
        if not self.is_initialized:
            raise RuntimeError("Agentic service not initialized")

        steps_limit = self._sanitize_max_steps(max_steps)
        normalized_context = self._normalize_context(user_context)

        session_id, session = self._get_or_create_session(session_id)
        self._append_history(session, "user", query)

        trace: List[Dict[str, Any]] = []
        plan = await self._build_plan(query, normalized_context, session, steps_limit)

        if self.trace_enabled:
            trace.append(
                {
                    "step": 1,
                    "action": "plan",
                    "status": "completed",
                    "detail": plan,
                }
            )

        if plan.get("needs_clarification"):
            clarification = plan.get("clarification_question") or (
                "Can you share a bit more detail so I can guide you correctly?"
            )
            self._append_history(session, "assistant", clarification)
            return {
                "response": clarification,
                "sources": [],
                "confidence": 0.6,
                "session_id": session_id,
                "trace": trace if self.trace_enabled else None,
            }

        gathered_docs: List[Dict[str, Any]] = []
        executed_steps = 1
        for idx, step in enumerate(plan.get("steps", []), start=1):
            tool_name = step.get("tool", "retrieve_documents")
            tool_input = step.get("input") or query
            reason = step.get("reason", "")
            executed_steps += 1

            if tool_name == "retrieve_documents":
                docs = await self.rag_service.retrieve_documents(tool_input, k=5)
                if docs:
                    gathered_docs = docs

                if self.trace_enabled:
                    trace.append(
                        {
                            "step": executed_steps,
                            "action": "retrieve_documents",
                            "status": "completed",
                            "detail": {
                                "query": tool_input,
                                "reason": reason,
                                "documents_found": len(docs),
                            },
                        }
                    )

            elif tool_name == "final_answer":
                if self.trace_enabled:
                    trace.append(
                        {
                            "step": executed_steps,
                            "action": "final_answer",
                            "status": "ready",
                            "detail": {
                                "reason": reason,
                            },
                        }
                    )
                break

            else:
                if self.trace_enabled:
                    trace.append(
                        {
                            "step": executed_steps,
                            "action": tool_name,
                            "status": "skipped",
                            "detail": {
                                "reason": "Unsupported tool, skipped.",
                            },
                        }
                    )

        if not gathered_docs:
            gathered_docs = await self.rag_service.retrieve_documents(query, k=5)
            if self.trace_enabled:
                trace.append(
                    {
                        "step": executed_steps + 1,
                        "action": "retrieve_documents",
                        "status": "completed",
                        "detail": {
                            "query": query,
                            "reason": "Fallback retrieval because no documents were gathered.",
                            "documents_found": len(gathered_docs),
                        },
                    }
                )

        response_text = await self._generate_grounded_answer(
            query=query,
            context_docs=gathered_docs,
            normalized_context=normalized_context,
            session=session,
        )

        self._append_history(session, "assistant", response_text)
        session["last_sources"] = gathered_docs[:5]

        return {
            "response": response_text,
            "sources": gathered_docs,
            "confidence": self._estimate_confidence(gathered_docs),
            "session_id": session_id,
            "trace": trace if self.trace_enabled else None,
        }

    def get_session_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return a lightweight snapshot of a stored session."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session_id,
            "turn_count": len(session.get("history", [])),
            "history": session.get("history", [])[-10:],
            "last_sources": session.get("last_sources", []),
        }

    def clear_session(self, session_id: str) -> bool:
        """Delete a stored session."""
        return self.sessions.pop(session_id, None) is not None

    async def _build_plan(
        self,
        query: str,
        normalized_context: Dict[str, Any],
        session: Dict[str, Any],
        max_steps: int,
    ) -> Dict[str, Any]:
        """Ask the model for a compact execution plan in JSON format."""
        default_plan = {
            "goal": "Provide a grounded and actionable answer for public service navigation.",
            "needs_clarification": False,
            "clarification_question": "",
            "steps": [
                {
                    "tool": "retrieve_documents",
                    "input": query,
                    "reason": "Collect relevant policy and program context.",
                },
                {
                    "tool": "final_answer",
                    "input": "",
                    "reason": "Produce a concise answer with practical next steps.",
                },
            ],
        }

        history_text = self._format_history(session.get("history", []))
        context_text = json.dumps(normalized_context, ensure_ascii=True)[:1200]

        planner_prompt = f"""Create a JSON plan for the question below.

Question: {query}
Session history: {history_text}
External context: {context_text}

Available tools:
- retrieve_documents: semantic retrieval from public-service knowledge base.
- final_answer: write the final answer from gathered evidence.

Return JSON only with this schema:
{{
  "goal": "string",
  "needs_clarification": boolean,
  "clarification_question": "string",
  "steps": [
    {{"tool": "retrieve_documents|final_answer", "input": "string", "reason": "string"}}
  ]
}}

Rules:
- At most {max_steps} steps.
- If the user request is too ambiguous, set needs_clarification=true.
- Ensure the final step is final_answer.
"""

        try:
            raw_plan = await self.llm_service.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a planning module that returns strict JSON only.",
                    },
                    {"role": "user", "content": planner_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            parsed_plan = self._extract_json(raw_plan)
            return self._sanitize_plan(parsed_plan, query, max_steps)
        except Exception as exc:
            logger.warning("Planner failed, using default plan: %s", str(exc))
            return default_plan

    async def _generate_grounded_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        normalized_context: Dict[str, Any],
        session: Dict[str, Any],
    ) -> str:
        """Generate final answer using retrieved context and session memory."""
        context_blocks: List[str] = []
        for idx, doc in enumerate(context_docs[:4], start=1):
            source = doc.get("metadata", {}).get("source", "knowledge_base")
            excerpt = doc.get("content", "")[:700]
            context_blocks.append(f"[{idx}] Source: {source}\n{excerpt}")

        joined_context = "\n\n".join(context_blocks) if context_blocks else "No retrieved documents available."
        history_text = self._format_history(session.get("history", []), max_items=8)
        context_text = json.dumps(normalized_context, ensure_ascii=True)[:1200]

        answer_prompt = f"""User question: {query}

Conversation memory:
{history_text}

Structured user context:
{context_text}

Retrieved evidence:
{joined_context}

Write a helpful response that:
1) answers directly,
2) provides step-by-step actions,
3) highlights any eligibility caveats,
4) mentions uncertainty when evidence is missing.
"""

        try:
            response = await self.llm_service.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a public service navigation assistant. "
                            "Use grounded evidence and avoid fabricating program rules."
                        ),
                    },
                    {"role": "user", "content": answer_prompt},
                ],
                temperature=0.3,
                max_tokens=700,
            )

            if response and response.strip():
                return response.strip()

            # Fallback to classic generation path.
            return await self.llm_service.generate_response(
                query=query,
                context_docs=context_docs,
                user_context=normalized_context,
            )
        except Exception as exc:
            logger.warning("Grounded answer generation failed, using classic fallback: %s", str(exc))
            return await self.llm_service.generate_response(
                query=query,
                context_docs=context_docs,
                user_context=normalized_context,
            )

    def _sanitize_plan(self, plan: Dict[str, Any], query: str, max_steps: int) -> Dict[str, Any]:
        """Validate and normalize planner output to a safe execution format."""
        allowed_tools = {"retrieve_documents", "final_answer"}

        if not isinstance(plan, dict):
            plan = {}

        steps: List[Dict[str, str]] = []
        for raw_step in plan.get("steps", []):
            if not isinstance(raw_step, dict):
                continue

            tool = str(raw_step.get("tool", "retrieve_documents")).strip()
            if tool not in allowed_tools:
                continue

            steps.append(
                {
                    "tool": tool,
                    "input": str(raw_step.get("input", query)).strip(),
                    "reason": str(raw_step.get("reason", "")).strip(),
                }
            )

            if len(steps) >= max_steps:
                break

        if not steps:
            steps = [
                {
                    "tool": "retrieve_documents",
                    "input": query,
                    "reason": "Default retrieval step.",
                },
                {
                    "tool": "final_answer",
                    "input": "",
                    "reason": "Default response step.",
                },
            ]

        if steps[-1]["tool"] != "final_answer":
            if len(steps) >= max_steps:
                steps[-1] = {
                    "tool": "final_answer",
                    "input": "",
                    "reason": "Ensure final answer step exists.",
                }
            else:
                steps.append(
                    {
                        "tool": "final_answer",
                        "input": "",
                        "reason": "Conclude after retrieval.",
                    }
                )

        return {
            "goal": str(
                plan.get(
                    "goal",
                    "Provide a grounded response for the user request.",
                )
            ),
            "needs_clarification": bool(plan.get("needs_clarification", False)),
            "clarification_question": str(plan.get("clarification_question", "")).strip(),
            "steps": steps,
        }

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract and parse a JSON object from free-form model output."""
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        candidate_match = re.search(r"\{[\s\S]*\}", text)
        if not candidate_match:
            return {}

        candidate = candidate_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}

    def _sanitize_max_steps(self, max_steps: Optional[int]) -> int:
        """Clamp max_steps within allowed boundaries."""
        if max_steps is None:
            return self.default_max_steps
        return max(1, min(8, int(max_steps)))

    def _get_or_create_session(self, session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Get an existing session or create a new one."""
        if not session_id:
            session_id = uuid.uuid4().hex[:12]

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "last_sources": [],
            }

        return session_id, self.sessions[session_id]

    @staticmethod
    def _normalize_context(user_context: Optional[Any]) -> Dict[str, Any]:
        """Normalize user context into a predictable dictionary payload."""
        if user_context is None:
            return {}

        if isinstance(user_context, dict):
            return dict(user_context)

        if isinstance(user_context, list):
            return {
                "conversation_history": [
                    item
                    for item in user_context
                    if isinstance(item, dict) and item.get("content")
                ][-12:]
            }

        return {"raw_context": str(user_context)}

    @staticmethod
    def _format_history(history: List[Dict[str, str]], max_items: int = 6) -> str:
        """Convert history items to a compact prompt-friendly string."""
        if not history:
            return "No prior turns."

        lines = []
        for item in history[-max_items:]:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            if content:
                lines.append(f"{role}: {content[:250]}")

        return "\n".join(lines) if lines else "No prior turns."

    @staticmethod
    def _append_history(session: Dict[str, Any], role: str, content: str):
        """Append conversation turn and keep a bounded history."""
        session.setdefault("history", []).append({"role": role, "content": content})
        session["history"] = session["history"][-24:]

    @staticmethod
    def _estimate_confidence(context_docs: List[Dict[str, Any]]) -> float:
        """Estimate confidence score from document retrieval relevance."""
        if not context_docs:
            return 0.45

        relevances: List[float] = []
        for doc in context_docs[:4]:
            relevance = doc.get("relevance")
            if isinstance(relevance, (int, float)):
                relevances.append(max(0.0, min(1.0, float(relevance))))

        if not relevances:
            return 0.72

        average_relevance = sum(relevances) / len(relevances)
        return max(0.4, min(0.98, average_relevance))

    @staticmethod
    def _safe_int(value: Optional[str], default: int) -> int:
        """Convert optional string to int with fallback."""
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
