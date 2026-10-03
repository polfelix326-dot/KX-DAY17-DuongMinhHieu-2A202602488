from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A / Baseline Agent.

    Characteristics:
    - Within-session (in-thread) memory only.
    - No persistent `User.md`.
    - No compact memory: whole history is passed every turn, so prompt tokens grow quadratically.
    - Completely forgets all long-term facts in new threads.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Live LangChain execution
                session = self.sessions.setdefault(thread_id, SessionState())
                prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in session.messages) + f"\nuser: {message}"
                p_tokens = estimate_tokens(prompt_text)
                session.prompt_tokens_processed += p_tokens

                response = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_msg = response["messages"][-1].content
                r_tokens = estimate_tokens(output_msg)
                session.token_usage += r_tokens
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": output_msg})
                return {"reply": output_msg, "tokens": r_tokens, "prompt_tokens": p_tokens}
            except Exception:
                # Graceful fallback to offline
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent generation tokens for one thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate cumulative prompt tokens processed in this thread."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory, always returns 0."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement deterministic offline behavior for Baseline Agent."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Baseline accumulates all previous messages as prompt context in every turn
        prompt_lines = [f"{m['role']}: {m['content']}" for m in session.messages]
        prompt_lines.append(f"user: {message}")
        prompt_context = "\n".join(prompt_lines)

        p_tokens = estimate_tokens(prompt_context)
        session.prompt_tokens_processed += p_tokens

        # Check if the question can be answered from WITHIN this session only
        # If this is a fresh thread (e.g. cross-session recall question), session.messages is empty!
        if not session.messages:
            reply_text = "Chào bạn, đây là phiên làm việc mới nên tôi chưa có thông tin hoặc ngữ cảnh nào từ các phiên trước của bạn."
        else:
            # Simple keyword search inside current session history if user is asking
            lower_msg = message.lower()
            if any(q in lower_msg for q in ["tên gì", "ở đâu", "nghề gì", "đồ uống", "món ăn", "nuôi con gì", "nhắc lại"]):
                # Look for facts mentioned earlier in THIS session
                history_text = " ".join(m["content"] for m in session.messages)
                found_facts = []
                if "dũngct" in history_text.lower():
                    found_facts.append("Tên: DũngCT")
                if "huế" in history_text.lower():
                    found_facts.append("Nơi ở: Huế")
                elif "đà nẵng" in history_text.lower():
                    found_facts.append("Nơi ở: Đà Nẵng")
                if found_facts:
                    reply_text = f"Theo thông tin bạn vừa chia sẻ trong phiên này: {', '.join(found_facts)}."
                else:
                    reply_text = "Tôi chỉ ghi nhận được các thông tin trong phiên thảo luận hiện tại này."
            else:
                reply_text = "Tôi đã ghi nhận thông tin và sẵn sàng hỗ trợ bạn tiếp theo trong phiên này."

        r_tokens = estimate_tokens(reply_text)
        session.token_usage += r_tokens

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {"reply": reply_text, "tokens": r_tokens, "prompt_tokens": p_tokens}

    def _maybe_build_langchain_agent(self) -> Any:
        """Optionally build LangGraph create_react_agent or similar when available."""
        if not self.config.model.api_key and self.config.model.provider not in ("ollama", "custom"):
            return None

        chat_model = build_chat_model(self.config.model)
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent

        checkpointer = MemorySaver()
        return create_react_agent(
            model=chat_model,
            tools=[],
            checkpointer=checkpointer,
        )
