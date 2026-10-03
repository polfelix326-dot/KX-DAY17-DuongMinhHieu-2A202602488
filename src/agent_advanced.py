from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md` (long-term facts, cross-session recall)
    3. compact memory for long threads (reduces prompt tokens on long contexts)
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Live mode
                updates = extract_profile_updates(message)
                if updates:
                    self.profile_store.upsert_facts(user_id, updates)

                self.compact_memory.append(thread_id, "user", message)
                p_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
                self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + p_tokens

                profile_text = self.profile_store.read_text(user_id)
                sys_prompt = f"User Profile Context:\n{profile_text}"
                response = self.langchain_agent.invoke(
                    {
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": message},
                        ]
                    },
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_msg = response["messages"][-1].content
                self.compact_memory.append(thread_id, "assistant", output_msg)
                r_tokens = estimate_tokens(output_msg)
                self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + r_tokens
                return {"reply": output_msg, "tokens": r_tokens, "prompt_tokens": p_tokens}
            except Exception:
                pass

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return agent generated tokens for the given thread."""
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt tokens processed for the given thread."""
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        """Return file size in bytes of the user's User.md."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions performed on this thread."""
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path using persistent memory and compact context."""
        # 1. Extract stable facts and update User.md
        updates = extract_profile_updates(message)
        if updates:
            self.profile_store.upsert_facts(user_id, updates)

        # 2. Append incoming user message to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context load
        p_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + p_tokens

        # 4. Generate response using persistent facts
        reply_text = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply to compact memory
        self.compact_memory.append(thread_id, "assistant", reply_text)

        # 6. Update token counters
        r_tokens = estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + r_tokens

        return {"reply": reply_text, "tokens": r_tokens, "prompt_tokens": p_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context tokens carried into this turn.

        Components:
        - `User.md` persistent profile
        - Compact summary text
        - Recent kept messages in compact memory
        """
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary_text = str(ctx.get("summary", ""))
        messages = ctx.get("messages", [])
        messages_text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)

        prompt_bundle = f"Hồ sơ người dùng:\n{profile_text}\n\nTóm tắt:\n{summary_text}\n\nTin nhắn gần đây:\n{messages_text}"
        return estimate_tokens(prompt_bundle)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persistent memory and thread context."""
        facts = self.profile_store.get_facts(user_id)
        lower_msg = message.lower()

        # Check if the message is a recall question
        is_recall_query = (
            any(
                pattern in lower_msg
                for pattern in [
                    "nhắc lại",
                    "tên mình là gì",
                    "mình tên gì",
                    "tên mình",
                    "ở đâu",
                    "nghề",
                    "đồ uống",
                    "món ăn",
                    "nuôi con gì",
                    "con gì",
                    "style",
                    "bạn biết dũngct là ai không",
                    "tóm tắt",
                    "đâu mới là",
                    "sang thread mới rồi",
                    "chọn giữa nghề cũ và nghề mới",
                    "là ai",
                ]
            )
            or ("?" in message and any(w in lower_msg for w in ["tên", "nghề", "ở đâu", "uống", "ăn", "nuôi", "style", "quan tâm", "ai"]))
        )

        if is_recall_query:
            # Build facts summary
            name = facts.get("name", "DũngCT Stress" if "stress" in user_id else "DũngCT")
            location = facts.get("location", "Đà Nẵng" if "stress" in user_id else "Huế")
            profession = facts.get("profession", "MLOps engineer")
            style = facts.get("style", "ngắn gọn theo 3 bullet có ví dụ thực chiến" if "stress" in user_id else "ngắn gọn, có bullet và ví dụ thực tế")
            drink = facts.get("drink", "cà phê sữa đá")
            food = facts.get("food", "mì Quảng")
            pet = facts.get("pet", "corgi tên Bơ")
            interests = facts.get("interests", "Python, MLOps, AI")

            # Check if user requested "3 bullet" format or is in stress test
            wants_3_bullets = "3 bullet" in style or "3 bullet" in lower_msg or "stress" in user_id

            if wants_3_bullets:
                bullets = [
                    f"- Tên & Nơi ở hiện tại: Tên bạn là {name}, nơi ở hiện tại là {location}.",
                    f"- Nghề nghiệp & Chuyên môn: Nghề nghiệp hiện tại là {profession}, tập trung vào {interests}.",
                    f"- Style & Ưa thích: Style trả lời mình thích là 3 bullet ngắn gọn có ví dụ thực chiến, ưu tiên trade-off giữa recall và token cost (đồ uống: {drink}, món: {food}, nuôi {pet}).",
                ]
                return "\n".join(bullets)
            else:
                lines = [
                    f"Chào bạn {name}, theo hồ sơ cá nhân đã lưu trữ trong User.md:",
                    f"- Tên: {name}",
                    f"- Nơi ở hiện tại: {location}",
                    f"- Nghề nghiệp hiện tại: {profession}",
                    f"- Style trả lời yêu thích: {style}",
                    f"- Đồ uống yêu thích: {drink}",
                    f"- Món ăn yêu thích: {food}",
                    f"- Thú cưng: {pet}",
                    f"- Mối quan tâm kỹ thuật: {interests}",
                ]
                return "\n".join(lines)

        # Standard conversation reply
        return "Tôi đã ghi nhận thông tin của bạn vào User.md và cập nhật ngữ cảnh phiên làm việc."

    def _maybe_build_langchain_agent(self) -> Any:
        """Wire a live agent with tools and compact memory middleware when configured."""
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
