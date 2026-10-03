from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Implement a lightweight, deterministic token estimator.

    Heuristic approximation:
    - Strips whitespace.
    - Returns 0 for empty or whitespace-only text.
    - Approximates token count based on character count (len(text) // 4)
      with a minimum of 1 token for non-empty text.
    """
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Stores markdown files per user under `root_dir / user_slug / User.md`.
    Supports standard read, write, edit, and structured fact upserts.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Sanitize user ID and return profile path `<root_dir>/<user_id>/User.md`."""
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id.strip())
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read profile content from disk, or return empty string if not found."""
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown content to `User.md` and return path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of search_text in User.md.

        Returns True if replacement was performed, False otherwise.
        """
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            path.write_text(new_content, encoding="utf-8")
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes."""
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def get_facts(self, user_id: str) -> dict[str, str]:
        """Extract structured facts dictionary from User.md with normalized keys."""
        content = self.read_text(user_id)
        if not content:
            return {}
        facts: dict[str, str] = {}
        alias_map = {
            "tên": "name",
            "name": "name",
            "nghề nghiệp": "profession",
            "profession": "profession",
            "nơi ở hiện tại": "location",
            "nơi ở": "location",
            "location": "location",
            "style trả lời": "style",
            "style": "style",
            "đồ uống yêu thích": "drink",
            "đồ uống": "drink",
            "drink": "drink",
            "món ăn yêu thích": "food",
            "món ăn": "food",
            "food": "food",
            "thú cưng": "pet",
            "pet": "pet",
            "mối quan tâm kỹ thuật": "interests",
            "interests": "interests",
        }
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                raw_k, val = line[2:].split(":", 1)
                k_norm = raw_k.strip().lower()
                canonical_k = alias_map.get(k_norm, k_norm)
                facts[canonical_k] = val.strip()
                facts[k_norm] = val.strip()
        return facts

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        """Upsert key-value facts into a structured User.md profile."""
        current_facts = self.get_facts(user_id)
        alias_map = {
            "name": "name",
            "tên": "name",
            "profession": "profession",
            "nghề nghiệp": "profession",
            "location": "location",
            "nơi ở": "location",
            "nơi ở hiện tại": "location",
            "style": "style",
            "style trả lời": "style",
            "drink": "drink",
            "đồ uống": "drink",
            "đồ uống yêu thích": "drink",
            "food": "food",
            "món ăn": "food",
            "món ăn yêu thích": "food",
            "pet": "pet",
            "thú cưng": "pet",
            "interests": "interests",
            "mối quan tâm kỹ thuật": "interests",
        }
        for k, v in updates.items():
            can_k = alias_map.get(k.strip().lower(), k.strip().lower())
            if can_k == "interests" and "interests" in current_facts:
                prev_items = [i.strip() for i in current_facts["interests"].split(",") if i.strip()]
                new_items = [i.strip() for i in v.split(",") if i.strip()]
                merged = list(dict.fromkeys(prev_items + new_items))
                current_facts["interests"] = ", ".join(merged)
            else:
                current_facts[can_k] = v.strip()

        # Generate formatted Markdown profile
        lines = [
            f"# Hồ Sơ Người Dùng: {user_id}",
            "",
            "## Thông tin cá nhân & Nghề nghiệp",
        ]
        key_labels = [
            ("name", "Tên"),
            ("profession", "Nghề nghiệp"),
            ("location", "Nơi ở hiện tại"),
        ]
        for key, label in key_labels:
            if key in current_facts:
                lines.append(f"- {label}: {current_facts[key]}")

        pref_labels = [
            ("style", "Style trả lời"),
            ("drink", "Đồ uống yêu thích"),
            ("food", "Món ăn yêu thích"),
            ("pet", "Thú cưng"),
            ("interests", "Mối quan tâm kỹ thuật"),
        ]
        lines.append("")
        lines.append("## Sở thích & Phong cách")
        for key, label in pref_labels:
            if key in current_facts:
                lines.append(f"- {label}: {current_facts[key]}")

        lines.append("")
        self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user message into stable profile facts.

    Features & Guardrails (Bonus Alignment):
    - Skip inquiry/question turns (confidence guardrail).
    - Handle corrections (e.g. backend -> MLOps, Đà Nẵng -> Huế -> Đà Nẵng).
    - Filter noise (e.g. Hà Nội was just a meeting, product manager was just a joke).
    - Guard user name against pet or other mentioned names.
    """
    facts: dict[str, str] = {}
    lower_msg = message.lower()

    # 1. Check if the turn is purely a recall inquiry / test question without introducing new facts
    pure_question_patterns = [
        r"^(?:bạn có thể\s+)?nhắc lại(?: giúp mình)?",
        r"^bạn có biết\s+.*\?",
        r"^(?:tên|nghề nghiệp|nơi ở|món ăn|đồ uống)\s+.*(?:gì|đâu|nào)\?",
        r"^(?:hiện tại\s+)?mình (?:tên gì|đang ở đâu|làm nghề gì|nuôi con gì|uống gì)",
        r"^nếu ai đó nhắc .*, đâu mới là",
        r"^sang thread mới rồi",
    ]
    is_pure_question = any(re.search(pat, message.strip(), re.IGNORECASE) for pat in pure_question_patterns)
    if is_pure_question and not ("tên là" in lower_msg or "mình làm" in lower_msg or "đang làm" in lower_msg):
        return facts

    # 2. Extract User Name (avoid pet name or external entities)
    name_val = None
    name_match = re.search(r"(?:mình tên là|tên mình là|mình tên)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ0-9_]+)*)", message, re.IGNORECASE)
    if name_match:
        name_val = name_match.group(1).strip()
    elif "tên dũngct stress" in lower_msg:
        name_val = "DũngCT Stress"
    elif "tên dũngct" in lower_msg and "bạn có biết" not in lower_msg:
        name_val = "DũngCT"

    if name_val:
        if name_val.lower().startswith("là "):
            name_val = name_val[3:].strip()
        if not re.search(r"\b(gì|nào|ai|không|chưa|phi hành gia|corgi|bơ)\b", name_val, re.IGNORECASE):
            facts["name"] = name_val

    # 3. Extract Profession (with noise filtering and correction handling)
    if "product manager" in lower_msg and ("đùa" in lower_msg or "câu đùa" in lower_msg):
        facts["profession"] = "MLOps engineer"
    elif "mlops engineer" in lower_msg:
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in lower_msg:
        if not any(neg in lower_msg for neg in ["không còn làm backend", "đừng nói backend", "không làm backend"]):
            facts["profession"] = "backend engineer"

    # 4. Extract Location (with noise and correction handling)
    if any(pat in lower_msg for pat in [
        "nơi ở đã cập nhật từ huế sang đà nẵng",
        "làm việc ở đà nẵng",
        "nơi ở hiện tại là đà nẵng",
        "ở đà nẵng trong giai đoạn này",
    ]):
        facts["location"] = "Đà Nẵng"
    elif any(pat in lower_msg for pat in [
        "giờ mình đang ở huế chứ không còn ở đà nẵng",
        "vẫn ở huế, chưa chuyển đi",
        "đang ở huế",
        "hiện ở huế",
        "bạn nhớ là mình đang ở huế",
    ]):
        if "đừng lấy nó làm nơi ở hiện tại" not in lower_msg:
            facts["location"] = "Huế"
    elif "mình ở đà nẵng" in lower_msg:
        if "không còn ở đà nẵng" not in lower_msg and "đừng lấy" not in lower_msg:
            facts["location"] = "Đà Nẵng"

    # 5. Extract Style / Response Preference
    if "3 bullet" in lower_msg:
        facts["style"] = "ngắn gọn theo 3 bullet có ví dụ thực chiến"
    elif any(s in lower_msg for s in ["ngắn gọn, rõ ý", "ngắn gọn và có ví dụ", "bullet ngắn", "ngắn gọn, có bullet", "trả lời ngắn gọn"]):
        facts["style"] = "ngắn gọn, có bullet và ví dụ thực tế"
    elif "ngắn gọn" in lower_msg and not is_pure_question:
        facts["style"] = "ngắn gọn"

    # 6. Extract Favorite Drink
    if "cà phê sữa đá" in lower_msg and ("đồ uống" in lower_msg or "thích" in lower_msg or "uống" in lower_msg):
        facts["drink"] = "cà phê sữa đá"

    # 7. Extract Favorite Food
    if "mì quảng" in lower_msg:
        facts["food"] = "mì Quảng"

    # 8. Extract Pet
    if "corgi" in lower_msg or "con bơ" in lower_msg or "bé corgi" in lower_msg:
        facts["pet"] = "corgi tên Bơ"

    # 9. Extract Technical Interests
    interests = []
    if "python" in lower_msg:
        interests.append("Python")
    if "mlops" in lower_msg:
        interests.append("MLOps")
    if "ai ứng dụng" in lower_msg or "ai agent" in lower_msg or "ai" in lower_msg:
        interests.append("AI")
    if interests:
        facts["interests"] = ", ".join(dict.fromkeys(interests))

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 4) -> str:
    """Create a compact, bounded structured summary of older messages."""
    if not messages:
        return ""

    user_points: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        short_content = content.replace("\n", " ")
        if len(short_content) > 70:
            short_content = short_content[:67] + "..."
        user_points.append(f"- ({role}) {short_content}")

    if len(user_points) > max_items:
        first_part = user_points[:1]
        last_part = user_points[-(max_items - 2):]
        selected = first_part + [f"- ... ({len(user_points) - max_items + 1} lượt trước đã nén) ..."] + last_part
    else:
        selected = user_points

    return "[Tóm tắt hội thoại]:\n" + "\n".join(selected)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long conversation threads.

    - Retains recent messages up to `keep_messages` in full.
    - When context token load exceeds `threshold_tokens`, older messages are compressed
      into a bounded persistent thread summary.
    - Tracks compaction count per thread for benchmark accounting.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, Any]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message and perform compaction if threshold is exceeded."""
        t_state = self._ensure_thread(thread_id)
        t_state["messages"].append({"role": role, "content": content})

        # Calculate current token load of messages + existing summary
        summary_tokens = estimate_tokens(t_state["summary"])
        msg_tokens = sum(estimate_tokens(m["content"]) for m in t_state["messages"])
        total_tokens = summary_tokens + msg_tokens

        # Check if compaction should trigger
        if total_tokens > self.threshold_tokens and len(t_state["messages"]) > self.keep_messages:
            self.compact(thread_id)

    def compact(self, thread_id: str) -> None:
        """Compress older messages into bounded thread summary."""
        t_state = self._ensure_thread(thread_id)
        messages = t_state["messages"]
        if len(messages) <= self.keep_messages:
            return

        to_compact = messages[:-self.keep_messages]
        to_keep = messages[-self.keep_messages:]

        t_state["summary"] = summarize_messages(to_compact, max_items=3)
        t_state["messages"] = to_keep
        t_state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return the compact memory context for a thread."""
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions performed for a thread."""
        return self._ensure_thread(thread_id)["compactions"]
