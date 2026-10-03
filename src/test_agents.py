from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests with reduced compact threshold."""
    repo_root = Path(__file__).resolve().parent.parent
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=repo_root,
        data_dir=repo_root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, read, updated, and edited."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(profiles_dir)
    user_id = "test_user_01"

    # Initial write
    initial_text = "# Profile\n- Name: DũngCT\n- Location: Da Nang"
    file_path = store.write_text(user_id, initial_text)
    assert file_path.exists()
    assert store.file_size(user_id) > 0

    # Read back
    content = store.read_text(user_id)
    assert content == initial_text

    # Edit existing text
    edited = store.edit_text(user_id, "Da Nang", "Hue")
    assert edited is True
    updated_content = store.read_text(user_id)
    assert "- Location: Hue" in updated_content
    assert "Da Nang" not in updated_content

    # Edit non-existent text
    failed_edit = store.edit_text(user_id, "NonExistentLocation", "Saigon")
    assert failed_edit is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction when exceeding threshold."""
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "test_thread_compact"

    # Message 1
    manager.append(thread_id, "user", "Xin chào agent, đây là thông điệp mở đầu cho bài kiểm tra nén bộ nhớ.")
    # Message 2
    manager.append(thread_id, "assistant", "Chào bạn, tôi đã ghi nhận thông điệp và sẵn sàng hỗ trợ bạn tiếp theo.")
    # Message 3
    manager.append(thread_id, "user", "Hãy tiếp tục theo dõi các thay đổi hệ thống và tổng hợp tiến độ công việc hàng ngày.")
    # Message 4 (exceeds threshold)
    manager.append(thread_id, "assistant", "Tôi đang theo dõi tiến độ, hệ thống hoạt động ổn định và log đã được cập nhật đầy đủ.")

    ctx = manager.context(thread_id)
    assert manager.compaction_count(thread_id) >= 1
    assert len(ctx["messages"]) <= 2
    assert len(ctx["summary"]) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    user_id = "dungct_recall_test"

    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    # Session 1: establish fact
    baseline.reply(user_id, "thread_session_1", "Chào bạn, mình tên là DũngCT và hiện tại mình đang ở Huế.")
    advanced.reply(user_id, "thread_session_1", "Chào bạn, mình tên là DũngCT và hiện tại mình đang ở Huế.")

    # Session 2: ask in a brand new thread
    base_res = baseline.reply(user_id, "thread_session_2", "Hiện tại mình đang ở đâu?")
    adv_res = advanced.reply(user_id, "thread_session_2", "Hiện tại mình đang ở đâu?")

    # Baseline should NOT know the location in a new thread
    assert "huế" not in base_res["reply"].lower()

    # Advanced should recall the location from persistent User.md
    assert "huế" in adv_res["reply"].lower()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    # Configure tight threshold to force compaction during the thread
    cfg.compact_threshold_tokens = 60
    cfg.compact_keep_messages = 2

    user_id = "user_stress_test"
    thread_id = "stress_thread"

    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    long_turns = [
        "Nội dung số 1: Giới thiệu hệ thống phân tán và kiến trúc microservices trong xử lý dữ liệu lớn.",
        "Nội dung số 2: Phân tích cơ chế đồng thuận Raft và Paxos trong việc lưu trữ trạng thái có tính nhất quán cao.",
        "Nội dung số 3: Tối ưu hóa pipeline CI/CD với containerization và orchestration bằng Kubernetes.",
        "Nội dung số 4: Thiết kế hệ thống logging tập trung với Elasticsearch, Logstash và Kibana.",
        "Nội dung số 5: Đo lường độ trễ mạng và throughput giữa các cụm server đa vùng địa lý.",
        "Nội dung số 6: Tích hợp mô hình ngôn ngữ lớn vào backend với kỹ thuật Retrieval-Augmented Generation.",
        "Nội dung số 7: Đánh giá trade-off giữa chi phí token và độ chính xác của câu trả lời từ LLM.",
        "Nội dung số 8: Xây dựng cơ chế cache phân tầng sử dụng Redis và in-memory data structures.",
        "Nội dung số 9: Quản lý vòng đời bộ nhớ agent thông qua kỹ thuật nén tóm tắt ngữ cảnh cũ.",
        "Nội dung số 10: Tổng hợp tất cả các kiến thức trên thành tài liệu kỹ thuật hoàn chỉnh cho đội ngũ.",
    ]

    for turn in long_turns:
        baseline.reply(user_id, thread_id, turn)
        advanced.reply(user_id, thread_id, turn)

    base_prompt_tokens = baseline.prompt_token_usage(thread_id)
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)

    # Advanced agent compaction must have triggered
    assert advanced.compaction_count(thread_id) > 0
    assert baseline.compaction_count(thread_id) == 0

    # Advanced agent should have processed fewer prompt tokens due to compaction
    assert adv_prompt_tokens < base_prompt_tokens
