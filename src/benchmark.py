from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from tabulate import tabulate


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Calculate recall fraction (0.0 to 1.0) of expected key phrases present in answer."""
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in answer_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline benchmark.

    Combines:
    - 70% recall score
    - 30% formatting & conciseness (structured bullets, appropriate length)
    """
    recall = recall_points(answer, expected)
    format_score = 0.0

    # Reward structured response (bullets or concise lines)
    if "- " in answer or "• " in answer or "\n" in answer:
        format_score += 0.5
    # Reward non-empty, reasonable length (not overly bloated, not 1-word)
    if 20 <= len(answer) <= 800:
        format_score += 0.5

    return round(0.7 * recall + 0.3 * format_score, 3)


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over conversations and recall questions."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    user_ids: set[str] = set()
    initial_memory_sizes: dict[str, int] = {}

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        user_ids.add(user_id)

        if hasattr(agent, "memory_file_size") and user_id not in initial_memory_sizes:
            initial_memory_sizes[user_id] = agent.memory_file_size(user_id)

        # 1. Feed conversation turns in the conversation's thread
        for turn in conv.get("turns", []):
            res = agent.reply(user_id=user_id, thread_id=conv_id, message=turn)
            total_agent_tokens += res.get("tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)

        # 2. Ask recall questions in a FRESH thread (cross-session test)
        for q_idx, q_item in enumerate(conv.get("recall_questions", [])):
            fresh_thread_id = f"{conv_id}_recall_{q_idx}"
            question_text = q_item["question"]
            expected = q_item.get("expected_contains", [])

            q_res = agent.reply(user_id=user_id, thread_id=fresh_thread_id, message=question_text)
            total_agent_tokens += q_res.get("tokens", 0)
            total_prompt_tokens += q_res.get("prompt_tokens", 0)

            ans_text = q_res.get("reply", "")
            rec = recall_points(ans_text, expected)
            qual = heuristic_quality(ans_text, expected)
            recall_scores.append(rec)
            quality_scores.append(qual)

        # Count compactions if available
        if hasattr(agent, "compaction_count"):
            total_compactions += agent.compaction_count(conv_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Calculate memory growth in bytes
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        for u in user_ids:
            final_size = agent.memory_file_size(u)
            init_size = initial_memory_sizes.get(u, 0)
            memory_growth += max(0, final_size - init_size)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean Markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both benchmark suites: Standard Benchmark and Long-Context Stress Benchmark."""
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("=" * 80)
    print("PHASE 2 - TRACK 3 - DAY 17: MEMORY SYSTEMS BENCHMARK")
    print("=" * 80)

    # 1. Standard Benchmark
    if standard_path.exists():
        print("\n### 1. Standard Benchmark (10 conversations, multi-session recall)")
        print(f"Data source: {standard_path.name}")
        std_convs = load_conversations(standard_path)

        # Baseline Agent
        base_std = BaselineAgent(config, force_offline=True)
        base_std_row = run_agent_benchmark("Baseline Agent", base_std, std_convs, config)

        # Clean state for Advanced Agent to measure memory growth accurately
        user_ids = {c["user_id"] for c in std_convs}
        for u in user_ids:
            p = config.state_dir / "profiles" / u
            if p.exists():
                shutil.rmtree(p)

        # Advanced Agent
        adv_std = AdvancedAgent(config, force_offline=True)
        adv_std_row = run_agent_benchmark("Advanced Agent", adv_std, std_convs, config)

        print("\n" + format_rows([base_std_row, adv_std_row]))

    # 2. Long-Context Stress Benchmark
    if stress_path.exists():
        print("\n### 2. Long-Context Stress Benchmark (High token volume, compaction stress)")
        print(f"Data source: {stress_path.name}")
        stress_convs = load_conversations(stress_path)

        # Baseline Agent
        base_stress = BaselineAgent(config, force_offline=True)
        base_stress_row = run_agent_benchmark("Baseline Agent", base_stress, stress_convs, config)

        # Clean state for Advanced Agent to measure memory growth accurately
        user_ids = {c["user_id"] for c in stress_convs}
        for u in user_ids:
            p = config.state_dir / "profiles" / u
            if p.exists():
                shutil.rmtree(p)

        # Advanced Agent
        adv_stress = AdvancedAgent(config, force_offline=True)
        adv_stress_row = run_agent_benchmark("Advanced Agent", adv_stress, stress_convs, config)

        print("\n" + format_rows([base_stress_row, adv_stress_row]))

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
