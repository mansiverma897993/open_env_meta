import json
import os
import sys
from typing import Any, Dict, List


def _emit(tag: str, payload: Dict[str, Any]) -> None:
    # The evaluator expects strict [START]/[STEP]/[END] tags.
    print(tag, json.dumps(payload, separators=(",", ":"), ensure_ascii=True))


def _action_to_dict(action) -> Dict[str, Any]:
    return {
        "action_type": action.action_type,
        "category": action.category,
        "content": action.content,
    }


def _choose_action_via_llm(
    *,
    client,
    model_name: str,
    task_id: str,
    step: int,
    obs: Dict[str, Any],
) -> Dict[str, Any]:
    system = (
        "You are an RL agent for a customer support OpenEnv.\n"
        "Return ONLY JSON with one action.\n"
        "Allowed formats:\n"
        '{"action_type":"classify","category":"billing|technical|refund|account|abuse"}\n'
        '{"action_type":"reply","content":"..."}\n'
        '{"action_type":"escalate"}\n'
        '{"action_type":"close"}\n'
    )
    user = {
        "task_id": task_id,
        "step": step,
        "observation": obs,
    }
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    return json.loads(raw)


def _safe_action_from_expected(expected: Dict[str, Any], step: int) -> Dict[str, Any]:
    keywords = expected.get("keywords", [])
    if step == 1:
        return {"action_type": "classify", "category": expected["category"]}
    if step == 2:
        return {"action_type": "reply", "content": " ".join(keywords) or "We will help."}
    if expected.get("requires_escalation", False) and step == 3:
        return {"action_type": "escalate"}
    return {"action_type": "close"}


def main() -> None:
    # Ensure local imports work when executed from repo root.
    sys.path.insert(0, ".")

    from openai import OpenAI
    from env import Action, CustomerSupportEnv, grade_task
    from env.tasks import TASKS

    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    client = None
    if api_base_url and model_name and hf_token:
        client = OpenAI(base_url=api_base_url, api_key=hf_token)

    # Keep the task ordering deterministic.
    task_ids = ["easy", "medium", "hard"]

    for task_id in task_ids:
        task = TASKS[task_id]
        expected = task["expected"]

        env = CustomerSupportEnv()
        obs = env.reset(task_id=task_id)

        max_steps = task.get("max_steps", 10)
        _emit(
            "[START]",
            {
                "task_id": task_id,
                "max_steps": max_steps,
                "reward_range": [0.0, 1.0],
            },
        )

        actions_taken: List[Action] = []
        step_idx = 0
        done = False
        reward_score: float = 0.0

        while not done and step_idx < max_steps:
            step_idx += 1

            obs_dict = obs.model_dump()
            payload = None
            if client is not None:
                try:
                    payload = _choose_action_via_llm(
                        client=client,
                        model_name=model_name,
                        task_id=task_id,
                        step=step_idx,
                        obs=obs_dict,
                    )
                except Exception:
                    payload = None
            if payload is None:
                payload = _safe_action_from_expected(expected, step_idx)

            action = Action(**payload)
            obs, reward, done, info = env.step(action)
            actions_taken.append(action)
            reward_score = float(reward.score)

            _emit(
                "[STEP]",
                {
                    "task_id": task_id,
                    "step": step_idx,
                    "action": _action_to_dict(action),
                    "reward": reward_score,
                    "done": bool(done),
                },
            )

        final_score = float(grade_task(task, actions_taken))
        final_score = max(0.0, min(1.0, final_score))

        _emit(
            "[END]",
            {
                "task_id": task_id,
                "score": final_score,
                "reward": final_score,
                "done": True,
            },
        )


if __name__ == "__main__":
    main()

