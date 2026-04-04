import json
import os
from typing import List, Dict, Any


def load_tickets(path=None) -> List[Dict[str, Any]]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.json")
    with open(path) as f:
        return json.load(f)


def format_observation(obs) -> str:
    lines = [
        f"Ticket : {obs.ticket_id}",
        f"Status : {obs.status}",
        f"Query  : {obs.customer_query}",
    ]
    for i, msg in enumerate(obs.history, 1):
        lines.append(f"  [{i}] {msg}")
    return "\n".join(lines)


def log_step(step, action, reward):
    cat = action.category or "-"
    print(f"step {step:>2} | {action.action_type:<10} cat={cat:<12} score={reward.score:.2f} | {reward.feedback}")
