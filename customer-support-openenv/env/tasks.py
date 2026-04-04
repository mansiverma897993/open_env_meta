TASKS = {
    "easy": {
        "id": "easy",
        "description": "Classify a customer ticket into the right category.",
        "input": {
            "ticket_id": "T001",
            "customer_query": "I was charged twice for my order #ORD-8821 and need the duplicate payment removed.",
            "history": [],
            "status": "open",
        },
        "expected": {
            "category": "billing",
            "keywords": ["refund", "charge", "payment", "duplicate", "billing"],
            "requires_escalation": False,
        },
        "max_steps": 5,
    },

    "medium": {
        "id": "medium",
        "description": "Classify the ticket and give a helpful reply.",
        "input": {
            "ticket_id": "T002",
            "customer_query": "The app keeps crashing on my iPhone 15. I already restarted my phone twice.",
            "history": [],
            "status": "open",
        },
        "expected": {
            "category": "technical",
            "keywords": ["reinstall", "update", "cache", "support", "technical", "version"],
            "requires_escalation": False,
        },
        "max_steps": 8,
    },

    "hard": {
        "id": "hard",
        "description": "Full pipeline — classify, reply, escalate if needed, then close.",
        "input": {
            "ticket_id": "T003",
            "customer_query": "I have been waiting three weeks for a refund your team promised. I am considering legal action.",
            "history": [
                "Agent: We apologise. Your refund is being processed.",
                "Customer: Two weeks and still nothing!",
                "Agent: We escalated this to our billing team.",
                "Customer: Another week gone. I want to speak to a manager!",
            ],
            "status": "pending",
        },
        "expected": {
            "category": "billing",
            "keywords": ["escalat", "manager", "priority", "urgent", "legal", "refund", "apologize", "sorry"],
            "requires_escalation": True,
        },
        "max_steps": 10,
    },
}

TASK_LIST = list(TASKS.values())