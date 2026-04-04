from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Observation(BaseModel):
    ticket_id: str
    customer_query: str
    history: List[str]
    status: str


class Action(BaseModel):
    action_type: str  # classify | reply | escalate | close
    content: Optional[str] = None
    category: Optional[str] = None


class Reward(BaseModel):
    score: float
    feedback: str
    breakdown: Dict[str, Any] = {}