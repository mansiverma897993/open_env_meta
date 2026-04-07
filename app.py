import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from env.environment import CustomerSupportEnv
from env.models import Action
from env.tasks import TASKS

app = FastAPI(title="Customer Support OpenEnv", version="1.0.0")

# one env per session
sessions = {}

def get_env(session_id="default"):
    if session_id not in sessions:
        sessions[session_id] = CustomerSupportEnv()
    return sessions[session_id]

class ResetRequest(BaseModel):
    task_id: str | None = None
    session_id: str = "default"


def _reset_impl(*, task_id: str | None, session_id: str):
    env = get_env(session_id)
    try:
        obs = env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs_dump = obs.model_dump()
    return {
        "observation": obs_dump,
        "state": obs_dump,
        "reward_range": [0.0, 1.0],
        "task": {
            "id": env.current_task["id"],
            "description": env.current_task["description"],
            "max_steps": env.current_task["max_steps"],
        },
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><body style="font-family:sans-serif;background:#0f1117;color:#e0e0e0;max-width:700px;margin:50px auto;padding:0 24px">
    <h1 style="color:#7ee787">Customer Support OpenEnv</h1>
    <p>An OpenEnv RL environment for customer support automation.</p>
    <h2 style="color:#58a6ff">Endpoints</h2>
    <ul>
      <li><a href="/docs" style="color:#58a6ff">/docs</a> &mdash; Swagger UI</li>
      <li><code>GET /reset?task_id=easy|medium|hard</code></li>
      <li><code>POST /step</code> &mdash; send an Action</li>
      <li><code>GET /state</code></li>
      <li><a href="/tasks" style="color:#58a6ff">GET /tasks</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/reset")
def reset(task_id: str = None, session_id: str = "default"):
    return _reset_impl(task_id=task_id, session_id=session_id)


@app.post("/reset")
def reset_post(
    req: ResetRequest | None = None,
    task_id: str | None = None,
    session_id: str = "default",
):
    # Prefer values from the request body, but fall back to query params.
    effective_task_id = task_id if req is None else (req.task_id if req.task_id is not None else task_id)
    effective_session_id = session_id if req is None else req.session_id
    return _reset_impl(task_id=effective_task_id, session_id=effective_session_id)


@app.post("/step")
def step(action: Action, session_id: str = "default"):
    env = get_env(session_id)
    if not env.current_task:
        raise HTTPException(400, "Call /reset first.")
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    obs_dump = obs.model_dump()
    return {
        "observation": obs_dump,
        "state": obs_dump,
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str = "default"):
    env = get_env(session_id)
    if not env.current_task:
        raise HTTPException(400, "Call /reset first.")
    return env.state()


@app.post("/state")
def state_post(session_id: str = "default"):
    return state(session_id=session_id)


@app.get("/tasks")
def list_tasks():
    return [
        {
            "id": t["id"],
            "description": t["description"],
            "max_steps": t["max_steps"],
            "requires_escalation": t["expected"]["requires_escalation"],
        }
        for t in TASKS.values()
    ]


@app.post("/tasks")
def list_tasks_post():
    return list_tasks()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)