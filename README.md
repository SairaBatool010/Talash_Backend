# Hackathon Team Matchmaker — Backend

Recommender + AI agent API for matching hackathon teammates by skill complementarity
and semantic similarity of bios/interests.

## Install

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux/Git Bash:
source .venv/bin/activate  # or .venv/Scripts/activate on Windows Git Bash

pip install -r requirements.txt
```

## Set up your .env

Copy the example file and add your OpenAI key:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-your-key-here
```

## Run locally

```bash
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000` by default (use `--port` to change it if that port is blocked).
Check it's alive: `GET http://127.0.0.1:8000/health` → `{"status": "ok"}`

## Deploying (Render / Railway)

The app reads the port from the `PORT` environment variable via the included `Procfile`:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set `OPENAI_API_KEY` as an environment variable on the hosting platform (don't commit `.env`).

## Data model

No real database yet — profiles are stored in-memory in `data.py` (`all_profiles`,
`get_profile`, `save_profile`), structured so it can be swapped for a real DB later
without touching `recommender.py` or `agent.py`.

No authentication — `user_id` is just a string you pass directly in requests.

---

## Endpoints

### `GET /health`

Health check.

**Response**
```json
{ "status": "ok" }
```

---

### `POST /profiles`

Create or update a profile (matched by `id`).

**Request**
```json
{
  "id": "16",
  "name": "Jordan",
  "skills_have": ["Go", "Kubernetes"],
  "skills_want": ["React", "Design"],
  "bio": "Backend/infra engineer who wants to build something polished this weekend.",
  "roles_wanted": ["Backend"],
  "availability": "Full weekend"
}
```

**Response** — the saved profile, same shape as the request.

---

### `GET /profiles/{user_id}`

Fetch a single profile.

**Response**
```json
{
  "id": "2",
  "name": "Raj",
  "skills_have": ["Python", "FastAPI", "PostgreSQL"],
  "skills_want": ["Frontend", "React"],
  "bio": "Backend engineer, comfortable with APIs and databases...",
  "roles_wanted": ["Backend"],
  "availability": "Full weekend"
}
```

Returns `404` if the user doesn't exist.

---

### `GET /recommendations/{user_id}?top_n=5`

Ranked teammate suggestions for a user, with AI-generated reasons for each match.
`top_n` is optional (default 5).

**Response**
```json
{
  "matches": [
    {
      "user_id": "2",
      "name": "Raj",
      "score": 0.64,
      "skills": ["Python", "FastAPI", "PostgreSQL"],
      "reason": "Raj is a backend engineer with FastAPI experience, complementing Amina's frontend skills and desire for backend support."
    }
  ]
}
```

Returns `404` if the user doesn't exist.

---

### `POST /team/gaps`

Given a list of team member ids, identify which skills/roles the team collectively
wants but nobody on the team currently has.

**Request**
```json
{ "team_member_ids": ["1", "5"] }
```

**Response**
```json
{
  "team_members": [
    { "id": "1", "name": "Amina" },
    { "id": "5", "name": "Deja" }
  ],
  "team_has": ["CSS", "Figma", "Prototyping", "React", "UI Design"],
  "team_wants": ["Backend", "FastAPI"],
  "gaps": ["Backend", "FastAPI"]
}
```

---

### `POST /invite`

Send a teammate invite. Stored in-memory (not persisted across restarts).

**Request**
```json
{
  "from_user_id": "1",
  "to_user_id": "2",
  "reason": "Raj's backend skills complement my frontend focus."
}
```

**Response**
```json
{
  "status": "sent",
  "message": "Invite from Amina to Raj recorded.",
  "invite": {
    "from_user_id": "1",
    "to_user_id": "2",
    "reason": "Raj's backend skills complement my frontend focus."
  }
}
```

---

## Project structure

| File | Purpose |
|---|---|
| `data.py` | In-memory profile store + dummy data (swappable for a real DB) |
| `recommender.py` | Embedding + skill-complementarity + scoring, no OpenAI dependency |
| `agent.py` | OpenAI tool-calling agent: match reasons, team gap analysis, invites |
| `main.py` | FastAPI routes |

Run `python recommender.py` or `python agent.py` directly to sanity-check each layer
in isolation without starting the server.
