from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data import get_profile, save_profile
from agent import get_matches_with_reasons, get_team_matches_with_reasons, detect_team_gaps, send_invite

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Profile(BaseModel):
    id: str
    name: str
    skills_have: list[str] = []
    skills_want: list[str] = []
    bio: str = ""
    roles_wanted: list[str] = []
    availability: str = ""
    project_interests: list[str] = []
    education_background: str = ""
    team_status: str = "looking_for_team"  # looking_for_team | has_team_looking_for_more | not_looking
    team_id: Optional[str] = None


class TeamGapsRequest(BaseModel):
    team_member_ids: list[str]


class InviteRequest(BaseModel):
    from_user_id: str
    to_user_id: str
    reason: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, top_n: int = 5):
    if get_profile(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    data = get_matches_with_reasons(user_id, top_n)
    matches = [
        {
            "user_id": candidate["user_id"],
            "name": candidate["name"],
            "score": candidate["score"],
            "skills": candidate["skills_have"],
            "reason": candidate.get("reason"),
        }
        for candidate in data.get("candidates", [])
    ]
    return {"matches": matches}


@app.get("/team-recommendations/{user_id}")
def get_team_recommendations(user_id: str, top_n: int = 5):
    if get_profile(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    return get_team_matches_with_reasons(user_id, top_n)


@app.post("/team/gaps")
def team_gaps(request: TeamGapsRequest):
    return detect_team_gaps(request.team_member_ids)


@app.post("/invite")
def invite(request: InviteRequest):
    return send_invite(request.from_user_id, request.to_user_id, request.reason)


@app.post("/profiles")
def create_or_update_profile(profile: Profile):
    saved = save_profile(profile.model_dump())
    return saved


@app.get("/profiles/{user_id}")
def read_profile(user_id: str):
    profile = get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="User not found")
    return profile
