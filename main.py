import random
import string
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data import (
    get_profile,
    save_profile,
    all_profiles,
    all_groups,
    get_group,
    save_group,
    all_group_invites,
    get_group_invite,
    save_group_invite,
    all_channels,
    get_channel,
    save_channel,
    get_messages,
    add_message,
    next_id,
    create_admin_invite,
    get_admin_invite,
)
from agent import get_matches_with_reasons, get_team_matches_with_reasons, detect_team_gaps, send_invite, generate_explanation
from recommender import embed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Profile(BaseModel):
    """
    Request/response shape for /profiles, matching the frontend's contract.
    Internally, profiles are stored with different key names (id, project_interests,
    team_id) since recommender.py/agent.py depend on those throughout — this model
    translates at the API boundary via to_internal()/from_internal() below.
    """
    user_id: str
    name: str
    email: str = ""
    skills_have: list[str] = []
    skills_want: list[str] = []
    interests: list[str] = []
    bio: str = ""
    roles_wanted: list[str] = []
    availability: str = ""
    group_id: Optional[str] = None
    education_background: str = ""
    team_status: str = "looking_for_team"  # looking_for_team | has_team_looking_for_more | not_looking

    def to_internal(self) -> dict:
        return {
            "id": self.user_id,
            "name": self.name,
            "email": self.email,
            "skills_have": self.skills_have,
            "skills_want": self.skills_want,
            "project_interests": self.interests,
            "bio": self.bio,
            "roles_wanted": self.roles_wanted,
            "availability": self.availability,
            "team_id": self.group_id,
            "education_background": self.education_background,
            "team_status": self.team_status,
        }


def profile_from_internal(profile: dict) -> dict:
    """Translate an internally-stored profile dict into the frontend's field names."""
    return {
        "user_id": profile["id"],
        "name": profile["name"],
        "email": profile.get("email", ""),
        "skills_have": profile.get("skills_have", []),
        "skills_want": profile.get("skills_want", []),
        "interests": profile.get("project_interests", []),
        "bio": profile.get("bio", ""),
        "roles_wanted": profile.get("roles_wanted", []),
        "availability": profile.get("availability", ""),
        "group_id": profile.get("team_id"),
    }


class TeamGapsRequest(BaseModel):
    team_member_ids: list[str]


class ConfirmInviteRequest(BaseModel):
    code: str


class AdminInviteRequest(BaseModel):
    name: str
    email: str


class EmbeddingsRequest(BaseModel):
    input: str = ""
    text: str = ""


class RecommendationExplanationRequest(BaseModel):
    requester: dict
    candidate: dict
    score: float
    components: dict = {}
    instruction: str = ""


class InviteRequest(BaseModel):
    to_user_id: str
    user_id: Optional[str] = None  # frontend's field name for the sender
    from_user_id: Optional[str] = None  # backward-compatible alias
    reason: Optional[str] = None

    def sender_id(self) -> Optional[str]:
        return self.user_id or self.from_user_id


class GroupCreateRequest(BaseModel):
    user_id: str


class GroupInviteRequest(BaseModel):
    group_id: str
    to_user_id: str


class GroupInviteRespondRequest(BaseModel):
    accept: bool


class MessageCreateRequest(BaseModel):
    user_id: str
    body: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embeddings")
def create_embedding(request: EmbeddingsRequest):
    text = request.text or request.input
    vector = embed(text)
    return {"embedding": vector.tolist()}


@app.post("/recommendation-explanations")
def recommendation_explanations(request: RecommendationExplanationRequest):
    explanation = generate_explanation(
        request.requester, request.candidate, request.score, request.components, request.instruction
    )
    return {"explanation": explanation}


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, top_n: int = 5):
    if get_profile(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    data = get_matches_with_reasons(user_id, top_n)
    recommendations = [
        {
            "user_id": candidate["user_id"],
            "name": candidate["name"],
            "skills": candidate["skills_have"],
            "shared_skills": candidate["shared_skills"],
            "complementary_skills": candidate["complementary_skills"],
            "reason": candidate.get("reason"),
            "invite_status": candidate["invite_status"],
            "score": candidate["score"],
            "components": candidate["components"],
        }
        for candidate in data.get("candidates", [])
    ]
    return {"recommendations": recommendations}


@app.get("/team-recommendations/{user_id}")
def get_team_recommendations(user_id: str, top_n: int = 5):
    if get_profile(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    return get_team_matches_with_reasons(user_id, top_n)


@app.post("/team/gaps")
def team_gaps(request: TeamGapsRequest):
    if not request.team_member_ids:
        raise HTTPException(status_code=400, detail="team_member_ids must not be empty")

    result = detect_team_gaps(request.team_member_ids)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/invite")
def invite(request: InviteRequest):
    sender_id = request.sender_id()
    if not sender_id:
        raise HTTPException(status_code=422, detail="user_id (sender) is required")

    result = send_invite(sender_id, request.to_user_id, request.reason)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/admin/invites")
def create_invite(request: AdminInviteRequest):
    return create_admin_invite(request.name, request.email)


@app.post("/invite/confirm")
def confirm_invite(request: ConfirmInviteRequest):
    entry = get_admin_invite(request.code)
    if entry is None:
        raise HTTPException(status_code=404, detail="Invalid or expired code")
    return entry


@app.post("/profiles")
def create_or_update_profile(profile: Profile):
    saved = save_profile(profile.to_internal())
    return profile_from_internal(saved)


@app.get("/profiles/{user_id}")
def read_profile(user_id: str):
    profile = get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="User not found")
    return profile_from_internal(profile)


# ---------------------------------------------------------------------------
# Groups, channels, messages (Fix 6)
# ---------------------------------------------------------------------------

def _generate_group_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@app.post("/group/create")
def create_group(request: GroupCreateRequest):
    if get_profile(request.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    group_id = next_id("group")
    group = {
        "id": group_id,
        "code": _generate_group_code(),
        "owner_id": request.user_id,
        "member_ids": [request.user_id],
    }
    save_group(group)

    creator = get_profile(request.user_id)
    channel_id = next_id("channel")
    save_channel(
        {
            "id": channel_id,
            "name": f"{creator['name']}'s group",
            "description": "Group channel",
            "type": "group",
            "group_id": group_id,
            "allows_posting": True,
        }
    )

    return {"id": group_id, "code": group["code"]}


@app.get("/users/search")
def search_users(q: str = ""):
    query = q.strip().lower()
    results = []
    for p in all_profiles():
        if not query:
            match = True
        else:
            match = query in p["name"].lower() or any(query in s.lower() for s in p.get("skills_have", []))
        if match:
            results.append(
                {
                    "id": p["id"],
                    "name": p["name"],
                    "email": p.get("email", ""),
                    "skills_have": p.get("skills_have", []),
                }
            )
    return {"users": results}


@app.post("/group/invite")
def group_invite(request: GroupInviteRequest):
    group = get_group(request.group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")
    if get_profile(request.to_user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    invite_id = next_id("invite")
    save_group_invite(
        {
            "id": invite_id,
            "group_id": request.group_id,
            "from_user_id": group["owner_id"],
            "to_user_id": request.to_user_id,
            "status": "pending",
        }
    )
    return {"status": "sent"}


@app.get("/group/invites")
def group_invites(user_id: str = ""):
    invitations = []
    for inv in all_group_invites():
        if inv["to_user_id"] != user_id:
            continue
        from_profile = get_profile(inv["from_user_id"])
        group = get_group(inv["group_id"])
        invitations.append(
            {
                "id": inv["id"],
                "from_name": from_profile["name"] if from_profile else "Unknown",
                "group_name": group["id"] if group else "Unknown",
                "status": inv["status"],
            }
        )
    return {"invitations": invitations}


@app.post("/group/invite/{invite_id}/respond")
def respond_to_group_invite(invite_id: str, request: GroupInviteRespondRequest):
    invite = get_group_invite(invite_id)
    if invite is None:
        raise HTTPException(status_code=404, detail="Invite not found")

    invite["status"] = "accepted" if request.accept else "declined"
    save_group_invite(invite)

    if request.accept:
        group = get_group(invite["group_id"])
        if group and invite["to_user_id"] not in group["member_ids"]:
            group["member_ids"].append(invite["to_user_id"])
            save_group(group)

    return {"status": invite["status"]}


@app.get("/groups/{group_id}/members")
def group_members(group_id: str):
    group = get_group(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")

    members = []
    for uid in group["member_ids"]:
        profile = get_profile(uid)
        if profile is None:
            continue
        members.append(
            {
                "user_id": profile["id"],
                "name": profile["name"],
                "email": profile.get("email", ""),
                "role": "owner" if uid == group["owner_id"] else "member",
            }
        )
    return {"members": members}


@app.get("/channels")
def list_channels(user_id: str = ""):
    user_groups = {g["id"] for g in all_groups() if user_id in g["member_ids"]}
    channels = [c for c in all_channels() if c["group_id"] is None or c["group_id"] in user_groups]
    return {"channels": channels}


@app.get("/channels/{channel_id}/messages")
def get_channel_messages(channel_id: str):
    if get_channel(channel_id) is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    return {"messages": get_messages(channel_id)}


@app.post("/channels/{channel_id}/messages")
def post_channel_message(channel_id: str, request: MessageCreateRequest):
    channel = get_channel(channel_id)
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    if not channel.get("allows_posting", True):
        raise HTTPException(status_code=403, detail="This channel does not allow posting")

    author = get_profile(request.user_id)
    if author is None:
        raise HTTPException(status_code=404, detail="User not found")

    message = {
        "id": next_id("message"),
        "user_id": request.user_id,
        "author_name": author["name"],
        "body": request.body,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    add_message(channel_id, message)
    return message
