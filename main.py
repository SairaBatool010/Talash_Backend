import random
import string
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    all_invite_requests,
    get_invite_request,
    save_invite_request,
    get_form_schema,
    save_form_schema,
)
from agent import get_matches_with_reasons, get_team_matches_with_reasons, detect_team_gaps, generate_explanation
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

    Internally, profiles are stored with different key names for the legacy
    recommender-matching fields (project_interests, team_id, team_status —
    still used as-is by recommender.py/agent.py's /team-recommendations flow)
    since rewriting those would ripple through the scoring logic. The NEW
    group system (group/invite/admin dashboard) uses its own internal
    "group_id" and "custom_fields" keys, kept deliberately separate from the
    legacy "team_id" so the two systems don't collide. This model translates
    at the API boundary via to_internal()/profile_from_internal() below.

    NOTE: the "team_status" field in the API response is NOT the legacy
    recommender vocabulary (looking_for_team/has_team_looking_for_more/
    not_looking) — it's freshly computed from group_id + group capacity
    (available/forming/finalized) per the new contract. See compute_team_status().
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
    custom_fields: dict = {}
    education_background: str = ""

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
            "group_id": self.group_id,
            "custom_fields": self.custom_fields,
            "education_background": self.education_background,
        }


def compute_team_status(profile: dict) -> str:
    """available (no group) / forming (group under capacity) / finalized (group full)."""
    group_id = profile.get("group_id")
    if not group_id:
        return "available"
    group = get_group(group_id)
    if group is None:
        return "available"
    capacity = group.get("capacity", 5)
    member_count = len(group.get("member_ids", []))
    return "finalized" if member_count >= capacity else "forming"


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
        "group_id": profile.get("group_id"),
        "custom_fields": profile.get("custom_fields", {}),
        "team_status": compute_team_status(profile),
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
    from_user_id: Optional[str] = None  # accepted alias, same value
    note: Optional[str] = None

    def sender_id(self) -> Optional[str]:
        return self.user_id or self.from_user_id


class InviteRespondRequest(BaseModel):
    accept: bool


class InviteMessageCreateRequest(BaseModel):
    user_id: str
    body: str


class ProfileQuestion(BaseModel):
    id: str
    key: str
    label: str
    type: str  # short_text | long_text | multi_select | single_select
    required: Optional[bool] = None
    options: Optional[list[str]] = None
    baseline: Optional[bool] = None


class GroupCreateRequest(BaseModel):
    user_id: str
    name: Optional[str] = None


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


def _invite_request_response(record: dict) -> dict:
    """Attach from_name/to_name (derived, not stored) to a stored invite-request record."""
    from_profile = get_profile(record["from_user_id"])
    to_profile = get_profile(record["to_user_id"])
    return {
        "id": record["id"],
        "from_user_id": record["from_user_id"],
        "to_user_id": record["to_user_id"],
        "from_name": from_profile["name"] if from_profile else "Unknown",
        "to_name": to_profile["name"] if to_profile else "Unknown",
        "note": record.get("note"),
        "status": record["status"],
        "created_at": record["created_at"],
    }


@app.post("/invite")
def invite(request: InviteRequest):
    sender_id = request.sender_id()
    if not sender_id:
        raise HTTPException(status_code=422, detail="user_id (sender) is required")
    if get_profile(sender_id) is None or get_profile(request.to_user_id) is None:
        raise HTTPException(status_code=404, detail="one or both user_ids not found")

    record = {
        "id": next_id("invite-request"),
        "from_user_id": sender_id,
        "to_user_id": request.to_user_id,
        "note": request.note,
        "status": "sent",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_invite_request(record)
    return _invite_request_response(record)


@app.get("/invite/{invite_id}/messages")
def get_invite_messages(invite_id: str):
    if get_invite_request(invite_id) is None:
        raise HTTPException(status_code=404, detail="Invite not found")
    return get_messages(invite_id)


@app.post("/invite/{invite_id}/messages")
def post_invite_message(invite_id: str, request: InviteMessageCreateRequest):
    if get_invite_request(invite_id) is None:
        raise HTTPException(status_code=404, detail="Invite not found")

    author = get_profile(request.user_id)
    if author is None:
        raise HTTPException(status_code=404, detail="User not found")

    message = {
        "id": next_id("message"),
        "channel_id": invite_id,
        "user_id": request.user_id,
        "author_name": author["name"],
        "body": request.body,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    add_message(invite_id, message)
    return message


@app.post("/invite/{invite_id}/respond")
def respond_to_invite_request(invite_id: str, request: InviteRespondRequest):
    invite_req = get_invite_request(invite_id)
    if invite_req is None:
        raise HTTPException(status_code=404, detail="Invite not found")

    if not request.accept:
        invite_req["status"] = "declined"
        save_invite_request(invite_req)
        return {"status": "declined"}

    sender_id = invite_req["from_user_id"]
    to_user_id = invite_req["to_user_id"]
    sender_profile = get_profile(sender_id)
    sender_group_id = sender_profile.get("group_id") if sender_profile else None
    sender_group = get_group(sender_group_id) if sender_group_id else None

    if sender_group is not None:
        capacity = sender_group.get("capacity", 5)
        member_count = len(sender_group.get("member_ids", []))
        return JSONResponse(
            status_code=409,
            content={
                "error": "sender_already_grouped",
                "group_name": sender_group.get("name", sender_group["id"]),
                "group_id": sender_group["id"],
                "open_spots": capacity - member_count,
            },
        )

    # Sender has no group yet: form a brand-new one with sender + recipient.
    group_id = next_id("group")
    to_profile = get_profile(to_user_id)
    group = {
        "id": group_id,
        "code": _generate_group_code(),
        "name": f"{sender_profile['name']} & {to_profile['name']}" if sender_profile and to_profile else group_id,
        "owner_id": sender_id,
        "member_ids": [sender_id, to_user_id],
        "capacity": 5,
    }
    save_group(group)

    if sender_profile is not None:
        sender_profile["group_id"] = group_id
        save_profile(sender_profile)
    if to_profile is not None:
        to_profile["group_id"] = group_id
        save_profile(to_profile)

    invite_req["status"] = "accepted"
    save_invite_request(invite_req)
    return {"status": "accepted"}


@app.get("/invites")
def list_invite_requests(user_id: str = ""):
    results = []
    for inv in all_invite_requests():
        if inv["from_user_id"] != user_id and inv["to_user_id"] != user_id:
            continue
        response = _invite_request_response(inv)
        response["direction"] = "incoming" if inv["to_user_id"] == user_id else "outgoing"
        results.append(response)
    return results


@app.get("/admin/form-schema")
def get_admin_form_schema():
    return {"questions": get_form_schema()}


@app.post("/admin/form-schema")
def set_admin_form_schema(questions: list[ProfileQuestion]):
    saved = save_form_schema([q.model_dump(exclude_none=True) for q in questions])
    return {"questions": saved}


def _team_rows() -> list:
    rows = []
    for group in all_groups():
        capacity = group.get("capacity", 5)
        member_count = len(group.get("member_ids", []))
        rows.append(
            {
                "id": group["id"],
                "name": group.get("name", group["id"]),
                "member_count": member_count,
                "capacity": capacity,
                "status": "complete" if member_count >= capacity else "forming",
            }
        )
    return rows


def _no_group_profiles() -> list:
    return [p for p in all_profiles() if not p.get("group_id")]


@app.get("/admin/teams/summary")
def admin_teams_summary():
    rows = _team_rows()
    return {
        "no_group": len(_no_group_profiles()),
        "partial": sum(1 for r in rows if r["status"] == "forming"),
        "complete": sum(1 for r in rows if r["status"] == "complete"),
    }


@app.get("/admin/teams")
def admin_teams():
    return {
        "teams": _team_rows(),
        "no_group_participants": [{"user_id": p["id"], "name": p["name"]} for p in _no_group_profiles()],
    }


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
    creator = get_profile(request.user_id)
    if creator is None:
        raise HTTPException(status_code=404, detail="User not found")

    group_id = next_id("group")
    group_name = request.name or f"{creator['name']}'s Team"
    group = {
        "id": group_id,
        "name": group_name,
        "code": _generate_group_code(),
        "owner_id": request.user_id,
        "member_ids": [request.user_id],
        "capacity": 5,
    }
    save_group(group)

    creator["group_id"] = group_id
    save_profile(creator)

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

    return {"id": group_id, "code": group["code"], "name": group["name"]}


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
                "group_name": group.get("name", group["id"]) if group else "Unknown",
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

            joiner = get_profile(invite["to_user_id"])
            if joiner is not None:
                joiner["group_id"] = invite["group_id"]
                save_profile(joiner)

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
