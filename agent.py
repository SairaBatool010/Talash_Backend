"""
Agent layer — OpenAI tool-calling loop on top of the raw recommender.

Three tools:
  - recommend_matches: wraps recommender.recommend()
  - detect_team_gaps: compares a team's collective skills/wants against what's missing
  - send_invite: appends to an in-memory pending-invites list

The agent is responsible for generating a short, specific one-sentence reason
per match (referencing real skills/bio content), not just returning raw scores.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from data import all_profiles, all_teams, get_profile, get_team
from recommender import recommend, recommend_for_team_formation

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# In-memory pending invites store.
_pending_invites = []


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def recommend_matches(user_id: str, top_n: int = 5):
    profiles = all_profiles()
    user = get_profile(user_id)
    if user is None:
        return {"error": f"user_id {user_id} not found"}

    results = recommend(user_id, profiles, top_n=top_n)
    return {
        "user": {
            "id": user["id"],
            "name": user["name"],
            "bio": user["bio"],
            "skills_have": user["skills_have"],
            "skills_want": user["skills_want"],
        },
        "candidates": [
            {
                "user_id": p["id"],
                "name": p["name"],
                "bio": p["bio"],
                "skills_have": p["skills_have"],
                "skills_want": p["skills_want"],
                "score": round(score, 3),
            }
            for p, score in results
        ],
    }


def team_recommend(user_id: str, top_n: int = 5):
    """
    Team-formation-aware recommendations, respecting team_status/team_id rules
    (see recommender.recommend_for_team_formation for the exact restrictions:
    full teams and "not_looking" users are never recommended).
    """
    user = get_profile(user_id)
    if user is None:
        return {"error": f"user_id {user_id} not found"}

    if user.get("team_status") == "not_looking":
        return {
            "user": {"id": user["id"], "name": user["name"], "team_status": user["team_status"]},
            "individuals": [],
            "teams": [],
            "note": "This user is not currently looking for a team.",
        }

    result = recommend_for_team_formation(user_id, all_profiles(), all_teams(), top_n=top_n)

    return {
        "user": {
            "id": user["id"],
            "name": user["name"],
            "bio": user["bio"],
            "skills_have": user["skills_have"],
            "skills_want": user["skills_want"],
            "project_interests": user.get("project_interests", []),
            "team_status": user["team_status"],
            "team_id": user.get("team_id"),
        },
        "individuals": [
            {
                "user_id": p["id"],
                "name": p["name"],
                "bio": p["bio"],
                "skills_have": p["skills_have"],
                "skills_want": p["skills_want"],
                "project_interests": p.get("project_interests", []),
                "score": round(score, 3),
            }
            for p, score in result["individuals"]
        ],
        "teams": [
            {
                "team_id": t["id"],
                "name": t["name"],
                "project_idea": t["project_idea"],
                "members": len(t["member_ids"]),
                "max_size": t["max_size"],
                "looking_for_skills": t["looking_for_skills"],
                "score": round(score, 3),
            }
            for t, score in result["teams"]
        ],
    }


def detect_team_gaps(team_member_ids: list):
    members = [get_profile(uid) for uid in team_member_ids]
    members = [m for m in members if m is not None]

    if not members:
        return {"error": "no valid team members found"}

    team_has = set()
    team_wants = set()
    for m in members:
        team_has.update(m.get("skills_have", []))
        team_wants.update(m.get("skills_want", []))

    # Gaps = things the team wants/needs but nobody on the team currently has.
    gaps = sorted(team_wants - team_has)

    return {
        "team_members": [{"id": m["id"], "name": m["name"]} for m in members],
        "team_has": sorted(team_has),
        "team_wants": sorted(team_wants),
        "gaps": gaps,
    }


def send_invite(from_user_id: str, to_user_id: str, reason: str):
    from_profile = get_profile(from_user_id)
    to_profile = get_profile(to_user_id)
    if from_profile is None or to_profile is None:
        return {"error": "one or both user_ids not found"}

    invite = {
        "from_user_id": from_user_id,
        "to_user_id": to_user_id,
        "reason": reason,
    }
    _pending_invites.append(invite)
    return {
        "status": "sent",
        "message": f"Invite from {from_profile['name']} to {to_profile['name']} recorded.",
        "invite": invite,
    }


def get_pending_invites():
    return _pending_invites


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_matches",
            "description": "Get ranked candidate teammates for a user based on skill complementarity and bio similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The id of the user requesting matches."},
                    "top_n": {"type": "integer", "description": "How many candidates to return.", "default": 5},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "team_recommend",
            "description": "Get team-formation-aware recommendations for a user: other solo individuals looking for a team, and/or open teams looking for more members. Respects team_status (skips users marked not_looking) and team capacity (never recommends a full team).",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The id of the user requesting matches."},
                    "top_n": {"type": "integer", "description": "How many candidates to return.", "default": 5},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_team_gaps",
            "description": "Given a list of user ids already on a team, identify skills/roles the team collectively wants but nobody on the team currently has.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_member_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "User ids of current team members.",
                    },
                },
                "required": ["team_member_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_invite",
            "description": "Send a teammate invite from one user to another with a reason.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_user_id": {"type": "string"},
                    "to_user_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Why this invite is being sent."},
                },
                "required": ["from_user_id", "to_user_id", "reason"],
            },
        },
    },
]

_TOOL_IMPL = {
    "recommend_matches": lambda args: recommend_matches(args["user_id"], args.get("top_n", 5)),
    "team_recommend": lambda args: team_recommend(args["user_id"], args.get("top_n", 5)),
    "detect_team_gaps": lambda args: detect_team_gaps(args["team_member_ids"]),
    "send_invite": lambda args: send_invite(args["from_user_id"], args["to_user_id"], args["reason"]),
}

SYSTEM_PROMPT = """You are a matchmaking assistant for a hackathon team-formation app.

When asked for teammate suggestions for a user, always call recommend_matches first
to get real candidate data — never invent candidates or scores.

For each candidate returned, generate a short, specific one-sentence reason explaining
why they're a good match. Reference ONLY actual skills, roles, or bio details that literally appear in the two
profiles' skills_have/skills_want/bio fields returned by the tool — never attribute a
skill to a person unless it is explicitly listed for them. Do not use generic filler
like "great match" or "good fit" without specifics.

When asked to analyze a team's gaps, call detect_team_gaps and summarize the result
in plain language, noting which skills/roles are missing.

When asked to send an invite, call send_invite with a clear, specific reason.

Always ground your answers in tool results — do not fabricate profile data.
"""


def get_matches_with_reasons(user_id: str, top_n: int = 5):
    """
    Get ranked candidates for user_id (via the raw recommender) and ask the LLM
    to generate a short, specific one-sentence reason per candidate, grounded
    only in the actual profile fields. Returns the candidates list with a
    'reason' string added to each entry (score/skills untouched).

    This bypasses the open-ended tool-calling loop so the output shape stays
    strictly structured for the API response.
    """
    data = recommend_matches(user_id, top_n)
    if "error" in data:
        return data

    if not data["candidates"]:
        return data

    prompt = f"""User profile:
{json.dumps(data["user"])}

Candidate profiles (ranked by match score):
{json.dumps(data["candidates"])}

For each candidate, write ONE short, specific sentence explaining why they're a good
match for the user. Reference only skills/bio details that literally appear in the
skills_have/skills_want/bio fields above — never attribute a skill to someone who
doesn't have it listed. Do not use generic filler like "great match" without specifics.

You MUST include an entry for EVERY candidate listed above, with no omissions.

Return ONLY a JSON object of the form:
{{"reasons": {{"<user_id>": "<one sentence reason>", ...}}}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
        reasons = parsed.get("reasons", {})
    except (json.JSONDecodeError, AttributeError):
        reasons = {}

    for candidate in data["candidates"]:
        candidate["reason"] = reasons.get(
            candidate["user_id"],
            f"{candidate['name']} is a ranked candidate based on skill and bio similarity.",
        )

    return data


def get_team_matches_with_reasons(user_id: str, top_n: int = 5):
    """
    Team-formation-aware version of get_matches_with_reasons: calls
    team_recommend() (which already applies the not_looking / full-team
    restrictions), then asks the LLM for one grounded reason per individual
    and per team. Returns the same dict shape as team_recommend with a
    'reason' string added to each entry in individuals/teams.
    """
    data = team_recommend(user_id, top_n)
    if "error" in data or "note" in data:
        return data

    if not data["individuals"] and not data["teams"]:
        return data

    prompt = f"""User profile:
{json.dumps(data["user"])}

Candidate individuals (ranked by match score):
{json.dumps(data["individuals"])}

Candidate open teams (ranked by match score):
{json.dumps(data["teams"])}

For each candidate individual, write ONE short sentence explaining why they're a good
teammate for the user, referencing only their actual skills_have/skills_want/bio/project_interests.

For each candidate team, write ONE short sentence explaining why the user fits that team,
referencing the team's project_idea/looking_for_skills and the user's actual skills/interests.

Never attribute a skill or interest to someone/something that doesn't have it listed above.

Return ONLY a JSON object of the form:
{{"individual_reasons": {{"<user_id>": "<reason>", ...}}, "team_reasons": {{"<team_id>": "<reason>", ...}}}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
        individual_reasons = parsed.get("individual_reasons", {})
        team_reasons = parsed.get("team_reasons", {})
    except (json.JSONDecodeError, AttributeError):
        individual_reasons = {}
        team_reasons = {}

    for candidate in data["individuals"]:
        candidate["reason"] = individual_reasons.get(
            candidate["user_id"],
            f"{candidate['name']} is a ranked candidate based on skill and interest similarity.",
        )
    for team in data["teams"]:
        team["reason"] = team_reasons.get(
            team["team_id"],
            f"{team['name']} is a ranked team fit based on project and skill overlap.",
        )

    return data


def run_agent(user_message: str, max_turns: int = 6):
    """
    Full tool-calling loop: send system + user message + tools to the model,
    execute whichever tool(s) it calls, feed results back, repeat until the
    model returns a final answer (no more tool calls).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        choice = response.choices[0]
        message = choice.message

        if not message.tool_calls:
            return message.content

        # Append the assistant's tool-call message, then each tool result.
        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            impl = _TOOL_IMPL.get(name)
            result = impl(args) if impl else {"error": f"unknown tool {name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    return "Agent did not converge to a final answer within max_turns."


if __name__ == "__main__":
    print("=== Test 1: teammate suggestions for Amina (id=1) ===")
    result = run_agent("Who should user 1 team up with? Suggest a few teammates and explain why.")
    print(result)

    print("\n=== Test 2: team gap analysis ===")
    result = run_agent("My team currently has user 1 and user 5 on it. What skills are we missing?")
    print(result)

    print("\n=== Pending invites (should be empty unless agent chose to invite) ===")
    print(get_pending_invites())
