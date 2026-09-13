"""
Data access layer — in-memory store for hackathon profiles and teams.

Swappable design: all reads/writes go through get_profile / save_profile /
all_profiles / get_team / save_team / all_teams, so this module can later be
replaced with a real DB without touching recommender.py or agent.py.

team_status on a profile is one of:
  - "looking_for_team"        — solo, wants to join a team
  - "has_team_looking_for_more" — already on a team, team wants more members
  - "not_looking"             — solo or on a full team, don't recommend to/for them

team_id is None for someone not yet on a team.
"""

import random
import string

_PROFILES = [
    {
        "id": "1",
        "name": "Amina",
        "skills_have": ["React", "CSS", "Figma"],
        "skills_want": ["Backend", "FastAPI"],
        "bio": "Frontend dev who loves building clean, accessible UIs. Looking to team up with someone strong in backend/APIs so we can ship a full-stack product fast.",
        "roles_wanted": ["Frontend"],
        "availability": "Full weekend",
        "project_interests": ["EdTech", "Accessibility"],
        "education_background": "BS Computer Science",
        "team_status": "looking_for_team",
        "team_id": None,
    },
    {
        "id": "2",
        "name": "Raj",
        "skills_have": ["Python", "FastAPI", "PostgreSQL"],
        "skills_want": ["Frontend", "React"],
        "bio": "Backend engineer, comfortable with APIs and databases. Want to pair with a frontend person so I'm not stuck building ugly UIs myself.",
        "roles_wanted": ["Backend"],
        "availability": "Full weekend",
        "project_interests": ["FinTech", "Developer Tools"],
        "education_background": "BS Computer Engineering",
        "team_status": "has_team_looking_for_more",
        "team_id": "T1",
    },
    {
        "id": "3",
        "name": "Sofia",
        "skills_have": ["Machine Learning", "Python", "PyTorch"],
        "skills_want": ["Product", "Design"],
        "bio": "ML researcher interested in NLP and recommendation systems. Want to work on something socially impactful this weekend, ideally with a product-minded teammate.",
        "roles_wanted": ["ML Engineer"],
        "availability": "Saturday only",
        "project_interests": ["Social Good", "NLP"],
        "education_background": "MS Data Science",
        "team_status": "looking_for_team",
        "team_id": None,
    },
    {
        "id": "4",
        "name": "Liu",
        "skills_have": ["Product Management", "User Research"],
        "skills_want": ["Machine Learning", "Data Science"],
        "bio": "Ex-startup PM who's good at scoping MVPs and talking to users. Would love to pair with an ML person to build something that actually solves a problem.",
        "roles_wanted": ["Product"],
        "availability": "Full weekend",
        "project_interests": ["Social Good", "Productivity"],
        "education_background": "MBA",
        "team_status": "not_looking",
        "team_id": "T2",
    },
    {
        "id": "5",
        "name": "Deja",
        "skills_have": ["UI Design", "Figma", "Prototyping"],
        "skills_want": [],
        "bio": "Designer, mostly here for the vibes and to make things look good.",
        "roles_wanted": ["Design"],
        "availability": "Saturday only",
        "project_interests": ["Design Systems"],
        "education_background": "BFA Graphic Design",
        "team_status": "looking_for_team",
        "team_id": None,
    },
    {
        "id": "6",
        "name": "Marco",
        "skills_have": ["UI Design", "Branding"],
        "skills_want": ["UI Design"],
        "bio": "Visual designer who wants to collab with another designer on branding.",
        "roles_wanted": ["Design"],
        "availability": "Full weekend",
        "project_interests": ["Branding", "Consumer Apps"],
        "education_background": "BA Visual Arts",
        "team_status": "has_team_looking_for_more",
        "team_id": "T1",
    },
    {
        "id": "7",
        "name": "Priya",
        "skills_have": ["Node.js", "Express", "MongoDB"],
        "skills_want": ["React", "Mobile"],
        "bio": "Backend-leaning full-stack dev. Building a lot of hackathon MVPs lately and want a frontend/mobile partner this time.",
        "roles_wanted": ["Backend", "Full-stack"],
        "availability": "Full weekend",
        "project_interests": ["Consumer Apps", "Mobile"],
        "education_background": "BS Information Systems",
        "team_status": "has_team_looking_for_more",
        "team_id": "T1",
    },
    {
        "id": "8",
        "name": "Tom",
        "skills_have": ["Swift", "iOS", "React Native"],
        "skills_want": ["Backend", "Node.js"],
        "bio": "Mobile developer, iOS and React Native. Need someone to build a backend/API so my app has real data.",
        "roles_wanted": ["Mobile"],
        "availability": "Full weekend",
        "project_interests": ["Consumer Apps", "Mobile"],
        "education_background": "BS Software Engineering",
        "team_status": "has_team_looking_for_more",
        "team_id": "T1",
    },
    {
        "id": "9",
        "name": "Grace",
        "skills_have": ["Data Science", "Pandas", "SQL"],
        "skills_want": ["Frontend", "Visualization"],
        "bio": "Data analyst who's great at wrangling messy datasets and finding insights. Want to team with someone who can turn that into a slick dashboard.",
        "roles_wanted": ["Data Scientist"],
        "availability": "Sunday only",
        "project_interests": ["Sustainability", "Data Visualization"],
        "education_background": "BS Statistics",
        "team_status": "not_looking",
        "team_id": "T2",
    },
    {
        "id": "10",
        "name": "Hiro",
        "skills_have": ["DevOps", "AWS", "Docker"],
        "skills_want": ["Full-stack"],
        "bio": "Infra person. Can get anything deployed fast. Looking for a full-stack team that needs solid deployment/CI support.",
        "roles_wanted": ["DevOps"],
        "availability": "Full weekend",
        "project_interests": ["Developer Tools", "Cloud"],
        "education_background": "BS Computer Science",
        "team_status": "not_looking",
        "team_id": "T2",
    },
    {
        "id": "11",
        "name": "Nadia",
        "skills_have": ["Public Speaking", "Pitching", "Business Strategy"],
        "skills_want": ["Engineering", "Design"],
        "bio": "Business/strategy background, not super technical. Great at pitching and shaping the story. Hoping to join a team that already has the technical build handled.",
        "roles_wanted": ["Business", "Pitch"],
        "availability": "Full weekend",
        "project_interests": ["FinTech", "Social Good"],
        "education_background": "BA Business Administration",
        "team_status": "not_looking",
        "team_id": "T2",
    },
    {
        "id": "12",
        "name": "Kwame",
        "skills_have": ["Python", "Machine Learning", "Computer Vision"],
        "skills_want": ["UI Design", "Frontend"],
        "bio": "CV.",
        "roles_wanted": ["ML Engineer"],
        "availability": "Full weekend",
        "project_interests": ["Healthcare", "Computer Vision"],
        "education_background": "MS Artificial Intelligence",
        "team_status": "looking_for_team",
        "team_id": None,
    },
    {
        "id": "13",
        "name": "Elena",
        "skills_have": ["React", "TypeScript", "GraphQL"],
        "skills_want": ["Machine Learning"],
        "bio": "Frontend engineer who enjoys building polished, fast web apps. Want to pair with an ML person to build something AI-powered with a great UI on top.",
        "roles_wanted": ["Frontend"],
        "availability": "Full weekend",
        "project_interests": ["Healthcare", "AI Products"],
        "education_background": "BS Computer Science",
        "team_status": "looking_for_team",
        "team_id": None,
    },
    {
        "id": "14",
        "name": "Ben",
        "skills_have": [],
        "skills_want": ["Everything, I'm new to this"],
        "bio": "First hackathon ever! I know a little bit of Python from a course but mostly want to learn. Happy to help wherever needed.",
        "roles_wanted": ["Any"],
        "availability": "Full weekend",
        "project_interests": [],
        "education_background": "High school student",
        "team_status": "looking_for_team",
        "team_id": None,
    },
    {
        "id": "15",
        "name": "Yusuf",
        "skills_have": ["Backend", "Django", "PostgreSQL", "System Design"],
        "skills_want": ["React", "Design"],
        "bio": "Senior backend engineer, comfortable owning architecture and data models end-to-end. Looking for a frontend/design-minded partner to make the product presentable.",
        "roles_wanted": ["Backend"],
        "availability": "Full weekend",
        "project_interests": ["FinTech", "Developer Tools"],
        "education_background": "MS Computer Science",
        "team_status": "not_looking",
        "team_id": "T2",
    },
]

# Team T1 has 4/5 members and is open to 1 more (tests the "recommend into open team" path).
# Team T2 is intentionally full (5/5) to test the "don't recommend full teams" rule.
_TEAMS = [
    {
        "id": "T1",
        "name": "Team Consumer Apps",
        "project_idea": "A mobile app connecting local volunteers with elderly neighbors for errands.",
        "member_ids": ["2", "6", "7", "8"],
        "looking_for_skills": ["React"],
        "max_size": 5,
    },
    {
        "id": "T2",
        "name": "Team FinTech Backend",
        "project_idea": "A budgeting tool for freelancers with automated invoice tracking.",
        "member_ids": ["15", "9", "10", "11", "4"],
        "looking_for_skills": [],
        "max_size": 5,
    },
]


def all_profiles():
    """Return all profiles."""
    return _PROFILES


def get_profile(user_id: str):
    """Return a single profile dict by id, or None if not found."""
    for profile in _PROFILES:
        if profile["id"] == user_id:
            return profile
    return None


def save_profile(profile: dict):
    """Create or update a profile (matched by id). Returns the saved profile."""
    for i, existing in enumerate(_PROFILES):
        if existing["id"] == profile["id"]:
            _PROFILES[i] = profile
            return profile
    _PROFILES.append(profile)
    return profile


def all_teams():
    """Return all teams."""
    return _TEAMS


def get_team(team_id: str):
    """Return a single team dict by id, or None if not found."""
    for team in _TEAMS:
        if team["id"] == team_id:
            return team
    return None


def save_team(team: dict):
    """Create or update a team (matched by id). Returns the saved team."""
    for i, existing in enumerate(_TEAMS):
        if existing["id"] == team["id"]:
            _TEAMS[i] = team
            return team
    _TEAMS.append(team)
    return team


# ---------------------------------------------------------------------------
# Groups, channels, and messages (Fix 6) — a separate concept from _TEAMS
# above. _TEAMS is what the recommender scores people against for team
# formation; _GROUPS/_CHANNELS/_MESSAGES back the frontend's chat/group
# workflow (create a group, invite into it, post in its channel) and don't
# feed into recommender.py at all.
# ---------------------------------------------------------------------------

_GROUPS = []
_GROUP_INVITES = []
_CHANNELS = [
    {
        "id": "general",
        "name": "general",
        "description": "Hackathon-wide announcements and chat",
        "type": "general",
        "group_id": None,
        "allows_posting": True,
    }
]
_MESSAGES = {"general": []}

_next_ids = {}


def next_id(kind: str) -> str:
    n = _next_ids.setdefault(kind, 1)
    _next_ids[kind] += 1
    return f"{kind}-{n}"


def all_groups():
    return _GROUPS


def get_group(group_id: str):
    for g in _GROUPS:
        if g["id"] == group_id:
            return g
    return None


def save_group(group: dict):
    for i, existing in enumerate(_GROUPS):
        if existing["id"] == group["id"]:
            _GROUPS[i] = group
            return group
    _GROUPS.append(group)
    return group


def all_group_invites():
    return _GROUP_INVITES


def get_group_invite(invite_id: str):
    for inv in _GROUP_INVITES:
        if inv["id"] == invite_id:
            return inv
    return None


def save_group_invite(invite: dict):
    for i, existing in enumerate(_GROUP_INVITES):
        if existing["id"] == invite["id"]:
            _GROUP_INVITES[i] = invite
            return invite
    _GROUP_INVITES.append(invite)
    return invite


def all_channels():
    return _CHANNELS


def get_channel(channel_id: str):
    for c in _CHANNELS:
        if c["id"] == channel_id:
            return c
    return None


def save_channel(channel: dict):
    for i, existing in enumerate(_CHANNELS):
        if existing["id"] == channel["id"]:
            _CHANNELS[i] = channel
            return channel
    _CHANNELS.append(channel)
    _MESSAGES.setdefault(channel["id"], [])
    return channel


def get_messages(channel_id: str):
    return _MESSAGES.get(channel_id, [])


def add_message(channel_id: str, message: dict):
    _MESSAGES.setdefault(channel_id, []).append(message)
    return message


# ---------------------------------------------------------------------------
# Admin-issued registration invite codes: an admin creates one for a named
# person (before they have a profile/user_id at all), the person later
# redeems the code via /invite/confirm to learn their assigned user_id and
# proceed to fill out their profile via POST /profiles.
# ---------------------------------------------------------------------------

_ADMIN_INVITES = {}  # code -> {"user_id", "name", "email"}


def create_admin_invite(name: str, email: str) -> dict:
    user_id = next_id("invited-user")
    entry = {"user_id": user_id, "name": name, "email": email}
    code = _random_code()
    _ADMIN_INVITES[code] = entry
    return {"code": code, **entry}


def _random_code(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def get_admin_invite(code: str):
    return _ADMIN_INVITES.get(code)


# ---------------------------------------------------------------------------
# Invite requests: person-to-person "let's team up" requests (distinct from
# _GROUP_INVITES above, which is an existing group owner inviting someone
# into an already-formed group). Each request gets its own id, which doubles
# as the grouping key for its request-scoped chat via get_messages/add_message
# above (reused as-is, with invite_id passed in place of channel_id).
# ---------------------------------------------------------------------------

_INVITE_REQUESTS = []


def all_invite_requests():
    return _INVITE_REQUESTS


def get_invite_request(invite_id: str):
    for inv in _INVITE_REQUESTS:
        if inv["id"] == invite_id:
            return inv
    return None


def save_invite_request(invite: dict):
    for i, existing in enumerate(_INVITE_REQUESTS):
        if existing["id"] == invite["id"]:
            _INVITE_REQUESTS[i] = invite
            return invite
    _INVITE_REQUESTS.append(invite)
    return invite


# ---------------------------------------------------------------------------
# Admin-defined profile form schema — a single stored list of question
# definitions the admin can read/replace. Starts empty (no custom questions).
# ---------------------------------------------------------------------------

_FORM_SCHEMA = []


def get_form_schema():
    return _FORM_SCHEMA


def save_form_schema(questions: list):
    global _FORM_SCHEMA
    _FORM_SCHEMA = questions
    return _FORM_SCHEMA


if __name__ == "__main__":
    profiles = all_profiles()
    print(f"{len(profiles)} profiles loaded")
    for p in profiles:
        print(f"- {p['id']}: {p['name']} | has={p['skills_have']} | wants={p['skills_want']} | status={p['team_status']} | team={p['team_id']}")

    teams = all_teams()
    print(f"\n{len(teams)} teams loaded")
    for t in teams:
        print(f"- {t['id']}: {t['name']} | {len(t['member_ids'])}/{t['max_size']} members | needs={t['looking_for_skills']}")
