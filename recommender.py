"""
Recommender core — no OpenAI dependency, fully testable offline.

Combines:
  - semantic similarity of bios (sentence-transformers embeddings, cosine sim)
  - skill complementarity (mutual "have vs want" overlap)
into a single weighted match score.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from data import all_profiles, all_teams, get_team

# Weights are easy to tune here.
SEMANTIC_WEIGHT = 0.5
SKILL_WEIGHT = 0.5

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    """Embed a piece of text into a vector using all-MiniLM-L6-v2."""
    if not text:
        text = ""
    return _get_model().encode(text, convert_to_numpy=True)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def skill_complementarity(profile_a: dict, profile_b: dict) -> float:
    """
    Proportion of mutual skill/want overlap: how much of what A wants, B has,
    and how much of what B wants, A has. Averaged into a single 0-1 score.

    Returns 0.0 if either side has nothing to want (no basis for complementarity).
    """
    a_have = set(profile_a.get("skills_have", []))
    a_want = set(profile_a.get("skills_want", []))
    b_have = set(profile_b.get("skills_have", []))
    b_want = set(profile_b.get("skills_want", []))

    scores = []

    if a_want:
        scores.append(len(a_want & b_have) / len(a_want))
    if b_want:
        scores.append(len(b_want & a_have) / len(b_want))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def match_score(profile_a: dict, profile_b: dict) -> float:
    """Weighted combination of bio semantic similarity and skill complementarity."""
    emb_a = embed(profile_a.get("bio", ""))
    emb_b = embed(profile_b.get("bio", ""))
    semantic = _cosine_sim(emb_a, emb_b)
    # cosine sim can be slightly negative; clamp to 0-1 range
    semantic = max(0.0, min(1.0, semantic))

    skill = skill_complementarity(profile_a, profile_b)

    return SEMANTIC_WEIGHT * semantic + SKILL_WEIGHT * skill


def match_components(requester: dict, candidate: dict) -> dict:
    """
    Full score breakdown for a requester -> candidate match, for API responses
    that need to show more than just the final combined score.

    requester_gets: skills the candidate has that the requester wants
    candidate_gets: skills the requester has that the candidate wants
    mutual_skill_matches: union of requester_gets and candidate_gets (all the
      "have vs want" pairings that line up between the two people)
    shared_skills: skills both people already have in common
    """
    emb_a = embed(requester.get("bio", ""))
    emb_b = embed(candidate.get("bio", ""))
    semantic = max(0.0, min(1.0, _cosine_sim(emb_a, emb_b)))

    skill = skill_complementarity(requester, candidate)

    r_have = set(requester.get("skills_have", []))
    r_want = set(requester.get("skills_want", []))
    c_have = set(candidate.get("skills_have", []))
    c_want = set(candidate.get("skills_want", []))

    requester_gets = sorted(r_want & c_have)
    candidate_gets = sorted(c_want & r_have)
    shared_skills = sorted(r_have & c_have)
    mutual_skill_matches = sorted(set(requester_gets) | set(candidate_gets))

    return {
        "score": SEMANTIC_WEIGHT * semantic + SKILL_WEIGHT * skill,
        "semantic_similarity": semantic,
        "skill_complementarity": skill,
        "shared_skills": shared_skills,
        "complementary_skills": requester_gets,
        "mutual_skill_matches": mutual_skill_matches,
        "requester_gets": requester_gets,
        "candidate_gets": candidate_gets,
    }


def recommend(user_id: str, profiles: list, top_n: int = 5) -> list:
    """
    Return up to top_n (profile, score) tuples ranked by match_score,
    excluding the user themselves. Returns [] if user_id not found.
    """
    user_profile = None
    for p in profiles:
        if p["id"] == user_id:
            user_profile = p
            break
    if user_profile is None:
        return []

    candidates = [p for p in profiles if p["id"] != user_id]
    scored = [(p, match_score(user_profile, p)) for p in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


def _interest_text(profile: dict) -> str:
    """Bio + project interests + education, joined for semantic comparison."""
    parts = [profile.get("bio", "")]
    parts.extend(profile.get("project_interests", []))
    parts.append(profile.get("education_background", ""))
    return " ".join(p for p in parts if p)


def team_match_score(profile: dict, team: dict) -> float:
    """
    How well a person fits a team: semantic similarity of the person's
    bio/interests/education vs the team's project idea, combined with how
    many of the team's still-needed skills the person has.
    """
    person_emb = embed(_interest_text(profile))
    team_emb = embed(team.get("project_idea", ""))
    semantic = max(0.0, min(1.0, _cosine_sim(person_emb, team_emb)))

    needed = set(team.get("looking_for_skills", []))
    have = set(profile.get("skills_have", []))
    skill_fit = (len(needed & have) / len(needed)) if needed else 0.0

    return SEMANTIC_WEIGHT * semantic + SKILL_WEIGHT * skill_fit


def recommend_for_team_formation(user_id: str, profiles: list, teams: list, top_n: int = 5) -> dict:
    """
    Team-formation-aware recommendations, respecting team_status/team_id rules:
      - "not_looking": no recommendations at all.
      - "looking_for_team" (no team yet): recommend both other solo individuals
        who are also looking, and open teams (not full, not already a member).
      - "has_team_looking_for_more": recommend solo individuals who fit what
        THEIR team still needs (scored against the team, not just the asker).

    Returns {"individuals": [(profile, score)], "teams": [(team, score)]}.
    Either list may be empty depending on the user's status.
    """
    user = next((p for p in profiles if p["id"] == user_id), None)
    if user is None:
        return {"individuals": [], "teams": []}

    status = user.get("team_status", "looking_for_team")
    if status == "not_looking":
        return {"individuals": [], "teams": []}

    open_pool = [
        p for p in profiles
        if p["id"] != user_id and p.get("team_status") == "looking_for_team"
    ]

    if status == "has_team_looking_for_more":
        team = get_team(user.get("team_id"))
        if team is None or len(team["member_ids"]) >= team["max_size"]:
            return {"individuals": [], "teams": []}
        scored = [(p, team_match_score(p, team)) for p in open_pool]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return {"individuals": scored[:top_n], "teams": []}

    # status == "looking_for_team": recommend both solo people and open teams
    scored_individuals = [(p, match_score(user, p)) for p in open_pool]
    scored_individuals.sort(key=lambda pair: pair[1], reverse=True)

    open_teams = [
        t for t in teams
        if user_id not in t["member_ids"] and len(t["member_ids"]) < t["max_size"]
    ]
    scored_teams = [(t, team_match_score(user, t)) for t in open_teams]
    scored_teams.sort(key=lambda pair: pair[1], reverse=True)

    return {
        "individuals": scored_individuals[:top_n],
        "teams": scored_teams[:top_n],
    }


if __name__ == "__main__":
    profiles = all_profiles()
    teams = all_teams()

    for user_id in ["1", "2", "3", "5"]:
        user = next(p for p in profiles if p["id"] == user_id)
        print(f"\n=== Recommendations for {user['name']} (id={user_id}) ===")
        results = recommend(user_id, profiles, top_n=5)
        for candidate, score in results:
            print(f"  {score:.3f}  {candidate['name']:8s}  has={candidate['skills_have']}")

    print("\n\n=== Team-formation-aware recommendations ===")
    for user_id in ["1", "2", "9", "6"]:
        user = next(p for p in profiles if p["id"] == user_id)
        print(f"\n--- {user['name']} (status={user['team_status']}, team={user['team_id']}) ---")
        result = recommend_for_team_formation(user_id, profiles, teams, top_n=5)
        print("  Individuals:")
        for candidate, score in result["individuals"]:
            print(f"    {score:.3f}  {candidate['name']}")
        print("  Teams:")
        for team, score in result["teams"]:
            print(f"    {score:.3f}  {team['name']} ({len(team['member_ids'])}/{team['max_size']}, needs={team['looking_for_skills']})")
