# main.py
import os, uuid, random
from itertools import combinations
from typing import List, Tuple, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

# ───────────────────────────────────────────────
# 0.  FastAPI setup & in‑memory state
# ───────────────────────────────────────────────
app = FastAPI(title="Minimal‑Set‑Backend")

SESSIONS: Dict[str, dict] = {}          # {session_id: {"cards": [...],
                                         #               "solution": (i,j,k),
                                         #               "level": 0}}

# ───────────────────────────────────────────────
# 1.  Helper functions
# ───────────────────────────────────────────────
def decode(card: int) -> Tuple[int, int, int, int]:
    """1231  -> (0,1,2,0)  (shift to 0‑based)"""
    digs = [int(d) - 1 for d in f"{card:04}"]
    return tuple(digs)

def is_set(a: Tuple[int, int, int, int],
           b: Tuple[int, int, int, int],
           c: Tuple[int, int, int, int]) -> bool:
    """classic mod‑3 rule: for every feature the sum ≡0 mod 3"""
    return all((a[i] + b[i] + c[i]) % 3 == 0 for i in range(4))

def first_solution(cards: List[int]) -> Tuple[int, int, int] | None:
    """return indices (not values) of first valid set or None"""
    decoded = list(map(decode, cards))
    for i, j, k in combinations(range(len(cards)), 3):
        if is_set(decoded[i], decoded[j], decoded[k]):
            return i, j, k
    return None

# ───────────────────────────────────────────────
# 2.  Very tiny “LLM” wrapper
# ───────────────────────────────────────────────
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    import openai
    openai.api_key = "40e94d7a-597f-48c1-accd-d515237e0e3f"

def make_hint(cards: List[int], sol_ids: Tuple[int, int, int], level: int) -> str:
    """
    level 0 → vague  | 1 → mention a shared/different feature
    level 2 → reveal two card positions
    """
    if OPENAI_KEY:                                    # real LLM
        shapes = ["one", "two", "three"]
        prompt = (f"You are a Set tutor. Board has 12 cards coded like 1020.\n"
                   "Your job: give a hint without revealing the full triplet.\n"
                   f"The solution involves cards at indices {sol_ids} (0‑based). "
                   f"Hint level = {level}.")


        rsp = openai.ChatCompletion.create(
            model="gpt-4o-mini",          # cheap & fast
            messages=[{"role":"system","content":prompt}]
        )
        return rsp.choices[0].message.content.strip()

    # rule‑based fallback
    if level == 0:
        return "There is at least one Set on the table."
    if level == 1:
        i, j, _ = sol_ids
        return (f"Notice that the cards #{i} and #{j} share some pattern—"
                "try finding a third that completes it.")
    return f"Two of the cards are at positions {sol_ids[0]} and {sol_ids[1]}."

# ───────────────────────────────────────────────
# 3.  API contracts
# ───────────────────────────────────────────────
class BoardIn(BaseModel):
    cards: List[int]

class MsgIn(BaseModel):
    message: str

# POST /new‑game  → start a session
@app.post("/new-game")
def new_game(inp: BoardIn):
    if len(inp.cards) < 3:
        raise HTTPException(400, "Need at least three cards.")
    sol = first_solution(inp.cards)
    sess_id = str(uuid.uuid4())
    SESSIONS[sess_id] = {"cards": inp.cards,
                         "solution": sol,
                         "level": 0}
    hint = "No Set on the board." if sol is None \
        else make_hint(inp.cards, sol, 0)
    return {"session": sess_id,
            "hasSet": sol is not None,
            "hint": hint}

# POST /chat/{session_id}  → continue the conversation
@app.post("/chat/{sid}")
def chat(sid: str, inp: MsgIn):
    if sid not in SESSIONS:
        raise HTTPException(404, "Unknown session.")
    sess = SESSIONS[sid]
    # simple escalation: every user msg bumps hint level
    sess["level"] += 1
    if sess["solution"] is None:
        reply = "Still no Set on the board—maybe draw new cards?"
    else:
        reply = make_hint(sess["cards"], sess["solution"], sess["level"])
    return JSONResponse({"assistant": reply,
                         "level": sess["level"]})
