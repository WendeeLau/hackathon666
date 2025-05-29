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
        # prompt = (f"You are a Set tutor. Board has 12 cards coded like 1020.\n"
        #            "Your job: give a hint without revealing the full triplet.\n"
        #            f"The solution involves cards at indices {sol_ids} (0‑based). "
        #            f"Hint level = {level}.")

        sol_card_1_idx, sol_card_2_idx, sol_card_3_idx = sol_ids

        base_prompt = (
            f"Hello! you are a 'Set' card game tutor. This game involves finding a 'Set' of three cards.\n"
            f"Each card has four features: shape, color, number, and shading, with three possibilities for each feature.\n"
            f"A 'Set' consists of three cards where, for each of the four features, the attributes are either all the same on all three cards, or all different on all three cards.\n"
            f"There are currently 12 cards on the board, indexed from 0 to 11.\n\n"
            f"There is at least one 'Set' on the board!\n"
            f"To help you provide the best hint, one such 'Set' involves the cards at indices "
            f"{sol_card_1_idx}, {sol_card_2_idx}, and {sol_card_3_idx}. Do not directly reveal all three of these indices to the user unless the hint level specifically allows it."
        )

        hint_instructions = f"Your task is to give the user a friendly and helpful hint based on the current hint level. Please try to avoid giving away the full solution directly, unless the hint level is very high."

        if level == 0:
            hint_instructions += (
                f"Current Hint Level is 0 which means very gentle hint \n\n"
                f"Instruction: Politely inform the user that there is indeed at least one 'Set' to be found on the board and encourage them to keep looking.\n"
                f"Example: 'It looks like there's at least one Set hiding on the board! Keep up the great search please!'"
            )
        elif level == 1:
            hint_instructions += (
                f"Current Hint level is 1 ,which means you should suggest a direction or focus\n"
                f"**instructions**: you should: Gently guide the user to focus on ONE of the cards from the solution. \n"
                f"For example: 'Perhaps you could start by taking a closer look at the card at position {sol_card_1_idx}?' or 'The card at position {sol_card_2_idx} might offer a good starting point.'"
                f"**Please note**, only mention one card's specific index at a time."
            )
        elif level == 2:
            hint_instructions += (
                f"Current Hint Level is 2 which reveal 2wo cards.\n\n"
                f"**Instruction**: you should tell the user the indices of two of the cards that form the 'Set'.\n"
                f"Example: 'I've taken a peek for you, and the cards at positions {sol_card_1_idx} and {sol_card_2_idx} are part of a Set. Can you find the third one?'\n"
            )
        else:
            hint_instructions += (
                f"Current Hint Level is {level} ,this level means more specific help\n"
                f"Instruction: Since the hint level is very high(e.g., 3 or more), you can offer more direct assistance. For instance,"
                f" you could consider revealing the full 'Set' as a final step, but must ensure your tone remains helpful and guiding.\n"
                f"Example: 'Alright, let's narrow it down: the cards at {sol_card_1_idx} and {sol_card_2_idx} are part of a Set. The third card you're looking for is {sol_card_1_idx}'"
            )
        final_prompt = base_prompt + hint_instructions + "output your hint now.\n"

        rsp = openai.ChatCompletion.create(
            model="gpt-4o-mini",          # cheap & fast
            messages=[{"role":"system","content":final_prompt}]
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


