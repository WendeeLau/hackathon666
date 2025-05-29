# main.py
import os, uuid, logging, traceback, requests
from itertools import combinations
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

# Optional: Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# 0.  FastAPI setup & in‑memory state
# ───────────────────────────────────────────────
app = FastAPI(title="Minimal‑Set‑Backend")

SESSIONS: Dict[str, dict] = {}    # {session_id: {"cards": [...],
                                   #               "solution": (i,j,k),
                                   #               "conversation": []}}

# ───────────────────────────────────────────────
# 1.  Helper functions  (no type hints required)
# ───────────────────────────────────────────────
def decode(card):
    """e.g. 1231 → (0,1,2,0)"""
    digs = [int(d) - 1 for d in f"{card:04}"]
    return tuple(digs)

def is_set(a, b, c):
    """classic mod‑3 rule: sum of each attribute ≡ 0 (mod 3)"""
    return all((a[i] + b[i] + c[i]) % 3 == 0 for i in range(4))

def first_solution(cards):
    decoded = list(map(decode, cards))
    for i, j, k in combinations(range(len(cards)), 3):
        if is_set(decoded[i], decoded[j], decoded[k]):
            return i, j, k
    return None

def decode_card_features(card_number):
    """Convert card number to human-readable features"""
    card_str = f"{card_number:04d}"
    
    # Feature mappings based on your Swift code
    numbers = ["one", "two", "three"]
    colors = ["red", "green", "purple"] 
    shadings = ["solid", "striped", "open"]
    shapes = ["squiggle", "diamond", "oval"]
    
    # Extract features (convert 1-3 to 0-2 indexing)
    try:
        number = numbers[int(card_str[0]) - 1]
        color = colors[int(card_str[1]) - 1]
        shading = shadings[int(card_str[2]) - 1]
        shape = shapes[int(card_str[3]) - 1]
        
        return {
            "number": number,
            "color": color, 
            "shading": shading,
            "shape": shape,
            "description": f"{number} {color} {shading} {shape}{'s' if number != 'one' else ''}"
        }
    except (ValueError, IndexError):
        return {
            "number": "unknown",
            "color": "unknown",
            "shading": "unknown", 
            "shape": "unknown",
            "description": f"card {card_number}"
        }

def analyze_set_pattern(card1, card2, card3):
    """Analyze what makes three cards form a valid set"""
    features1 = decode_card_features(card1)
    features2 = decode_card_features(card2)
    features3 = decode_card_features(card3)
    
    patterns = []
    
    # Check each attribute
    for attr in ["number", "color", "shading", "shape"]:
        values = [features1[attr], features2[attr], features3[attr]]
        
        if len(set(values)) == 1:  # All same
            patterns.append(f"all have the same {attr} ({values[0]})")
        elif len(set(values)) == 3:  # All different
            patterns.append(f"all have different {attr}s ({', '.join(sorted(set(values)))})")
    
    return {
        "cards": [features1, features2, features3],
        "patterns": patterns,
        "descriptions": [f["description"] for f in [features1, features2, features3]],
        "features_list": [features1, features2, features3]
    }

# ───────────────────────────────────────────────
# 2.  HKBU ChatGPT wrapper (based on working code)
# ───────────────────────────────────────────────
API_KEY     = os.getenv("OPENAI_API_KEY")
BASIC_URL   = os.getenv("OPENAI_API_BASE", "https://genai.hkbu.edu.hk/general/rest")
MODEL_NAME  = os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-4-o-mini")
API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")

# Debug environment variables
print("=== HKBU ChatGPT Configuration ===")
print(f"API_KEY: {'✓ Set' if API_KEY else '✗ Missing'}")
if API_KEY:
    print(f"Key length: {len(API_KEY)} chars, starts with: {API_KEY[:10]}...")
print(f"BASIC_URL: {BASIC_URL}")
print(f"MODEL_NAME: {MODEL_NAME}")
print(f"API_VERSION: {API_VERSION}")

# Construct the full URL
if API_KEY and BASIC_URL:
    FULL_URL = f"{BASIC_URL}/deployments/{MODEL_NAME}/chat/completions/?api-version={API_VERSION}"
    print(f"Full URL: {FULL_URL}")
    
    # Test the connection
    LLM_OK = False
    try:
        print("🧪 Testing HKBU API connection...")
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': API_KEY
        }
        
        test_payload = {
            'messages': [{"role": "user", "content": "Say 'test'"}],
            'max_tokens': 5
        }
        
        response = requests.post(FULL_URL, json=test_payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            test_result = data['choices'][0]['message']['content']
            print(f"✅ HKBU API connection successful! Response: {test_result}")
            LLM_OK = True
        else:
            print(f"❌ HKBU API test failed with status {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            LLM_OK = False
            
    except Exception as e:
        print(f"❌ HKBU API connection test failed: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        LLM_OK = False
else:
    print("❌ Missing required configuration")
    LLM_OK = False

print(f"Final LLM_OK status: {LLM_OK}")
print("=" * 60)

def call_hkbu_chatgpt(messages):
    """Call HKBU ChatGPT API with conversation history"""
    if not LLM_OK:
        return None
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'api-key': API_KEY
        }
        
        payload = {
            'messages': messages,
            'max_tokens': 150,
            'temperature': 0.7
        }
        
        response = requests.post(FULL_URL, json=payload, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            logger.error(f"HKBU API call failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"HKBU API call exception: {e}")
        return None

def generate_natural_response(user_message, cards, sol_ids, conversation_history):
    """Generate a natural conversational response"""
    
    # Get solution information
    sol_cards = [cards[i] for i in sol_ids]
    set_analysis = analyze_set_pattern(sol_cards[0], sol_cards[1], sol_cards[2])
    solution_features = set_analysis["features_list"]
    
    # Create board description
    all_card_descriptions = []
    for i, card in enumerate(cards):
        features = decode_card_features(card)
        all_card_descriptions.append(f"Position {i}: {features['description']}")
    
    # Build conversation context for LLM
    system_message = {
        "role": "system",
        "content": f"""You are a helpful SET card game assistant. You're having a natural conversation with a player.

CURRENT BOARD:
{chr(10).join(all_card_descriptions)}

THE SOLUTION (keep this secret unless directly asked):
- Card 1: {set_analysis['descriptions'][0]}
- Card 2: {set_analysis['descriptions'][1]} 
- Card 3: {set_analysis['descriptions'][2]}
- Why it works: {' and '.join(set_analysis['patterns'])}

SOLUTION FEATURES:
- Shapes: {[f['shape'] for f in solution_features]}
- Colors: {[f['color'] for f in solution_features]}
- Numbers: {[f['number'] for f in solution_features]}
- Shadings: {[f['shading'] for f in solution_features]}

CONVERSATION GUIDELINES:
- Be natural and conversational
- Answer exactly what the user asks (no more, no less)
- If asked about a specific feature (like "is there a square"), give a direct yes/no answer
- If asked for hints, provide helpful but not overwhelming clues
- Don't repeat information you've already shared
- Keep responses concise and friendly
- If user seems stuck, offer gentle encouragement"""
    }
    
    # Build message history
    messages = [system_message]
    
    # Add conversation history
    for exchange in conversation_history:
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    if LLM_OK:
        logger.info(f"🤖 Making natural conversation call...")
        llm_response = call_hkbu_chatgpt(messages)
        
        if llm_response:
            logger.info(f"✅ Natural response generated")
            return llm_response
        else:
            logger.warning("❌ LLM call failed, using fallback")
    
    # Simple rule-based fallback for specific questions
    user_msg_lower = user_message.lower()
    
    # Check for specific feature questions
    if "square" in user_msg_lower:
        shapes_in_solution = [f['shape'] for f in solution_features]
        if "square" in shapes_in_solution:
            return "Yes, there is a square in the solution."
        else:
            return "No, there are no squares in the solution."
    
    # Check for other shapes
    for shape in ["squiggle", "diamond", "oval"]:
        if shape in user_msg_lower and ("is there" in user_msg_lower or "any" in user_msg_lower):
            shapes_in_solution = [f['shape'] for f in solution_features]
            if shape in shapes_in_solution:
                return f"Yes, there {'is a' if shapes_in_solution.count(shape) == 1 else 'are'} {shape} in the solution."
            else:
                return f"No, there are no {shape}s in the solution."
    
    # Check for colors
    for color in ["red", "green", "purple"]:
        if color in user_msg_lower and ("is there" in user_msg_lower or "any" in user_msg_lower):
            colors_in_solution = [f['color'] for f in solution_features]
            if color in colors_in_solution:
                return f"Yes, there {'is a' if colors_in_solution.count(color) == 1 else 'are'} {color} card in the solution."
            else:
                return f"No, there are no {color} cards in the solution."
    
    # General hint
    return "I can help you find the set! What would you like to know?"

# ───────────────────────────────────────────────
# 3.  API contracts (Pydantic needs field types)
# ───────────────────────────────────────────────
class BoardIn(BaseModel):
    cards: List[int]

class MsgIn(BaseModel):
    message: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "llm_available": LLM_OK,
        "api_type": "HKBU ChatGPT Natural Conversation",
        "model_name": MODEL_NAME
    }

@app.get("/")
def health_check():
    return {
        "Haha!s"
    }

# POST /new‑game  → start a session
@app.post("/new-game")
def new_game(inp: BoardIn):
    if len(inp.cards) < 3:
        raise HTTPException(400, "Need at least three cards.")
    
    logger.info(f"Starting new game with {len(inp.cards)} cards")
    sol = first_solution(inp.cards)
    sess_id = str(uuid.uuid4())
    
    SESSIONS[sess_id] = {
        "cards": inp.cards,
        "solution": sol,
        "conversation": []  # Store conversation history
    }
    
    if sol is None:
        hint = "I don't see a valid Set on this board. You might need to draw new cards."
        has_set = False
    else:
        hint = "Great! I can see there's a valid Set on the board. Ask me anything to help you find it!"
        has_set = True
    
    logger.info(f"Session {sess_id} created, has_set: {has_set}")
    
    return {
        "session": sess_id,
        "hasSet": has_set,
        "hint": hint,
        "llm_used": False
    }

# POST /chat/{session_id}  → continue the conversation
@app.post("/chat/{sid}")
def chat(sid: str, inp: MsgIn):
    if sid not in SESSIONS:
        raise HTTPException(404, "Unknown session.")
    
    sess = SESSIONS[sid]
    
    logger.info(f"Chat request for session {sid}, message: '{inp.message}'")
    
    if sess["solution"] is None:
        reply = "There's no Set on the current board. Would you like to try with different cards?"
    else:
        # Generate natural conversational response
        reply = generate_natural_response(
            inp.message, 
            sess["cards"], 
            sess["solution"], 
            sess["conversation"]
        )
        
        # Store this exchange in conversation history
        sess["conversation"].append({
            "user": inp.message,
            "assistant": reply
        })
        
        # Keep only last 10 exchanges to prevent memory issues
        if len(sess["conversation"]) > 10:
            sess["conversation"] = sess["conversation"][-10:]
    
    return JSONResponse({
        "assistant": reply,
        "llm_used": LLM_OK
    })

# Debug endpoint
@app.get("/debug/llm")
def debug_llm():
    return {
        "llm_ok": LLM_OK,
        "api_key_set": bool(API_KEY),
        "basic_url": BASIC_URL,
        "model_name": MODEL_NAME,
        "api_version": API_VERSION,
        "full_url": FULL_URL if API_KEY and BASIC_URL else None,
        "api_type": "HKBU ChatGPT Natural"
    }

# Test endpoint
@app.post("/test-llm")
def test_llm(inp: MsgIn):
    if not LLM_OK:
        raise HTTPException(503, "HKBU ChatGPT API not available")
    
    messages = [{"role": "user", "content": inp.message}]
    response = call_hkbu_chatgpt(messages)
    if response:
        return {"response": response, "status": "success"}
    else:
        raise HTTPException(500, "HKBU ChatGPT API call failed")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)