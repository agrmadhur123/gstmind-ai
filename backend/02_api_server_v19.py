"""
GSTMind Backend API Server
Built with FastAPI + Supabase
Deploy on Railway or Render (both free tier available)
"""

from fastapi import FastAPI, HTTPException, Header, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse as FastAPIStreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import httpx
import json
import random
import string
from datetime import datetime, timedelta
import hashlib
import hmac
import re
import math

app = FastAPI(title="GSTMind API", version="1.0.0")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── ENV VARS (set these in Railway/Render dashboard) ──
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_KEY = (os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
CLAUDE_API_KEY = (os.getenv("CLAUDE_API_KEY") or "").strip()
RAZORPAY_KEY_ID = (os.getenv("RAZORPAY_KEY_ID") or "").strip()
RAZORPAY_KEY_SECRET = (os.getenv("RAZORPAY_KEY_SECRET") or "").strip()
MSG91_AUTH_KEY = (os.getenv("MSG91_AUTH_KEY") or "").strip()
MSG91_TEMPLATE_ID = (os.getenv("MSG91_TEMPLATE_ID") or "").strip()
RESEND_API_KEY = (os.getenv("RESEND_API_KEY") or "").strip()
FROM_EMAIL = (os.getenv("FROM_EMAIL") or "noreply@gstmind.in").strip()
# Test account — remove before public launch
TEST_EMAIL = (os.getenv("TEST_EMAIL") or "test@gstmind.in").strip()
TEST_PASSWORD = (os.getenv("TEST_PASSWORD") or "").strip()
# Admin panel secret — same password the admin uses to log into admin.html
ADMIN_SECRET = (os.getenv("ADMIN_SECRET") or "gstmind@admin123").strip()
JWT_SECRET = (os.getenv("JWT_SECRET") or "gstmind-secret-change-this").strip()

# ════════════════════════════════════════════════════════════════
# SERVER-SIDE CHUNKING & BM25 RETRIEVAL
# ════════════════════════════════════════════════════════════════

GST_SYNONYMS = {
    'itc': ['input tax credit', 'input credit', 'credit'],
    'rcm': ['reverse charge', 'reverse charge mechanism'],
    'refund': ['refund claim', 'excess credit', 'accumulated credit'],
    'export': ['zero rated', 'lut', 'letter of undertaking'],
    'registration': ['gstin', 'gst number'],
    'invoice': ['tax invoice', 'bill', 'debit note', 'credit note'],
    'demand': ['show cause', 'scn', 'adjudication'],
    'penalty': ['penalty', 'fine', 'late fee', 'interest'],
    'composition': ['composition scheme', 'composition dealer'],
    'section 16': ['itc eligibility', 'input tax credit', '16(2)', '16(4)'],
    'section 73': ['demand non fraud', '3 year', 'normal period'],
    'section 74': ['demand fraud', '5 year', 'wilful misstatement'],
    'section 54': ['refund', 'two years', 'relevant date'],
}

STOP_WORDS = {
    'the','and','for','not','any','all','been','into','also','both',
    'its','their','this','that','with','from','have','will','were',
    'was','are','would','could','being','there','here','then','than',
    'thus','said','each','some','over','upon','very','well','even',
}

def tokenize_text(text: str) -> list:
    """Tokenize text for BM25 scoring."""
    tokens = re.findall(r'[a-zA-Z0-9()\/-]+', text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]

def expand_query(q: str) -> set:
    """Expand query with GST synonyms."""
    terms = set(tokenize_text(q))
    q_lower = q.lower()
    for key, syns in GST_SYNONYMS.items():
        if key in q_lower:
            for syn in syns:
                terms.update(tokenize_text(syn))
    return terms

def bm25_score(query_terms: set, doc_tokens: list, avg_len: float) -> float:
    """BM25 scoring."""
    k1, b = 1.5, 0.75
    tf_map = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1
    
    score = 0.0
    doc_len = len(doc_tokens)
    for term in query_terms:
        tf = tf_map.get(term, 0)
        if tf == 0:
            # Check partial match
            partials = [v for k, v in tf_map.items() if term in k or k in term]
            tf = len(partials) * 0.3
        if tf == 0:
            continue
        norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avg_len, 1)))
        score += norm_tf
    return score

def semantic_chunk(text: str, doc_name: str, doc_type: str, doc_date: str) -> list:
    """Split document into semantic chunks at paragraph boundaries."""
    # Split at double newlines (paragraph breaks)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip() and len(p.strip()) > 60]
    
    chunks = []
    chunk_idx = 0
    
    for para in paragraphs:
        words = para.split()
        if len(words) <= 250:
            # Small paragraph — use as-is
            chunks.append({
                'doc_name': doc_name,
                'doc_type': doc_type,
                'doc_date': doc_date,
                'chunk_text': para,
                'chunk_index': chunk_idx,
                'word_count': len(words)
            })
            chunk_idx += 1
        else:
            # Large paragraph — split at sentence boundaries with overlap
            sentences = re.split(r'(?<=[.!?])\s+', para)
            window, window_words = [], 0
            
            for sent in sentences:
                sent_words = sent.split()
                window.append(sent)
                window_words += len(sent_words)
                
                if window_words >= 200:
                    chunk_text = ' '.join(window)
                    chunks.append({
                        'doc_name': doc_name,
                        'doc_type': doc_type,
                        'doc_date': doc_date,
                        'chunk_text': chunk_text,
                        'chunk_index': chunk_idx,
                        'word_count': window_words
                    })
                    chunk_idx += 1
                    # Overlap — keep last 2 sentences
                    window = window[-2:] if len(window) >= 2 else window[-1:]
                    window_words = sum(len(s.split()) for s in window)
            
            # Remaining sentences
            if window and window_words > 30:
                chunks.append({
                    'doc_name': doc_name,
                    'doc_type': doc_type,
                    'doc_date': doc_date,
                    'chunk_text': ' '.join(window),
                    'chunk_index': chunk_idx,
                    'word_count': window_words
                })
                chunk_idx += 1
    
    return chunks

def compress_chunk(text: str) -> str:
    """Compress chunk text to save tokens."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('of the Central Goods and Services Tax Act, 2017', 'CGST Act')
    text = text.replace('of the Integrated Goods and Services Tax Act, 2017', 'IGST Act')
    text = text.replace('of the CGST Act, 2017', 'CGST Act')
    text = text.replace('of the IGST Act, 2017', 'IGST Act')
    text = re.sub(r'input tax credit', 'ITC', text, flags=re.IGNORECASE)
    text = re.sub(r'reverse charge mechanism', 'RCM', text, flags=re.IGNORECASE)
    return text.strip()



SUPABASE_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json"
}

# ════════════════════════════════
# HELPERS
# ════════════════════════════════
async def supabase_query(method: str, table: str, data=None, filters=""):
    url = f"{SUPABASE_URL}/rest/v1/{table}{filters}"
    headers = {**SUPABASE_HEADERS}
    if method in ("POST", "PATCH"):
        headers["Prefer"] = "return=representation"
    async with httpx.AsyncClient(timeout=60) as client:
        if method == "GET":
            r = await client.get(url, headers=headers)
        elif method == "POST":
            r = await client.post(url, headers=headers, json=data)
        elif method == "PATCH":
            r = await client.patch(url, headers=headers, json=data)
        elif method == "DELETE":
            # Add Prefer header so Supabase returns deleted rows (confirms delete worked)
            del_headers = {**headers, "Prefer": "return=representation"}
            r = await client.delete(url, headers=del_headers)
        if r.status_code in (200, 201):
            try:
                result = r.json()
                return result if isinstance(result, list) else [result]
            except:
                return []
        elif r.status_code == 204:
            return []
        else:
            print(f"Supabase error {r.status_code} on {table}: {r.text[:300]}")
            raise HTTPException(status_code=500, detail=f"Database error {r.status_code}: {r.text[:200]}")

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def simple_token(user_id: str):
    import base64, time
    payload = f"{user_id}:{int(time.time())}"
    sig = hmac.new(JWT_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token = base64.b64encode(f"{payload}:{sig}".encode()).decode()
    return token

def verify_token(token: str):
    import base64, time
    try:
        decoded = base64.b64decode(token).decode()
        parts = decoded.rsplit(":", 1)
        payload, sig = parts[0], parts[1]
        expected = hmac.new(JWT_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if sig != expected:
            return None
        user_id, ts = payload.split(":", 1)
        # Token valid for 30 days
        if int(time.time()) - int(ts) > 30 * 24 * 3600:
            return None
        return user_id
    except:
        return None

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "")
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id

# ════════════════════════════════
# AUTH ENDPOINTS
# ════════════════════════════════
class SignupEmail(BaseModel):
    fname: str
    lname: str
    email: str
    password: str
    profession: str

class LoginEmail(BaseModel):
    email: str
    password: str

class SendOTP(BaseModel):
    phone: str

class VerifyOTP(BaseModel):
    phone: str
    code: str
    fname: Optional[str] = None
    lname: Optional[str] = None
    profession: Optional[str] = None
    is_signup: bool = False

@app.post("/auth/signup/email")
async def signup_email(data: SignupEmail):
    # Check if email exists
    existing = await supabase_query("GET", "users", filters=f"?email=eq.{data.email}")
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    # Hash password
    pwd_hash = hashlib.sha256(data.password.encode()).hexdigest()
    user = {
        "fname": data.fname,
        "lname": data.lname,
        "name": f"{data.fname} {data.lname}",
        "email": data.email,
        "password_hash": pwd_hash,
        "profession": data.profession,
        "login_method": "email",
        "plan": "free",
        "query_count": 0,
        "query_limit": 10
    }
    result = await supabase_query("POST", "users", data=user)
    if isinstance(result, list) and result:
        user_id = result[0]["id"]
        token = simple_token(user_id)
        return {"token": token, "user": result[0]}
    raise HTTPException(status_code=500, detail="Signup failed")

@app.post("/auth/login/email")
async def login_email(data: LoginEmail):
    pwd_hash = hashlib.sha256(data.password.encode()).hexdigest()
    users = await supabase_query("GET", "users",
        filters=f"?email=eq.{data.email}&password_hash=eq.{pwd_hash}")
    if not users:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user = users[0]
    if user.get("blocked"):
        raise HTTPException(status_code=403, detail="Account suspended")
    token = simple_token(user["id"])
    return {"token": token, "user": user}


async def send_email_otp(to_email: str, otp: str, purpose: str = "login") -> bool:
    """Send OTP via Resend email API. Returns True if sent."""
    if not RESEND_API_KEY:
        print(f"[DEV MODE] Email OTP for {to_email}: {otp}")
        return False  # Return False so caller knows to include dev_otp in response
    
    subject = "GSTMind — Your OTP" if purpose == "login" else "GSTMind — Password Reset OTP"
    body = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:24px">
      <h2 style="color:#c9a96e">GSTMind</h2>
      <p>Your one-time password is:</p>
      <div style="font-size:36px;font-weight:700;letter-spacing:10px;color:#2a1f14;padding:20px;background:#faf8f5;border-radius:8px;text-align:center;font-family:monospace">{otp}</div>
      <p style="color:#666;font-size:13px;margin-top:16px">Valid for 10 minutes. Do not share with anyone.</p>
      <hr style="border:none;border-top:1px solid #eee;margin:20px 0">
      <p style="color:#999;font-size:11px">GSTMind — AI-powered GST Research for CAs</p>
    </div>
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                json={"from": FROM_EMAIL, "to": [to_email], "subject": subject, "html": body}
            )
            if r.status_code in (200, 201):
                print(f"Email OTP sent to {to_email}")
                return True
            print(f"Resend error {r.status_code}: {r.text[:300]}")
            return False
    except Exception as e:
        print(f"Email send error: {e}")
        return False

@app.post("/auth/otp/send")
async def send_otp(data: SendOTP):
    code = generate_otp()
    expires = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
    # Save OTP to database
    await supabase_query("POST", "otp_codes", data={
        "phone": data.phone, "code": code, "expires_at": expires
    })
    # Send via MSG91
    if MSG91_AUTH_KEY:
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://api.msg91.com/api/v5/otp",
                params={
                    "authkey": MSG91_AUTH_KEY,
                    "mobile": f"91{data.phone}",
                    "template_id": MSG91_TEMPLATE_ID,
                    "otp": code
                }
            )
    else:
        # Development mode — return OTP in response
        return {"message": "OTP sent", "dev_otp": code}
    return {"message": "OTP sent to your mobile number"}

@app.post("/auth/otp/verify")
async def verify_otp(data: VerifyOTP):
    now = datetime.utcnow().isoformat()
    otps = await supabase_query("GET", "otp_codes",
        filters=f"?phone=eq.{data.phone}&code=eq.{data.code}&used=eq.false&expires_at=gt.{now}")
    if not otps:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    # Mark OTP as used
    await supabase_query("PATCH", "otp_codes",
        data={"used": True}, filters=f"?id=eq.{otps[0]['id']}")
    if data.is_signup:
        # Create new user
        user_data = {
            "fname": data.fname, "lname": data.lname,
            "name": f"{data.fname} {data.lname}",
            "phone": data.phone, "profession": data.profession,
            "login_method": "otp", "plan": "free",
            "query_count": 0, "query_limit": 10
        }
        result = await supabase_query("POST", "users", data=user_data)
        if isinstance(result, list) and result:
            token = simple_token(result[0]["id"])
            return {"token": token, "user": result[0]}
    else:
        # Login existing user
        users = await supabase_query("GET", "users", filters=f"?phone=eq.{data.phone}")
        if not users:
            raise HTTPException(status_code=404, detail="No account found. Please sign up first.")
        user = users[0]
        if user.get("blocked"):
            raise HTTPException(status_code=403, detail="Account suspended")
        token = simple_token(user["id"])
        return {"token": token, "user": user}
    raise HTTPException(status_code=500, detail="Verification failed")

@app.get("/auth/me")
async def get_me(user_id: str = Depends(get_current_user)):
    # Test user — return fake profile without DB lookup
    if user_id == "test-user-000":
        return {
            "id": "test-user-000",
            "fname": "Test", "lname": "User", "name": "Test User",
            "email": TEST_EMAIL, "plan": "unlimited",
            "query_count": 0, "query_limit": 99999,
        }
    users = await supabase_query("GET", "users", filters=f"?id=eq.{user_id}")
    if not users:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    return users[0]

# ════════════════════════════════
# QUERY ENDPOINT
# ════════════════════════════════
class QueryRequest(BaseModel):
    messages: list
    system: str = ""      # Used for draft mode custom prompt
    mode: str = "query"
    query_text: str = ""
    is_followup: bool = False
    has_attachment: bool = False



class ForgotSend(BaseModel):
    email: str

class ForgotVerify(BaseModel):
    email: str
    otp: str

class ForgotReset(BaseModel):
    email: str
    new_password: str

@app.post("/auth/forgot/send")
async def forgot_send(data: ForgotSend):
    """Send OTP to email for password reset."""
    email = data.email.lower().strip()
    users = await supabase_query("GET", "users", filters=f"?email=eq.{email}")
    if not users:
        # Don't reveal if email exists — security best practice
        return {"message": "If this email is registered, an OTP has been sent."}
    
    code = generate_otp()
    expires = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    
    # Store OTP using email as identifier
    await supabase_query("POST", "otp_codes", data={
        "phone": f"email:{email}",  # reuse phone field with prefix
        "code": code,
        "expires_at": expires
    })
    
    sent = await send_email_otp(email, code, purpose="reset")
    if not sent and RESEND_API_KEY:
        raise HTTPException(status_code=500, detail="Failed to send OTP email. Please try again.")
    
    response = {"message": "OTP sent to your email" if sent else "Dev mode: OTP shown below"}
    if not sent:
        response["dev_otp"] = code  # Show OTP on screen when email not configured
    return response

@app.post("/auth/forgot/verify")
async def forgot_verify(data: ForgotVerify):
    """Verify OTP for password reset."""
    email = data.email.lower().strip()
    now = datetime.utcnow().isoformat()
    otps = await supabase_query("GET", "otp_codes",
        filters=f"?phone=eq.email:{email}&code=eq.{data.otp}&used=eq.false&expires_at=gt.{now}")
    if not otps:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    # Don't mark as used yet — mark when password is actually reset
    return {"message": "OTP verified"}

@app.post("/auth/forgot/reset")
async def forgot_reset(data: ForgotReset):
    """Reset password after OTP verification."""
    email = data.email.lower().strip()
    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    
    # Verify OTP is still valid
    now = datetime.utcnow().isoformat()
    otps = await supabase_query("GET", "otp_codes",
        filters=f"?phone=eq.email:{email}&used=eq.false&expires_at=gt.{now}")
    if not otps:
        raise HTTPException(status_code=400, detail="OTP expired. Please request a new one.")
    
    # Hash new password
    new_hash = hashlib.sha256(data.new_password.encode()).hexdigest()
    
    result = await supabase_query("PATCH", "users",
        data={"password_hash": new_hash},
        filters=f"?email=eq.{email}")
    
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Mark OTP as used
    await supabase_query("PATCH", "otp_codes",
        data={"used": True}, filters=f"?id=eq.{otps[0]['id']}")
    
    return {"message": "Password reset successfully"}

@app.post("/auth/email/otp/send")
async def send_email_otp_endpoint(data: ForgotSend):
    """Send login/signup OTP to email."""
    email = data.email.lower().strip()
    code = generate_otp()
    expires = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    
    await supabase_query("POST", "otp_codes", data={
        "phone": f"email:{email}",
        "code": code,
        "expires_at": expires
    })
    
    sent = await send_email_otp(email, code, purpose="login")
    if not sent and RESEND_API_KEY:
        raise HTTPException(status_code=500, detail="Failed to send OTP email. Please try again.")
    
    response = {"message": "OTP sent to your email" if sent else "Dev mode active"}
    if not sent:
        response["dev_otp"] = code  # Show on screen when Resend not configured
    return response

@app.post("/auth/email/otp/verify")
async def verify_email_otp(data: VerifyOTP):
    """Verify email OTP for login or signup."""
    email = data.phone  # phone field carries email with prefix
    if not email.startswith("email:"):
        email = f"email:{email}"
    now = datetime.utcnow().isoformat()
    otps = await supabase_query("GET", "otp_codes",
        filters=f"?phone=eq.{email}&code=eq.{data.code}&used=eq.false&expires_at=gt.{now}")
    if not otps:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    await supabase_query("PATCH", "otp_codes", data={"used": True}, filters=f"?id=eq.{otps[0]['id']}")
    
    clean_email = email.replace("email:", "")
    if data.is_signup:
        user_data = {
            "fname": data.fname, "lname": data.lname,
            "name": f"{data.fname} {data.lname}",
            "email": clean_email, "profession": data.profession,
            "login_method": "email_otp", "plan": "free",
            "query_count": 0, "query_limit": 10
        }
        result = await supabase_query("POST", "users", data=user_data)
        if isinstance(result, list) and result:
            token = simple_token(result[0]["id"])
            return {"token": token, "user": result[0]}
    else:
        users = await supabase_query("GET", "users", filters=f"?email=eq.{clean_email}")
        if not users:
            raise HTTPException(status_code=404, detail="No account found. Please sign up first.")
        user = users[0]
        token = simple_token(user["id"])
        return {"token": token, "user": user}
    raise HTTPException(status_code=500, detail="Verification failed")



class SetPassword(BaseModel):
    email: str
    password: str

@app.post("/auth/set-password")
async def set_password(data: SetPassword):
    """Set password after email OTP signup verification."""
    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    pw_hash = hashlib.sha256(data.password.encode()).hexdigest()
    result = await supabase_query("PATCH", "users",
        data={"password_hash": pw_hash},
        filters=f"?email=eq.{data.email.lower().strip()}")
    return {"message": "Password set"}



class TestLogin(BaseModel):
    email: str
    password: str

@app.post("/auth/test/login")
async def test_login(data: TestLogin):
    """Temporary test login — unlimited queries, no OTP, no DB write.
    Only works when TEST_PASSWORD env var is set.
    REMOVE before public launch."""
    if not TEST_PASSWORD:
        raise HTTPException(status_code=404, detail="Not found")
    if data.email.lower().strip() != TEST_EMAIL or data.password != TEST_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid test credentials")
    
    # Return a fake user with unlimited plan — no DB write
    test_user = {
        "id": "test-user-000",
        "fname": "Test",
        "lname": "User",
        "name": "Test User",
        "email": TEST_EMAIL,
        "plan": "unlimited",
        "query_count": 0,
        "query_limit": 99999,
        "query_limit_display": "∞",
    }
    token = simple_token("test-user-000")
    return {"token": token, "user": test_user}


class AdminTokenRequest(BaseModel):
    password: str

@app.post("/auth/admin/token")
async def get_admin_token(data: AdminTokenRequest):
    """Get a Railway API token for admin panel KB uploads.
    Uses the admin panel password — no email required."""
    if not ADMIN_SECRET or data.password != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin password")
    # Return a long-lived admin token (24 hours)
    import base64, time
    payload = f"admin-user:{int(time.time())}"
    sig = hmac.new(JWT_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token = base64.b64encode(f"{payload}:{sig}".encode()).decode()
    return {
        "token": token,
        "user": {
            "id": "admin-user",
            "fname": "Admin",
            "name": "Admin",
            "plan": "unlimited",
            "query_count": 0,
            "query_limit": 99999
        }
    }


# ════════════════════════════════════════════════════════
# ADMIN AUTH ENDPOINTS
# ════════════════════════════════════════════════════════

ADMIN_INVITE_CODE = (os.getenv("ADMIN_INVITE_CODE") or "GSTMIND-ADMIN-2024").strip()

class AdminSignup(BaseModel):
    fname: str
    lname: str
    email: str
    invite_code: str

class AdminLogin(BaseModel):
    email: str

class AdminOTPVerify(BaseModel):
    email: str
    otp: str

@app.post("/auth/admin/signup")
async def admin_signup(data: AdminSignup):
    """Admin signup — requires invite code. Sends OTP to email."""
    if data.invite_code != ADMIN_INVITE_CODE:
        raise HTTPException(status_code=403, detail="Invalid invite code")
    
    email = data.email.lower().strip()
    
    # Check if already exists
    existing = await supabase_query("GET", "users", filters=f"?email=eq.{email}")
    if existing:
        # Already exists — just send OTP to login
        code = generate_otp()
        expires = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
        await supabase_query("POST", "otp_codes", data={
            "phone": f"email:{email}", "code": code, "expires_at": expires
        })
        sent = await send_email_otp(email, code, purpose="login")
        resp = {"message": "Account exists — OTP sent to login", "exists": True}
        if not sent: resp["dev_otp"] = code
        return resp
    
    # Create admin user
    user_data = {
        "fname": data.fname, "lname": data.lname,
        "name": f"{data.fname} {data.lname}",
        "email": email, "profession": "Admin",
        "login_method": "email_otp", "plan": "unlimited",
        "query_count": 0, "query_limit": 999999,
        "is_admin": True
    }
    result = await supabase_query("POST", "users", data=user_data)
    if not isinstance(result, list) or not result:
        raise HTTPException(status_code=500, detail="Failed to create admin account")
    
    # Send OTP
    code = generate_otp()
    expires = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    await supabase_query("POST", "otp_codes", data={
        "phone": f"email:{email}", "code": code, "expires_at": expires
    })
    sent = await send_email_otp(email, code, purpose="login")
    resp = {"message": "Admin account created — OTP sent to email", "exists": False}
    if not sent: resp["dev_otp"] = code
    return resp

@app.post("/auth/admin/otp/send")
async def admin_otp_send(data: AdminLogin):
    """Send login OTP to admin email."""
    email = data.email.lower().strip()
    
    # Verify this is an admin account
    users = await supabase_query("GET", "users", filters=f"?email=eq.{email}&is_admin=eq.true")
    if not users:
        raise HTTPException(status_code=403, detail="No admin account found for this email")
    
    code = generate_otp()
    expires = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    await supabase_query("POST", "otp_codes", data={
        "phone": f"email:{email}", "code": code, "expires_at": expires
    })
    sent = await send_email_otp(email, code, purpose="login")
    resp = {"message": "OTP sent to your email"}
    if not sent: resp["dev_otp"] = code
    return resp

@app.post("/auth/admin/otp/verify")
async def admin_otp_verify(data: AdminOTPVerify):
    """Verify OTP and return admin token."""
    email = data.email.lower().strip()
    now = datetime.utcnow().isoformat()
    
    otps = await supabase_query("GET", "otp_codes",
        filters=f"?phone=eq.email:{email}&code=eq.{data.otp}&used=eq.false&expires_at=gt.{now}")
    if not otps:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    await supabase_query("PATCH", "otp_codes", data={"used": True},
        filters=f"?id=eq.{otps[0]['id']}")
    
    users = await supabase_query("GET", "users", filters=f"?email=eq.{email}&is_admin=eq.true")
    if not users:
        raise HTTPException(status_code=403, detail="Not an admin account")
    
    user = users[0]
    token = simple_token(user["id"])
    return {"token": token, "user": user}

@app.get("/auth/admin/me")
async def admin_me(user_id: str = Depends(get_current_user)):
    """Verify admin session."""
    if user_id == "test-user-000":
        raise HTTPException(status_code=403, detail="Test user cannot access admin")
    users = await supabase_query("GET", "users", filters=f"?id=eq.{user_id}&is_admin=eq.true")
    if not users:
        raise HTTPException(status_code=403, detail="Not an admin account")
    return users[0]

# ════════════════════════════════════════════════════════════════
# KB ENDPOINTS — Document upload, list, delete
# ════════════════════════════════════════════════════════════════

class KBUploadRequest(BaseModel):
    name: str
    type: str = ""
    doc_date: str = ""
    content: str  # Full extracted text from PDF.js
    doc_identity: dict = {}

@app.post("/kb/upload")
async def kb_upload(data: KBUploadRequest, user_id: str = Depends(get_current_user)):
    """Receive extracted text from admin, chunk it, store in Supabase."""
    if not data.content or len(data.content.strip()) < 50:
        raise HTTPException(status_code=400, detail="Document content too short")

    # Create document record
    doc_record = {
        "name": data.name,
        "type": data.type,
        "doc_date": data.doc_date,
        "content": data.content[:5000],  # Store first 5000 chars as preview
        "chunk_count": 0,
        "is_builtin": False,
        "added_by": user_id,
        "file_size": len(data.content),
        "doc_identity": data.doc_identity
    }

    # Insert document
    try:
        doc_result = await supabase_query("POST", "kb_documents", data=doc_record)
    except Exception as e:
        print(f"KB document save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save document: {str(e)[:200]}")
    if not doc_result:
        raise HTTPException(status_code=500, detail="Failed to save document — empty result")
    
    doc_id = doc_result[0]["id"]

    # Generate semantic chunks
    chunks = semantic_chunk(data.content, data.name, data.type, data.doc_date)
    
    if not chunks:
        # Delete the doc if no chunks generated
        await supabase_query("DELETE", "kb_documents", filters=f"?id=eq.{doc_id}")
        raise HTTPException(status_code=400, detail="Could not extract any content from document")

    # Store chunks in batches of 50
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        chunk_records = [
            {
                "doc_id": doc_id,
                "doc_name": ch["doc_name"],
                "doc_type": ch["doc_type"],
                "doc_date": ch["doc_date"],
                "chunk_text": ch["chunk_text"],
                "chunk_index": ch["chunk_index"],
                "word_count": ch["word_count"]
            }
            for ch in batch
        ]
        await supabase_query("POST", "kb_chunks", data=chunk_records)

    # Update chunk count on document
    await supabase_query("PATCH", "kb_documents",
        data={"chunk_count": len(chunks)},
        filters=f"?id=eq.{doc_id}")

    print(f"KB upload: {data.name} → {len(chunks)} chunks stored")
    return {
        "id": doc_id,
        "name": data.name,
        "chunks": len(chunks),
        "words": sum(ch["word_count"] for ch in chunks)
    }


@app.get("/kb/diagnose")
async def kb_diagnose(user_id: str = Depends(get_current_user)):
    """Check KB table structure."""
    try:
        # Try to get table columns
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/kb_documents?limit=0",
                headers=SUPABASE_HEADERS
            )
            return {
                "status": r.status_code,
                "headers": dict(r.headers),
                "body": r.text[:500]
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/kb/preview/{doc_id}")
async def kb_preview(doc_id: str, user_id: str = Depends(get_current_user)):
    """Get first chunk of a document for preview + detect if scanned."""
    chunks = await supabase_query("GET", "kb_chunks",
        filters=f"?doc_id=eq.{doc_id}&order=chunk_index.asc&limit=3&select=chunk_text")
    
    if not chunks:
        return {"preview": "", "is_scanned": True, "chunk_count": 0}
    
    preview = " ".join(ch.get("chunk_text","") for ch in chunks)
    word_count = len(preview.split())
    is_scanned = word_count < 20  # Very low word count = likely scanned
    
    return {
        "preview": preview[:3000],
        "is_scanned": is_scanned,
        "word_count": word_count,
        "chunk_count": len(chunks)
    }

@app.get("/kb/documents")
async def kb_list(user_id: str = Depends(get_current_user)):
    """List all KB documents."""
    docs = await supabase_query("GET", "kb_documents",
        filters="?order=created_at.desc&select=id,name,type,doc_date,chunk_count,file_size,is_builtin,created_at")
    return docs or []


@app.delete("/kb/documents")
async def kb_delete_bulk(user_id: str = Depends(get_current_user), ids: str = ""):
    """Delete multiple documents by comma-separated IDs."""
    if not ids:
        raise HTTPException(status_code=400, detail="No IDs provided")
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    print(f"KB bulk delete: {len(id_list)} docs by user={user_id}")
    deleted = 0
    for doc_id in id_list:
        await supabase_query("DELETE", "kb_chunks", filters=f"?doc_id=eq.{doc_id}")
        await supabase_query("DELETE", "kb_documents", filters=f"?id=eq.{doc_id}")
        deleted += 1
    print(f"Bulk delete complete: {deleted} docs")
    return {"deleted": deleted}

@app.delete("/kb/documents/{doc_id}")
async def kb_delete(doc_id: str, user_id: str = Depends(get_current_user)):
    """Delete a document and all its chunks."""
    print(f"KB delete requested: doc_id={doc_id} by user={user_id}")
    
    # Delete chunks first (explicit, don't rely on cascade alone)
    chunks_deleted = await supabase_query("DELETE", "kb_chunks", filters=f"?doc_id=eq.{doc_id}")
    print(f"Chunks deleted: {len(chunks_deleted) if chunks_deleted else 0}")
    
    # Delete document
    result = await supabase_query("DELETE", "kb_documents", filters=f"?id=eq.{doc_id}")
    print(f"Document deleted: {result}")
    
    return {"deleted": doc_id, "chunks_removed": len(chunks_deleted) if chunks_deleted else 0}

async def retrieve_from_supabase(query_text: str, top_k: int = 10, is_followup: bool = False) -> list:
    """Retrieve relevant chunks from Supabase using BM25. Returns [] on any error."""
    limit = top_k if not is_followup else 6
    
    try:
        result = await supabase_query("GET", "kb_chunks",
            filters=f"?select=doc_name,doc_type,doc_date,chunk_text&limit=2000")
    except Exception as e:
        print(f"KB retrieval error (non-fatal): {e}")
        return []  # Fall back to built-in KB only — don't crash

    if not result:
        return []

    # Expand query terms
    query_terms = expand_query(query_text)
    if not query_terms:
        return []

    # Tokenize all chunks
    tokenized = [(r, tokenize_text(r["chunk_text"])) for r in result]
    avg_len = sum(len(t) for _, t in tokenized) / max(len(tokenized), 1)

    # Score each chunk
    scored = []
    for chunk_data, tokens in tokenized:
        score = bm25_score(query_terms, tokens, avg_len)
        if score > 0:
            scored.append((score, chunk_data))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Apply relevance threshold — only genuinely relevant chunks
    if scored:
        top_score = scored[0][0]
        threshold = top_score * 0.25
        scored = [(s, c) for s, c in scored if s >= threshold or len(scored) <= 3]

    # Deduplicate by doc+content prefix
    seen_keys = set()
    results = []
    for score, chunk in scored[:limit]:
        key = chunk["doc_name"] + chunk["chunk_text"][:50]
        if key not in seen_keys:
            seen_keys.add(key)
            results.append({
                "doc": chunk["doc_name"],
                "docDate": chunk.get("doc_date", ""),
                "docType": chunk.get("doc_type", ""),
                "chunk": chunk["chunk_text"],
                "score": score
            })

    return results

@app.post("/query")
async def run_query(data: QueryRequest, user_id: str = Depends(get_current_user)):
    # Get user
    users = await supabase_query("GET", "users", filters=f"?id=eq.{user_id}")
    if not users:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    user = users[0]

    if user.get("blocked"):
        raise HTTPException(status_code=403, detail="Account suspended")

    # Test user — bypass all limits
    is_test_user = user.get("id") == "test-user-000"
    
    if not is_test_user and user["plan"] == "free" and user["query_count"] >= user["query_limit"]:
        raise HTTPException(status_code=429, detail="Free trial limit reached. Please upgrade.")

    # Increment query count — skip for test user (no DB write)
    new_count = user["query_count"] + 1
    if not is_test_user:
        await supabase_query("PATCH", "users",
            data={"query_count": new_count},
            filters=f"?id=eq.{user_id}")
        if data.query_text and len(data.query_text) > 15:
            await supabase_query("POST", "query_history", data={
                "user_id": user_id,
                "query_text": data.query_text,
                "mode": data.mode
            })

    # ── Server-side KB retrieval ──
    query_for_retrieval = data.query_text
    # Enrich short follow-up queries with last user message
    if data.is_followup and data.query_text and len(data.query_text.split()) <= 5:
        last_user = next((m for m in reversed(data.messages[-6:]) if m.get("role")=="user"), None)
        if last_user:
            last_text = last_user.get("content","") if isinstance(last_user.get("content"), str) else ""
            if last_text and last_text != data.query_text:
                query_for_retrieval = last_text[:200] + " " + data.query_text

    retrieved_chunks = await retrieve_from_supabase(
        query_for_retrieval,
        top_k=10,
        is_followup=data.is_followup
    )

    # Build system prompt server-side
    base_prompt = """You are GSTMind, an expert AI legal research assistant for Indian GST law, built for Chartered Accountants and tax professionals.

RULES:
1. Answer using ONLY the knowledge base documents provided. If not found, clearly state it is not in the available documents.
2. Cite sources inline using superscript \u00b9 \u00b2 \u00b3 immediately after each cited fact.
3. Start your response with exactly: CONFIDENCE: HIGH or CONFIDENCE: MEDIUM or CONFIDENCE: LOW
   HIGH = directly stated in Act/Rule/Notification | MEDIUM = Circular/FAQ/AAR | LOW = interpretive
4. Never guess or use knowledge beyond the documents provided.
5. Always cite exact provisions: "Section 54(1) of the CGST Act, 2017\u00b9"

Write in flowing professional paragraphs. Structure: Legal Position \u2192 Applicable Provisions \u2192 Conditions & Exceptions \u2192 Practical Implication \u2192 Conclusion

End with:
REFERENCES:
1. [Document Name] \u2014 [Section/Para cited]"""

    # Draft mode uses frontend-provided prompt (has notice context)
    if data.mode == "draft" and data.system:
        base_prompt = data.system
    
    if data.is_followup and len(data.messages) > 2:
        base_prompt += "\n\nThis is a follow-up. Use conversation history for context."
    
    if data.has_attachment:
        base_prompt += "\n\nThe user has attached a document (notice/PDF). Read it carefully and respond accordingly."

    # No hardcoded built-in docs — all content comes from Supabase KB

    if retrieved_chunks:
        base_prompt += "\n\nKNOWLEDGE BASE:\n"
        by_doc = {}
        for r in retrieved_chunks:
            if r["doc"] not in by_doc:
                by_doc[r["doc"]] = []
            by_doc[r["doc"]].append(r["chunk"])
        for doc_name, chunks in by_doc.items():
            base_prompt += f"[{doc_name}]\n"
            for ch in chunks:
                base_prompt += compress_chunk(ch) + "\n"
        base_prompt += "\nCite document names as footnotes \u00b9\u00b2\u00b3."
    else:
        base_prompt += "\n\nNo specific documents found in the knowledge base for this query. Answer from general GST law knowledge if confident, otherwise advise the user to upload relevant documents."
    
    # Send retrieved sources to frontend for display
    sources_meta = [{"doc": r["doc"], "score": round(r["score"], 2)} for r in retrieved_chunks]

    async def stream_generator():
        import asyncio
        # First send meta: query count + sources
        yield f"data: {json.dumps({'type':'meta','query_count':new_count,'sources':sources_meta})}\n\n"

        max_retries = 3
        retry_delays = [8, 15, 25]

        for attempt in range(max_retries):
            succeeded = False
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream(
                        "POST",
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": CLAUDE_API_KEY,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                            "anthropic-beta": "prompt-caching-2024-07-31"
                        },
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 4000,
                            "stream": True,
                            "system": [
                                {
                                    "type": "text",
                                    "text": base_prompt,
                                    "cache_control": {"type": "ephemeral"}
                                }
                            ],
                            "messages": data.messages[-6:] if len(data.messages) > 6 else data.messages
                        }
                    ) as response:
                        if response.status_code == 429:
                            err_body = await response.aread()
                            print(f"Rate limit hit (attempt {attempt+1}): {err_body[:200]}")
                            if attempt < max_retries - 1:
                                wait = retry_delays[attempt]
                                yield f"data: {json.dumps({'type':'waiting','seconds':wait})}\n\n"
                                await asyncio.sleep(wait)
                                continue
                            else:
                                yield f"data: {json.dumps({'type':'error','message':'rate_limit'})}\n\n"
                                return

                        if response.status_code != 200:
                            err_body = await response.aread()
                            err_msg = 'AI service error'
                            try:
                                err_json = json.loads(err_body)
                                err_msg = err_json.get('error',{}).get('message', err_msg)
                            except:
                                pass
                            print(f"Claude API error {response.status_code}: {err_msg}")
                            yield f"data: {json.dumps({'type':'error','message':err_msg})}\n\n"
                            return

                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:]
                            if raw == "[DONE]":
                                break
                            try:
                                event = json.loads(raw)
                                etype = event.get("type","")
                                if etype == "content_block_delta":
                                    delta = event.get("delta",{})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text","")
                                        if text:
                                            yield f"data: {json.dumps({'type':'text','text':text})}\n\n"
                                elif etype == "message_stop":
                                    yield f"data: {json.dumps({'type':'done'})}\n\n"
                                    break
                            except Exception as parse_err:
                                print(f"Stream parse error: {parse_err}")
                                continue
                        succeeded = True

            except Exception as attempt_err:
                print(f"Attempt {attempt+1} error: {attempt_err}")
                if attempt < max_retries - 1:
                    wait = retry_delays[attempt]
                    yield f"data: {json.dumps({'type':'waiting','seconds':wait})}\n\n"
                    await asyncio.sleep(wait)
                else:
                    yield f"data: {json.dumps({'type':'error','message':'Connection error. Please try again.'})}\n\n"
                    return

            if succeeded:
                break

    async def safe_stream_generator():
        try:
            async for chunk in stream_generator():
                yield chunk
        except Exception as e:
            import traceback
            print(f"Stream generator error: {e}")
            print(traceback.format_exc())
            yield f"data: {json.dumps({'type':'error','message':str(e)[:200]})}\n\n"

    return FastAPIStreamingResponse(
        safe_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )

# ════════════════════════════════
# PAYMENT ENDPOINTS (Razorpay)
# ════════════════════════════════
class CreateOrder(BaseModel):
    plan: str  # 'pro' or 'unlimited'

@app.post("/payment/create-order")
async def create_order(data: CreateOrder, user_id: str = Depends(get_current_user)):
    amounts = {"pro": 99900, "unlimited": 199900}  # in paise
    amount = amounts.get(data.plan)
    if not amount:
        raise HTTPException(status_code=400, detail="Invalid plan")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.razorpay.com/v1/orders",
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            json={"amount": amount, "currency": "INR",
                  "notes": {"user_id": user_id, "plan": data.plan}}
        )
    order = r.json()
    return {"order_id": order["id"], "amount": amount, "key": RAZORPAY_KEY_ID}

class VerifyPayment(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    plan: str

@app.post("/payment/verify")
async def verify_payment(data: VerifyPayment, user_id: str = Depends(get_current_user)):
    # Verify signature
    body = f"{data.razorpay_order_id}|{data.razorpay_payment_id}"
    expected_sig = hmac.new(
        RAZORPAY_KEY_SECRET.encode(), body.encode(), hashlib.sha256
    ).hexdigest()
    if expected_sig != data.razorpay_signature:
        raise HTTPException(status_code=400, detail="Payment verification failed")

    # Update user plan
    limits = {"pro": 100, "unlimited": 999999}
    await supabase_query("PATCH", "users",
        data={"plan": data.plan, "query_limit": limits[data.plan], "query_count": 0},
        filters=f"?id=eq.{user_id}")

    # Record subscription
    await supabase_query("POST", "subscriptions", data={
        "user_id": user_id, "plan": data.plan,
        "razorpay_payment_id": data.razorpay_payment_id,
        "razorpay_order_id": data.razorpay_order_id,
        "amount": 99900 if data.plan == "pro" else 199900,
        "status": "active"
    })
    return {"success": True, "plan": data.plan}

# ════════════════════════════════
# ADMIN ENDPOINTS
# ════════════════════════════════
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "gstmind-admin-secret")

def verify_admin(x_admin_secret: str = Header(None)):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Admin access required")
    return True

@app.get("/admin/users")
async def admin_get_users(admin=Depends(verify_admin)):
    return await supabase_query("GET", "users", filters="?order=joined_at.desc")

@app.patch("/admin/users/{user_id}")
async def admin_update_user(user_id: str, data: dict, admin=Depends(verify_admin)):
    return await supabase_query("PATCH", "users", data=data, filters=f"?id=eq.{user_id}")

@app.get("/admin/analytics")
async def admin_analytics(admin=Depends(verify_admin)):
    users = await supabase_query("GET", "users")
    history = await supabase_query("GET", "query_history",
        filters="?order=created_at.desc&limit=1000")
    total_queries = sum(u.get("query_count", 0) for u in users)
    paid = [u for u in users if u.get("plan") in ["pro", "unlimited"]]
    revenue = sum(999 if u["plan"]=="pro" else 1999 for u in paid)
    # Top queries
    query_counts = {}
    for h in history:
        q = h["query_text"]
        query_counts[q] = query_counts.get(q, 0) + 1
    top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "total_users": len(users),
        "paid_users": len(paid),
        "total_queries": total_queries,
        "revenue": revenue,
        "top_queries": [{"query": q, "count": c} for q, c in top_queries]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "GSTMind API"}

