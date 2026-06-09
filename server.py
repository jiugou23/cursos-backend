from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

import os
import logging
import secrets
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Annotated

import bcrypt
import jwt
from bson import ObjectId
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr, Field, BeforeValidator, ConfigDict

# ---------- DB ----------
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# ---------- App ----------
app = FastAPI(title="Cursos API")
api = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.cursostraderdeelite.com",
        "https://cursostraderdeelite.com",
        "http://www.cursostraderdeelite.com",
        "http://cursostraderdeelite.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- JWT / auth helpers ----------
JWT_ALGORITHM = "HS256"


def get_jwt_secret() -> str:
    return os.environ["JWT_SECRET"]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=60 * 24),
        "type": "access",
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "type": "refresh",
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    response.set_cookie("access_token", access_token, httponly=True, secure=False, samesite="none", max_age=60 * 60 * 24, path="/")
    response.set_cookie("refresh_token", refresh_token, httponly=True, secure=False, samesite="none", max_age=60 * 60 * 24 * 7, path="/")


def clear_auth_cookies(response: Response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")


async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if not token:
        raise HTTPException(status_code=401, detail="No autenticado")
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Token inválido")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="Usuario no encontrado")
        user["id"] = str(user["_id"])
        user.pop("_id", None)
        user.pop("password_hash", None)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido")


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Requiere permisos de administrador")
    return user


# ---------- Models ----------
PyObjectId = Annotated[str, BeforeValidator(lambda v: str(v) if isinstance(v, ObjectId) else v)]


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    name: str
    role: str


class AuthResponse(BaseModel):
    user: UserPublic
    access_token: str


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=1)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class LessonIn(BaseModel):
    title: str
    youtube_url: str
    duration: Optional[str] = ""
    order: int = 0


class ModuleIn(BaseModel):
    title: str
    order: int = 0


class CourseIn(BaseModel):
    title: str
    description: Optional[str] = ""
    level: str = "principiante"  # principiante | intermedio | avanzado
    thumbnail_url: Optional[str] = ""
    category: Optional[str] = ""


def _serialize(doc: dict) -> dict:
    if not doc:
        return doc
    doc["id"] = str(doc.pop("_id"))
    return doc


def _extract_yt_id(url: str) -> str:
    """Extract YouTube video id from any common URL format."""
    if not url:
        return ""
    url = url.strip()
    # Already an id
    if len(url) == 11 and "/" not in url and "." not in url:
        return url
    import re
    patterns = [
        r"(?:v=|/embed/|/shorts/|youtu\.be/|/v/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return url


# ---------- Auth endpoints ----------
@api.post("/auth/register", response_model=AuthResponse)
async def register(payload: RegisterIn, response: Response):
    email = payload.email.lower().strip()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    doc = {
        "email": email,
        "password_hash": hash_password(payload.password),
        "name": payload.name.strip(),
        "role": "user",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = await db.users.insert_one(doc)
    user_id = str(result.inserted_id)
    access = create_access_token(user_id, email)
    set_auth_cookies(response, access, create_refresh_token(user_id))
    return AuthResponse(
        user=UserPublic(id=user_id, email=email, name=doc["name"], role="user"),
        access_token=access,
    )


@api.post("/auth/login", response_model=AuthResponse)
async def login(payload: LoginIn, request: Request, response: Response):
    email = payload.email.lower().strip()
    ip = request.client.host if request.client else "unknown"
    identifier = f"{ip}:{email}"

    attempts_doc = await db.login_attempts.find_one({"identifier": identifier})
    if attempts_doc:
        if attempts_doc.get("count", 0) >= 5:
            locked_until = attempts_doc.get("locked_until")
            if locked_until and datetime.fromisoformat(locked_until) > datetime.now(timezone.utc):
                raise HTTPException(status_code=429, detail="Demasiados intentos. Intenta más tarde.")

    user = await db.users.find_one({"email": email})
    if not user or not verify_password(payload.password, user["password_hash"]):
        await db.login_attempts.update_one(
            {"identifier": identifier},
            {"$inc": {"count": 1},
             "$set": {"locked_until": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()}},
            upsert=True,
        )
        raise HTTPException(status_code=401, detail="Email o contraseña incorrectos")

    await db.login_attempts.delete_one({"identifier": identifier})
    uid = str(user["_id"])
    access = create_access_token(uid, email)
    set_auth_cookies(response, access, create_refresh_token(uid))
    return AuthResponse(
        user=UserPublic(id=uid, email=email, name=user.get("name", ""), role=user.get("role", "user")),
        access_token=access,
    )


@api.post("/auth/logout")
async def logout(response: Response, _user: dict = Depends(get_current_user)):
    clear_auth_cookies(response)
    return {"ok": True}


@api.get("/auth/me", response_model=UserPublic)
async def me(user: dict = Depends(get_current_user)):
    return UserPublic(id=user["id"], email=user["email"], name=user.get("name", ""), role=user.get("role", "user"))


@api.post("/auth/refresh")
async def refresh_token(request: Request, response: Response):
    rtoken = request.cookies.get("refresh_token")
    if not rtoken:
        raise HTTPException(status_code=401, detail="Sin refresh token")
    try:
        payload = jwt.decode(rtoken, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Token inválido")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="Usuario no encontrado")
        access = create_access_token(str(user["_id"]), user["email"])
        response.set_cookie("access_token", access, httponly=True, secure=False,
                            samesite="lax", max_age=60 * 60 * 24, path="/")
        return {"ok": True}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido")


# ---------- Courses ----------
@api.get("/courses")
async def list_courses(user: dict = Depends(get_current_user)):
    courses = []
    cursor = db.courses.find().sort("created_at", -1)
    async for c in cursor:
        c = _serialize(c)
        # compute totals + progress for user
        modules = await db.modules.find({"course_id": c["id"]}).to_list(1000)
        module_ids = [str(m["_id"]) for m in modules]
        total = await db.lessons.count_documents({"module_id": {"$in": module_ids}})
        completed = await db.progress.count_documents({
            "user_id": user["id"],
            "module_id": {"$in": module_ids},
            "completed": True,
        }) if module_ids else 0
        c["total_lessons"] = total
        c["completed_lessons"] = completed
        c["progress_pct"] = round((completed / total) * 100) if total else 0
        courses.append(c)
    return courses


@api.post("/courses")
async def create_course(payload: CourseIn, _: dict = Depends(require_admin)):
    doc = payload.model_dump()
    doc["created_at"] = datetime.now(timezone.utc).isoformat()
    result = await db.courses.insert_one(doc)
    doc.pop("_id", None)
    doc["id"] = str(result.inserted_id)
    return doc


@api.get("/courses/{course_id}")
async def get_course(course_id: str, user: dict = Depends(get_current_user)):
    try:
        course = await db.courses.find_one({"_id": ObjectId(course_id)})
    except Exception:
        raise HTTPException(status_code=404, detail="Curso no encontrado")
    if not course:
        raise HTTPException(status_code=404, detail="Curso no encontrado")
    course = _serialize(course)

    modules = []
    async for m in db.modules.find({"course_id": course_id}).sort("order", 1):
        m = _serialize(m)
        lessons = []
        async for l in db.lessons.find({"module_id": m["id"]}).sort("order", 1):
            l = _serialize(l)
            l["youtube_id"] = _extract_yt_id(l.get("youtube_url", ""))
            prog = await db.progress.find_one({
                "user_id": user["id"], "lesson_id": l["id"]
            })
            l["completed"] = bool(prog and prog.get("completed"))
            lessons.append(l)
        m["lessons"] = lessons
        modules.append(m)

    course["modules"] = modules
    total = sum(len(m["lessons"]) for m in modules)
    completed = sum(1 for m in modules for l in m["lessons"] if l["completed"])
    course["total_lessons"] = total
    course["completed_lessons"] = completed
    course["progress_pct"] = round((completed / total) * 100) if total else 0
    return course


@api.put("/courses/{course_id}")
async def update_course(course_id: str, payload: CourseIn, _: dict = Depends(require_admin)):
    await db.courses.update_one({"_id": ObjectId(course_id)}, {"$set": payload.model_dump()})
    doc = await db.courses.find_one({"_id": ObjectId(course_id)})
    return _serialize(doc)


@api.delete("/courses/{course_id}")
async def delete_course(course_id: str, _: dict = Depends(require_admin)):
    # cascade
    modules = await db.modules.find({"course_id": course_id}).to_list(1000)
    module_ids = [str(m["_id"]) for m in modules]
    if module_ids:
        await db.lessons.delete_many({"module_id": {"$in": module_ids}})
    await db.modules.delete_many({"course_id": course_id})
    await db.courses.delete_one({"_id": ObjectId(course_id)})
    return {"ok": True}


# ---------- Modules ----------
@api.post("/courses/{course_id}/modules")
async def create_module(course_id: str, payload: ModuleIn, _: dict = Depends(require_admin)):
    doc = payload.model_dump()
    doc["course_id"] = course_id
    doc["created_at"] = datetime.now(timezone.utc).isoformat()
    result = await db.modules.insert_one(doc)
    doc.pop("_id", None)
    doc["id"] = str(result.inserted_id)
    return doc


@api.put("/modules/{module_id}")
async def update_module(module_id: str, payload: ModuleIn, _: dict = Depends(require_admin)):
    await db.modules.update_one({"_id": ObjectId(module_id)}, {"$set": payload.model_dump()})
    doc = await db.modules.find_one({"_id": ObjectId(module_id)})
    return _serialize(doc)


@api.delete("/modules/{module_id}")
async def delete_module(module_id: str, _: dict = Depends(require_admin)):
    await db.lessons.delete_many({"module_id": module_id})
    await db.modules.delete_one({"_id": ObjectId(module_id)})
    return {"ok": True}


# ---------- Lessons ----------
@api.post("/modules/{module_id}/lessons")
async def create_lesson(module_id: str, payload: LessonIn, _: dict = Depends(require_admin)):
    doc = payload.model_dump()
    doc["module_id"] = module_id
    doc["created_at"] = datetime.now(timezone.utc).isoformat()
    result = await db.lessons.insert_one(doc)
    doc.pop("_id", None)
    doc["id"] = str(result.inserted_id)
    doc["youtube_id"] = _extract_yt_id(doc.get("youtube_url", ""))
    return doc


@api.put("/lessons/{lesson_id}")
async def update_lesson(lesson_id: str, payload: LessonIn, _: dict = Depends(require_admin)):
    await db.lessons.update_one({"_id": ObjectId(lesson_id)}, {"$set": payload.model_dump()})
    doc = await db.lessons.find_one({"_id": ObjectId(lesson_id)})
    doc = _serialize(doc)
    doc["youtube_id"] = _extract_yt_id(doc.get("youtube_url", ""))
    return doc


@api.delete("/lessons/{lesson_id}")
async def delete_lesson(lesson_id: str, _: dict = Depends(require_admin)):
    await db.lessons.delete_one({"_id": ObjectId(lesson_id)})
    await db.progress.delete_many({"lesson_id": lesson_id})
    return {"ok": True}


# ---------- Progress ----------
class ProgressIn(BaseModel):
    completed: bool = True


@api.post("/lessons/{lesson_id}/progress")
async def mark_lesson(lesson_id: str, payload: ProgressIn, user: dict = Depends(get_current_user)):
    lesson = await db.lessons.find_one({"_id": ObjectId(lesson_id)})
    if not lesson:
        raise HTTPException(status_code=404, detail="Clase no encontrada")
    await db.progress.update_one(
        {"user_id": user["id"], "lesson_id": lesson_id},
        {"$set": {
            "user_id": user["id"],
            "lesson_id": lesson_id,
            "module_id": lesson.get("module_id"),
            "completed": payload.completed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }},
        upsert=True,
    )
    return {"ok": True, "completed": payload.completed}


# ---------- User management (admin) ----------
class RoleUpdate(BaseModel):
    role: str  # "admin" | "user"


@api.get("/admin/users")
async def list_users(current: dict = Depends(require_admin)):
    users = []
    async for u in db.users.find().sort("created_at", -1):
        users.append({
            "id": str(u["_id"]),
            "email": u.get("email"),
            "name": u.get("name", ""),
            "role": u.get("role", "user"),
            "created_at": u.get("created_at"),
            "is_self": str(u["_id"]) == current["id"],
        })
    return users


@api.patch("/admin/users/{user_id}/role")
async def update_user_role(user_id: str, payload: RoleUpdate, current: dict = Depends(require_admin)):
    if payload.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Rol inválido")
    try:
        target = await db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if not target:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if str(target["_id"]) == current["id"] and payload.role != "admin":
        raise HTTPException(status_code=400, detail="No puedes quitar tu propio rol de administrador")
    await db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"role": payload.role}})
    return {"ok": True, "role": payload.role}


@api.delete("/admin/users/{user_id}")
async def delete_user(user_id: str, current: dict = Depends(require_admin)):
    if user_id == current["id"]:
        raise HTTPException(status_code=400, detail="No puedes eliminarte a ti mismo")
    try:
        await db.users.delete_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    await db.progress.delete_many({"user_id": user_id})
    return {"ok": True}


# ---------- Site settings (branding / footer) ----------
DEFAULT_SETTINGS = {
    "brand_name": "AULA",
    "brand_tagline": "Premium",
    "logo_url": "",
    "footer_columns": [
        {
            "title": "Navegación",
            "links": [
                {"label": "Mis Cursos", "href": "/cursos"},
                {"label": "Mi Perfil", "href": "/perfil"},
            ],
        },
        {
            "title": "Soporte",
            "links": [
                {"label": "Ayuda & FAQ", "href": "#"},
                {"label": "Términos de Uso", "href": "#"},
                {"label": "Privacidad", "href": "#"},
            ],
        },
    ],
    "social_links": [
        {"label": "Instagram", "icon": "instagram", "href": ""},
        {"label": "YouTube", "icon": "youtube", "href": ""},
    ],
    "copyright_text": "© 2026 AULA. Todos los derechos reservados.",
}


class SiteSettings(BaseModel):
    brand_name: str = "AULA"
    brand_tagline: str = "Premium"
    logo_url: str = ""
    footer_columns: list = Field(default_factory=list)
    social_links: list = Field(default_factory=list)
    copyright_text: str = ""


async def _get_settings_doc() -> dict:
    doc = await db.settings.find_one({"_id": "site"})
    if not doc:
        doc = {"_id": "site", **DEFAULT_SETTINGS}
        await db.settings.insert_one(doc)
    doc.pop("_id", None)
    # Fill in defaults for any missing keys
    for k, v in DEFAULT_SETTINGS.items():
        doc.setdefault(k, v)
    return doc


@api.get("/settings")
async def get_settings():
    return await _get_settings_doc()


@api.put("/settings")
async def update_settings(payload: SiteSettings, _: dict = Depends(require_admin)):
    data = payload.model_dump()
    await db.settings.update_one({"_id": "site"}, {"$set": data}, upsert=True)
    return await _get_settings_doc()


# ---------- Startup ----------
@app.on_event("startup")
async def on_startup():
    await db.users.create_index("email", unique=True)
    await db.modules.create_index("course_id")
    await db.lessons.create_index("module_id")
    await db.progress.create_index([("user_id", 1), ("lesson_id", 1)], unique=True)
    await db.login_attempts.create_index("identifier")

    admin_email = os.environ.get("ADMIN_EMAIL", "admin@cursos.com").lower()
    admin_password = os.environ.get("ADMIN_PASSWORD", "Admin123!")
    existing = await db.users.find_one({"email": admin_email})
    if existing is None:
        await db.users.insert_one({
            "email": admin_email,
            "password_hash": hash_password(admin_password),
            "name": "Administrador",
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"Admin creado: {admin_email}")
    else:
        # keep password in sync if env changed and ensure admin role
        updates = {}
        if not verify_password(admin_password, existing["password_hash"]):
            updates["password_hash"] = hash_password(admin_password)
        if existing.get("role") != "admin":
            updates["role"] = "admin"
        if updates:
            await db.users.update_one({"email": admin_email}, {"$set": updates})


@app.on_event("shutdown")
async def on_shutdown():
    client.close()


# ---------- Mount ----------
app.include_router(api)


# ---------- SendGrid / Email ----------
import urllib.request
import json as _json

def send_welcome_email(to_email: str, name: str, password: str):
    """Send welcome email with credentials via SendGrid."""
    from_email = os.environ.get("FROM_EMAIL", "noreply@cursostraderdeelite.com")
    api_key = os.environ.get("SENDGRID_API_KEY", "")
    if not api_key:
        logger.warning("SENDGRID_API_KEY not set, skipping email")
        return

    subject = "¡Bienvenido a Cursos Trader de Elite!"
    body = f"""
<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#0a0a0a;color:#f5f5f0;padding:40px;border-radius:12px">
  <h1 style="color:#d4af37;font-size:24px">¡Bienvenido, {name}!</h1>
  <p>Tu acceso a <strong>Cursos Trader de Elite</strong> está listo.</p>
  <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:20px;margin:24px 0">
    <p style="margin:0 0 8px"><strong>Email:</strong> {to_email}</p>
    <p style="margin:0"><strong>Contraseña:</strong> {password}</p>
  </div>
  <a href="https://www.cursostraderdeelite.com" style="display:inline-block;background:#d4af37;color:#0a0a0a;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:bold">
    Entrar a la Plataforma
  </a>
  <p style="color:#a0998a;font-size:12px;margin-top:24px">Si no realizaste esta compra, ignora este email.</p>
</div>
"""

    data = _json.dumps({
        "personalizations": [{"to": [{"email": to_email, "name": name}]}],
        "from": {"email": from_email, "name": "Cursos Trader de Elite"},
        "subject": subject,
        "content": [{"type": "text/html", "value": body}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    try:
        urllib.request.urlopen(req)
        logger.info(f"Welcome email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


import secrets as _secrets

def generate_password(length=10) -> str:
    alphabet = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(_secrets.choice(alphabet) for _ in range(length))


# ---------- Eduzz Webhook ----------
@api.post("/webhook/eduzz")
async def eduzz_webhook(request: Request):
    """
    Receives Eduzz postback and creates user automatically.
    Eduzz sends form-encoded POST with fields like:
      buyer_email, buyer_name, trans_status (paid=3)
    """
    try:
        body = await request.form()
        data = dict(body)
    except Exception:
        try:
            data = await request.json()
        except Exception:
            data = {}

    logger.info(f"Eduzz webhook received: {data}")

    # Only process paid transactions
    status = str(data.get("trans_status", data.get("status", "")))
    if status not in ("3", "paid", "approved", "complete", "completed"):
        return {"ok": True, "msg": "ignored"}

    email = str(data.get("buyer_email", data.get("email", ""))).lower().strip()
    name = str(data.get("buyer_name", data.get("name", "Cliente"))).strip()

    if not email:
        raise HTTPException(status_code=400, detail="No email in payload")

    # Check if user already exists
    existing = await db.users.find_one({"email": email})
    if existing:
        logger.info(f"User {email} already exists, skipping creation")
        return {"ok": True, "msg": "user already exists"}

    # Create user with random password
    password = generate_password()
    doc = {
        "email": email,
        "password_hash": hash_password(password),
        "name": name,
        "role": "user",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "eduzz"
    }
    await db.users.insert_one(doc)
    logger.info(f"Created user {email} from Eduzz webhook")

    # Send welcome email
    send_welcome_email(email, name, password)

    return {"ok": True, "msg": "user created", "email": email}
