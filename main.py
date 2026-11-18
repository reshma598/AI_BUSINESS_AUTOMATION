import os
import json
import sqlite3
import hashlib
import secrets
import smtplib

from email.message import EmailMessage
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from pypdf import PdfReader


# 1. CONFIGURATION
load_dotenv()

key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key) if key else None

app = FastAPI(title="AI Business Automation Agent")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "change-this-secret"),
    same_site="lax",
    https_only=False
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# 2. DATABASE
def db():
    con = sqlite3.connect("business.db")
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def query(sql, values=()):
    with db() as con:
        return [dict(r) for r in con.execute(sql, values).fetchall()]


def execute(sql, values=()):
    with db() as con:
        return con.execute(sql, values).lastrowid


def setup():
    tables = """
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, password TEXT,
        role TEXT DEFAULT 'Employee'
    );

    CREATE TABLE IF NOT EXISTS employees(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, role TEXT, attendance TEXT
    );

    CREATE TABLE IF NOT EXISTS customers(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT, phone TEXT, company TEXT
    );

    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task TEXT, status TEXT, deadline TEXT
    );

    CREATE TABLE IF NOT EXISTS agent_activities(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent TEXT, user_request TEXT, status TEXT, result TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS resumes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_name TEXT, job_role TEXT, score INTEGER,
        skills TEXT, recommendation TEXT, analysis TEXT
    );

    CREATE TABLE IF NOT EXISTS leads(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT, phone TEXT, company TEXT,
        requirement TEXT, score INTEGER DEFAULT 0,
        category TEXT DEFAULT 'Not Scored', analysis TEXT
    );

    CREATE TABLE IF NOT EXISTS projects(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, description TEXT, deadline TEXT,
        status TEXT DEFAULT 'Active'
    );

    CREATE TABLE IF NOT EXISTS project_tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER, task_name TEXT, description TEXT,
        assigned_to TEXT, priority TEXT DEFAULT 'Medium',
        status TEXT DEFAULT 'Pending', deadline TEXT,
        FOREIGN KEY(project_id) REFERENCES projects(id)
    );

    CREATE TABLE IF NOT EXISTS email_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipient TEXT, subject TEXT, body TEXT, status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    with db() as con:
        con.executescript(tables)

        # Support older database versions
        columns = {
            r["name"] for r in con.execute(
                "PRAGMA table_info(customers)"
            )
        }

        for col in ("phone", "company"):
            if col not in columns:
                con.execute(f"ALTER TABLE customers ADD COLUMN {col} TEXT")

        # Demo data
        if not con.execute("SELECT COUNT(*) FROM employees").fetchone()[0]:
            con.executemany(
                "INSERT INTO employees(name,role,attendance) VALUES(?,?,?)",
                [
                    ("Rahul", "Developer", "Present"),
                    ("Priya", "HR Manager", "Absent"),
                    ("Arun", "Designer", "Present")
                ]
            )

        if not con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]:
            con.executemany(
                "INSERT INTO customers(name,email) VALUES(?,?)",
                [
                    ("John", "john@gmail.com"),
                    ("Anjali", "anjali@gmail.com")
                ]
            )


# 3. AUTHENTICATION
class User(BaseModel):
    name: str = ""
    email: str
    password: str


def hash_password(password):
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 200000
    ).hex()
    return salt + "$" + hashed


def verify_password(password, stored):
    try:
        salt, hashed = stored.split("$")
        check = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 200000
        ).hex()
        return secrets.compare_digest(check, hashed)
    except ValueError:
        return False


def current_user(request):
    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(401, "Please login first.")

    users = query(
        "SELECT id,name,email,role FROM users WHERE id=?",
        (user_id,)
    )

    if not users:
        request.session.clear()
        raise HTTPException(401, "Invalid session.")

    return users[0]


def admin_only(request):
    user = current_user(request)

    if user["role"] != "Admin":
        raise HTTPException(403, "Admin access required.")

    return user


@app.post("/register")
def register(user: User):
    if not user.name.strip() or len(user.password) < 8:
        raise HTTPException(400, "Enter name and password with 8+ characters.")

    try:
        execute(
            "INSERT INTO users(name,email,password) VALUES(?,?,?)",
            (
                user.name.strip(),
                user.email.strip().lower(),
                hash_password(user.password)
            )
        )
        return {"message": "Registration successful."}

    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email already registered.")


@app.post("/login")
def login(user: User, request: Request):
    users = query(
        "SELECT id,password FROM users WHERE email=?",
        (user.email.strip().lower(),)
    )

    if not users or not verify_password(
        user.password, users[0]["password"]
    ):
        raise HTTPException(401, "Invalid email or password.")

    request.session.clear()
    request.session["user_id"] = users[0]["id"]

    return {"message": "Login successful."}


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"message": "Logout successful."}


@app.get("/me")
def me(request: Request):
    return current_user(request)


@app.get("/admin/users")
def admin_users(request: Request):
    admin_only(request)
    return {"users": query("SELECT id,name,email,role FROM users")}


# 4. ACTIVITY LOGGER
def log_activity(agent, request, status, result):
    execute(
        """INSERT INTO agent_activities
        (agent,user_request,status,result) VALUES(?,?,?,?)""",
        (agent, str(request), status, str(result))
    )


# 5. GEMINI AI
def ask_ai(prompt):
    if not client:
        raise HTTPException(503, "Gemini API key is missing.")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if not response.text:
            raise HTTPException(502, "Empty response from Gemini.")

        return response.text.strip()

    except Exception as e:
        print("GEMINI ERROR:", repr(e))
        raise HTTPException(502, f"Gemini Error: {str(e)}")


def ai_json(prompt):
    text = ask_ai(prompt)
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(502, "Invalid AI response.")


# 6. DASHBOARD
@app.get("/dashboard")
def dashboard(request: Request):
    user = current_user(request)

    return {
        "user": user,
        "employees": query("SELECT COUNT(*) AS n FROM employees")[0]["n"],
        "customers": query("SELECT COUNT(*) AS n FROM customers")[0]["n"],
        "leads": query("SELECT COUNT(*) AS n FROM leads")[0]["n"],
        "projects": query("SELECT COUNT(*) AS n FROM projects")[0]["n"],
        "pending_tasks": query(
            "SELECT COUNT(*) AS n FROM tasks WHERE status='Pending'"
        )[0]["n"]
    }


# 7. HRMS AGENT
class Employee(BaseModel):
    name: str
    role: str
    attendance: str = "Not Marked"


@app.get("/employees")
def employees(request: Request):
    current_user(request)
    return {"employees": query("SELECT * FROM employees")}


@app.post("/employees")
def add_employee(data: Employee, request: Request):
    current_user(request)

    eid = execute(
        "INSERT INTO employees(name,role,attendance) VALUES(?,?,?)",
        (data.name, data.role, data.attendance)
    )

    log_activity("HR Agent", data.name, "Completed", "Employee added")
    return {"id": eid, "message": "Employee added."}


@app.get("/resumes")
def resumes(request: Request):
    current_user(request)
    return {"resumes": query("SELECT * FROM resumes ORDER BY id DESC")}


@app.post("/screen-resume")
async def screen_resume(
    request: Request,
    file: UploadFile = File(...),
    job_role: str = Form(...),
    job_description: str = Form(...)
):
    current_user(request)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload PDF only.")

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "PDF size must be below 10 MB.")

    import io
    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        raise HTTPException(400, "Unable to read PDF.")

    if not text.strip():
        raise HTTPException(400, "PDF contains no readable text.")

    prompt = f"""
    Evaluate this resume against the job requirements.
    Return JSON with candidate_name, skills, score (0-100),
    recommendation (Shortlisted, Review, Not Recommended), analysis.
    Do not use protected characteristics.

    Job: {job_role}
    Description: {job_description}
    Resume: {text[:15000]}
    """

    result = ai_json(prompt)

    score = max(0, min(100, int(result.get("score", 0))))
    skills = ", ".join(result.get("skills", []))
    recommendation = result.get("recommendation", "Review")

    if recommendation not in ("Shortlisted", "Review", "Not Recommended"):
        recommendation = "Review"

    rid = execute(
        """INSERT INTO resumes
        (candidate_name,job_role,score,skills,recommendation,analysis)
        VALUES(?,?,?,?,?,?)""",
        (
            result.get("candidate_name", "Unknown"),
            job_role, score, skills, recommendation,
            result.get("analysis", "")
        )
    )

    log_activity("HR Agent", job_role, "Completed", "Resume screened")
    return {"id": rid, **result, "score": score}


# 8. CRM AGENT
class Lead(BaseModel):
    name: str
    email: str = ""
    phone: str = ""
    company: str = ""
    requirement: str = ""


@app.get("/leads")
def leads(request: Request):
    current_user(request)
    return {"leads": query("SELECT * FROM leads ORDER BY id DESC")}


@app.post("/add-lead")
def add_lead(data: Lead, request: Request):
    current_user(request)

    lid = execute(
        """INSERT INTO leads(name,email,phone,company,requirement)
        VALUES(?,?,?,?,?)""",
        (data.name, data.email, data.phone, data.company, data.requirement)
    )

    log_activity("CRM Agent", data.name, "Completed", "Lead added")
    return {"id": lid, "message": "Lead added."}


@app.post("/score-lead/{lead_id}")
def score_lead(lead_id: int, request: Request):
    current_user(request)

    rows = query("SELECT * FROM leads WHERE id=?", (lead_id,))

    if not rows:
        raise HTTPException(404, "Lead not found.")

    prompt = f"""
    Score this lead from 0 to 100.
    Return JSON with score, category (Hot, Warm, Cold), analysis.
    Use only supplied facts.

    Lead: {json.dumps(rows[0])}
    """

    result = ai_json(prompt)
    score = max(0, min(100, int(result.get("score", 0))))
    category = result.get("category", "Warm")

    if category not in ("Hot", "Warm", "Cold"):
        category = "Warm"

    execute(
        "UPDATE leads SET score=?,category=?,analysis=? WHERE id=?",
        (score, category, result.get("analysis", ""), lead_id)
    )

    log_activity("CRM Agent", lead_id, "Completed", category)
    return result


# 9. PROJECT MANAGEMENT
class Project(BaseModel):
    name: str
    description: str = ""
    deadline: str = ""


class ProjectTask(BaseModel):
    project_id: int
    task_name: str
    description: str = ""
    assigned_to: str = ""
    priority: str = "Medium"
    status: str = "Pending"
    deadline: str = ""


@app.get("/projects")
def projects(request: Request):
    current_user(request)
    return {"projects": query("SELECT * FROM projects")}


@app.post("/add-project")
def add_project(data: Project, request: Request):
    current_user(request)

    pid = execute(
        "INSERT INTO projects(name,description,deadline) VALUES(?,?,?)",
        (data.name, data.description, data.deadline)
    )

    log_activity("Project Agent", data.name, "Completed", "Project created")
    return {"id": pid, "message": "Project created."}


@app.get("/project-tasks")
def project_tasks(request: Request):
    current_user(request)
    return {"tasks": query("SELECT * FROM project_tasks")}


@app.post("/add-project-task")
def add_project_task(data: ProjectTask, request: Request):
    current_user(request)

    if data.priority not in ("Low", "Medium", "High"):
        raise HTTPException(400, "Invalid priority.")

    if data.status not in ("Pending", "In Progress", "Completed"):
        raise HTTPException(400, "Invalid status.")

    pid = query("SELECT id FROM projects WHERE id=?", (data.project_id,))

    if not pid:
        raise HTTPException(404, "Project not found.")

    tid = execute(
        """INSERT INTO project_tasks
        (project_id,task_name,description,assigned_to,priority,status,deadline)
        VALUES(?,?,?,?,?,?,?)""",
        (
            data.project_id, data.task_name, data.description,
            data.assigned_to, data.priority, data.status, data.deadline
        )
    )

    log_activity("Project Agent", data.task_name, "Completed", "Task created")
    return {"id": tid, "message": "Task created."}


@app.post("/generate-project-plan")
def generate_project_plan(project_id: int, request: Request):
    current_user(request)

    rows = query("SELECT * FROM projects WHERE id=?", (project_id,))

    if not rows:
        raise HTTPException(404, "Project not found.")

    p = rows[0]

    result = ai_json(f"""
    Generate 3 to 8 project tasks.
    Return JSON: {{"tasks":[{{"task_name":"Task","description":"Details",
    "priority":"Low"}}]}}
    Project: {p}
    """)

    created = []

    for task in result.get("tasks", [])[:8]:
        priority = task.get("priority", "Medium")

        if priority not in ("Low", "Medium", "High"):
            priority = "Medium"

        tid = execute(
            """INSERT INTO project_tasks
            (project_id,task_name,description,priority)
            VALUES(?,?,?,?)""",
            (
                project_id, task.get("task_name", "New Task"),
                task.get("description", ""), priority
            )
        )
        created.append(tid)

    log_activity("Project Agent", project_id, "Completed", "Plan generated")
    return {"created_tasks": created}


# 10. CENTRAL AI CHATBOT
# STAGE 10 - AI BUSINESS ASSISTANT

class Chat(BaseModel):
    message: str


@app.post("/chat")
def chatbot(data: Chat, request: Request):
    current_user(request)

    msg = data.message.strip()
    text = msg.lower()

    if not msg:
        raise HTTPException(400, "Enter a message.")

    agent = "Business Assistant"

    # GREETING
    if text in ("hi", "hello", "hey"):
        answer = "Hello! I am your AI Business Assistant. I can help with HR, CRM, Projects, Emails and business activities."

    # EMPLOYEES
    elif "absent" in text or "attendance" in text or "employee" in text:
        agent = "HR Agent"
        rows = query("SELECT name,role,attendance FROM employees")

        if "absent" in text:
            rows = [r for r in rows if r["attendance"].lower() == "absent"]

        answer = "\n".join(
            f'{r["name"]} - {r["role"]} - {r["attendance"]}'
            for r in rows
        ) or "No employee records found."

    # RESUMES
    elif "resume" in text or "candidate" in text:
        agent = "HR Agent"
        rows = query("SELECT candidate_name,job_role,score,recommendation FROM resumes")
        answer = "\n".join(
            f'{r["candidate_name"]} - {r["job_role"]} - Score: {r["score"]} - {r["recommendation"]}'
            for r in rows
        ) or "No resume records found."

    # CRM
    elif "lead" in text:
        agent = "CRM Agent"
        rows = query("SELECT name,company,score,category FROM leads")
        answer = "\n".join(
            f'{r["name"]} - {r["company"]} - {r["score"]} - {r["category"]}'
            for r in rows
        ) or "No leads found."

    elif "customer" in text:
        agent = "CRM Agent"
        rows = query("SELECT name,email FROM customers")
        answer = "\n".join(
            f'{r["name"]} - {r["email"]}' for r in rows
        ) or "No customers found."

    # PROJECTS AND TASKS
    elif "project" in text:
        agent = "Project Agent"
        rows = query("SELECT name,status,deadline FROM projects")
        answer = "\n".join(
            f'{r["name"]} - {r["status"]} - {r["deadline"]}'
            for r in rows
        ) or "No projects found."

    elif "task" in text:
        agent = "Project Agent"
        rows = query("SELECT task_name,status,deadline FROM project_tasks")
        answer = "\n".join(
            f'{r["task_name"]} - {r["status"]} - {r["deadline"]}'
            for r in rows
        ) or "No project tasks found."

    # EMAIL
    elif "email" in text:
        agent = "Email Agent"
        answer = "I can generate professional email drafts. Open the Email Agent, enter the details, review the draft and send it."

    # ACTIVITIES
    elif "activity" in text:
        agent = "Activity Agent"
        rows = query(
            "SELECT agent,user_request,status FROM agent_activities ORDER BY id DESC LIMIT 10"
        )
        answer = "\n".join(
            f'{r["agent"]}: {r["user_request"]} - {r["status"]}'
            for r in rows
        ) or "No activities recorded."

    # BUSINESS SUMMARY
    elif "summary" in text or "overview" in text:
        agent = "Business Assistant"
        tables = ["employees", "customers", "leads", "projects"]
        answer = "\n".join(
            f'{t.title()}: {query(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"]}'
            for t in tables
        )

    # GENERAL BUSINESS QUESTIONS
    else:
        agent = "Business Assistant"
        answer = ask_ai(
            "You are an AI Business Automation Assistant. "
            "Explain HRMS, CRM, project management, email automation "
            "and business workflows clearly. Never claim an action "
            "was performed unless it was actually executed.\n"
            + msg
        )

    log_activity(agent, msg, "Completed", answer)

    return {"agent": agent, "response": answer}


# 11. EMAIL AUTOMATION
class EmailData(BaseModel):
    purpose: str
    recipient_name: str
    details: str


class SendEmail(BaseModel):
    to: str
    subject: str
    body: str


@app.post("/generate-email")
def generate_email(data: EmailData, request: Request):
    current_user(request)

    prompt = f"""
    Write a professional business email.
    Purpose: {data.purpose}
    Recipient: {data.recipient_name}
    Details: {data.details}
    Include subject and email body. Do not invent facts.
    """

    draft = ask_ai(prompt)

    log_activity("Email Agent", data.purpose, "Completed", "Draft generated")

    return {"email": draft}


@app.post("/send-email")
def send_email(data: SendEmail, request: Request):
    current_user(request)

    sender = os.getenv("SMTP_EMAIL")
    password = os.getenv("SMTP_PASSWORD")

    if not sender or not password:
        raise HTTPException(503, "SMTP settings missing.")

    if "@" not in data.to or not data.subject.strip() or not data.body.strip():
        raise HTTPException(400, "Invalid email details.")

    try:
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = data.to
        msg["Subject"] = data.subject
        msg.set_content(data.body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)

        execute(
            """INSERT INTO email_logs(recipient,subject,body,status)
            VALUES(?,?,?,'Sent')""",
            (data.to, data.subject, data.body)
        )

        log_activity("Email Agent", data.subject, "Completed", "Email sent")

        return {"message": "Email sent successfully."}

    except Exception:
        log_activity("Email Agent", data.subject, "Failed", "Sending failed")
        raise HTTPException(502, "Email sending failed.")


@app.get("/email-history")
def email_history(request: Request):
    current_user(request)

    return {
        "emails": query(
            "SELECT id,recipient,subject,status,created_at FROM email_logs"
        )
    }


# 12. ACTIVITY CENTER
@app.get("/activities")
def activities(request: Request):
    current_user(request)

    return {
        "activities": query(
            "SELECT * FROM agent_activities ORDER BY id DESC LIMIT 100"
        )
    }


@app.get("/activity-summary")
def activity_summary(request: Request):
    current_user(request)

    rows = query(
        "SELECT status,COUNT(*) AS total FROM agent_activities GROUP BY status"
    )

    counts = {r["status"]: r["total"] for r in rows}

    return {
        "total": sum(counts.values()),
        "completed": counts.get("Completed", 0),
        "failed": counts.get("Failed", 0)
    }


@app.get("/agent-summary")
def agent_summary(request: Request):
    current_user(request)

    return {
        "agents": query(
            """SELECT agent,COUNT(*) AS total
            FROM agent_activities GROUP BY agent"""
        )
    }


# 13. HOME AND HEALTH
@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {
        "status": "Application Running",
        "gemini_connected": client is not None
    }


# INITIALIZE DATABASE
setup()