"""FinSight AI — Smart expense tracker with ML auto-categorisation, budgets, anomaly alerts & forecasting.
Run:  python app.py   →  http://127.0.0.1:5000   (demo: demo@example.com / Demo@1234)
"""
import csv
import io
import random
from datetime import date, datetime, timedelta

import numpy as np
from flask import Response, g, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from core import create_app, execute, login_required, query

CATEGORIES = {
    "Food & Dining": ("bi-cup-hot", "#FF7A59"), "Groceries": ("bi-basket", "#2CB67D"), "Transport": ("bi-car-front", "#3DA9FC"),
    "Shopping": ("bi-bag", "#E85D75"), "Bills & Utilities": ("bi-lightning-charge", "#F7B32B"), "Entertainment": ("bi-film", "#9B5DE5"),
    "Health": ("bi-heart-pulse", "#FF5470"), "Education": ("bi-mortarboard", "#00BBF9"), "Rent": ("bi-house", "#7F5AF0"),
    "Travel": ("bi-airplane", "#06D6A0"), "Salary": ("bi-briefcase", "#2CB67D"), "Freelance": ("bi-laptop", "#00C9FF"),
    "Investments": ("bi-graph-up-arrow", "#FFD166"), "Other": ("bi-three-dots", "#9AA0C3"),
}
SEED_TEXT = {
    "Food & Dining": "swiggy order zomato dinner pizza dominos burger kfc cafe coffee starbucks restaurant lunch biryani tea snacks mcdonalds chai",
    "Groceries": "bigbasket grocery dmart vegetables milk fruits blinkit zepto supermarket reliance fresh eggs rice atta",
    "Transport": "uber ola rapido metro petrol fuel diesel bus ticket auto parking toll fastag cab",
    "Shopping": "amazon flipkart myntra clothes shoes ajio electronics headphones meesho gift watch",
    "Bills & Utilities": "electricity bill airtel jio recharge wifi broadband water bill gas cylinder mobile postpaid dth",
    "Entertainment": "netflix prime video spotify hotstar movie pvr inox concert gaming steam bookmyshow",
    "Health": "pharmacy apollo medicine doctor hospital gym membership cult fit lab test dental",
    "Education": "udemy coursera course books exam fee tuition college fees certification",
    "Rent": "house rent monthly rent pg hostel maintenance flat",
    "Travel": "irctc train flight indigo makemytrip hotel oyo goibibo trip booking airbnb",
    "Salary": "salary credited monthly salary payroll company pay",
    "Freelance": "freelance project payment client invoice upwork fiverr consulting",
    "Investments": "mutual fund sip zerodha groww stocks dividend fd interest",
    "Other": "misc transfer atm withdrawal donation charity",
}
INCOME_CATS = {"Salary", "Freelance", "Investments"}

UI = dict(
    name="FinSight AI", short="Smart Money Manager", icon="💸", accent1="#7F5AF0", accent2="#2CB67D", home="dashboard",
    tagline="Track every rupee, let AI categorise your spending, stay inside budgets and see where your money is heading.",
    hero='Your money, <span class="gradient-text">understood by AI.</span>', badge="AI-powered finance",
    features=[("bi-robot", "Auto-categorisation", "ML model learns from your own corrections"),
              ("bi-bullseye", "Smart budgets", "Live progress bars & overspend alerts"),
              ("bi-graph-up-arrow", "Forecast & anomalies", "Predicts next month, flags unusual spends")],
    float=['<b class="gradient-text">₹12,480</b> saved this month 🎉', '🤖 "Swiggy order" → <b>Food & Dining</b> 96%'],
    nav=[("Dashboard", "dashboard", "bi-grid-1x2"), ("Transactions", "transactions", "bi-arrow-left-right"),
         ("Budgets", "budgets", "bi-bullseye"), ("Insights", "insights", "bi-lightbulb")],
    search_hint="Search transactions…", promo_title="AI tip", promo="Correct a category once — the model retrains and remembers.",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date TEXT NOT NULL, description TEXT NOT NULL, amount REAL NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('income','expense')),
    category TEXT NOT NULL, ai_confidence REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS budgets (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, category TEXT NOT NULL, monthly_limit REAL NOT NULL,
    PRIMARY KEY (user_id, category));
"""


# ------------------------------------------------------------------ ML categoriser
_models = {}


def get_model(uid):
    """Seed vocabulary + the user's own labelled history → per-user text classifier (cached until data changes)."""
    rows = query("SELECT description, category FROM transactions WHERE user_id=? ORDER BY id DESC LIMIT 2000", (uid,))
    key = (uid, len(rows), rows[0]["description"] if rows else "")
    if key in _models:
        return _models[key]
    X, y = [], []
    for cat, words in SEED_TEXT.items():
        w = words.split()
        for i in range(len(w)):
            X.append(" ".join(w[i:i + 2]))
            y.append(cat)
    for r in rows:
        X += [r["description"]] * 3  # user's history weighs more than seed text
        y += [r["category"]] * 3
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True),
                          LogisticRegression(max_iter=2000, C=6))
    model.fit(X, y)
    _models.clear()
    _models[key] = model
    return model


def predict_category(uid, text):
    m = get_model(uid)
    proba = m.predict_proba([text.lower()])[0]
    order = np.argsort(proba)[::-1][:3]
    return [(m.classes_[i], float(proba[i])) for i in order]


# ------------------------------------------------------------------ demo data
def seed_demo(uid):
    rng = random.Random(7)
    today = date.today()
    samples = {
        "Food & Dining": (["Swiggy order", "Zomato dinner", "Starbucks coffee", "Dominos pizza", "Biryani lunch"], 150, 900),
        "Groceries": (["BigBasket groceries", "DMart shopping", "Blinkit milk & eggs", "Vegetables market"], 300, 2500),
        "Transport": (["Uber ride", "Ola cab", "Petrol refill", "Metro card recharge"], 80, 1800),
        "Shopping": (["Amazon order", "Myntra clothes", "Flipkart headphones"], 500, 4500),
        "Bills & Utilities": (["Electricity bill", "Airtel broadband", "Jio recharge"], 300, 2200),
        "Entertainment": (["Netflix subscription", "PVR movie", "Spotify premium"], 120, 900),
        "Health": (["Apollo pharmacy", "Cult fit membership"], 200, 1800),
        "Education": (["Udemy course", "Coursera certificate"], 450, 3000),
    }
    for m in range(5, -1, -1):
        mstart = (today.replace(day=1) - timedelta(days=30 * m)).replace(day=1)
        execute("INSERT INTO transactions(user_id,date,description,amount,kind,category,ai_confidence) VALUES(?,?,?,?,?,?,?)",
                (uid, mstart.isoformat(), "Salary credited", 85000, "income", "Salary", 0.99))
        execute("INSERT INTO transactions(user_id,date,description,amount,kind,category,ai_confidence) VALUES(?,?,?,?,?,?,?)",
                (uid, (mstart + timedelta(days=2)).isoformat(), "House rent", 18000, "expense", "Rent", 0.98))
        if rng.random() < .6:
            execute("INSERT INTO transactions(user_id,date,description,amount,kind,category,ai_confidence) VALUES(?,?,?,?,?,?,?)",
                    (uid, (mstart + timedelta(days=12)).isoformat(), "Freelance project payment", rng.randint(8, 25) * 1000, "income", "Freelance", 0.93))
        for _ in range(rng.randint(26, 34)):
            cat = rng.choice(list(samples))
            names, lo, hi = samples[cat]
            d = mstart + timedelta(days=rng.randint(0, 27))
            if d > today:
                continue
            execute("INSERT INTO transactions(user_id,date,description,amount,kind,category,ai_confidence) VALUES(?,?,?,?,?,?,?)",
                    (uid, d.isoformat(), rng.choice(names), round(rng.uniform(lo, hi), -1), "expense", cat, round(rng.uniform(.8, .99), 2)))
    # one deliberate anomaly this month
    execute("INSERT INTO transactions(user_id,date,description,amount,kind,category,ai_confidence) VALUES(?,?,?,?,?,?,?)",
            (uid, today.isoformat(), "Amazon order - new phone", 28000, "expense", "Shopping", 0.91))
    for cat, lim in [("Food & Dining", 6000), ("Groceries", 8000), ("Transport", 5000), ("Shopping", 10000),
                     ("Entertainment", 2000), ("Bills & Utilities", 4500)]:
        execute("INSERT INTO budgets VALUES(?,?,?)", (uid, cat, lim))


app = create_app(__name__, UI, SCHEMA, seed_demo=seed_demo)


# ------------------------------------------------------------------ analytics helpers
def month_key(d):
    return d[:7]


def summary(uid):
    tx = query("SELECT * FROM transactions WHERE user_id=? ORDER BY date", (uid,))
    this_m = date.today().strftime("%Y-%m")
    prev_m = (date.today().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    agg = lambda m, k: sum(t["amount"] for t in tx if t["date"].startswith(m) and t["kind"] == k)
    inc, exp = agg(this_m, "income"), agg(this_m, "expense")
    pinc, pexp = agg(prev_m, "income"), agg(prev_m, "expense")
    balance = sum(t["amount"] if t["kind"] == "income" else -t["amount"] for t in tx)

    months = sorted({month_key(t["date"]) for t in tx})[-6:]
    trend = {"labels": [datetime.strptime(m, "%Y-%m").strftime("%b") for m in months],
             "income": [agg(m, "income") for m in months], "expense": [agg(m, "expense") for m in months]}
    cats = {}
    for t in tx:
        if t["kind"] == "expense" and t["date"].startswith(this_m):
            cats[t["category"]] = cats.get(t["category"], 0) + t["amount"]

    # anomaly detection: z-score of each expense vs that category's history
    anomalies = []
    by_cat = {}
    for t in tx:
        if t["kind"] == "expense":
            by_cat.setdefault(t["category"], []).append(t["amount"])
    for t in tx[::-1]:
        if t["kind"] != "expense" or not t["date"].startswith(this_m):
            continue
        hist = np.array(by_cat[t["category"]])
        if len(hist) >= 5 and hist.std() > 0:
            z = (t["amount"] - hist.mean()) / hist.std()
            if z > 2.5:
                anomalies.append({**t, "z": round(float(z), 1), "avg": round(float(hist.mean()))})

    # forecast next month expense: linear regression on monthly totals
    forecast = None
    full = [agg(m, "expense") for m in months[:-1]]
    if len(full) >= 3:
        xs = np.arange(len(full))
        slope, icpt = np.polyfit(xs, full, 1)
        forecast = max(0, float(slope * len(full) + icpt))

    budgets = query("SELECT * FROM budgets WHERE user_id=?", (uid,))
    for b in budgets:
        b["spent"] = cats.get(b["category"], 0)
        b["pct"] = round(100 * b["spent"] / b["monthly_limit"], 1) if b["monthly_limit"] else 0
    return dict(balance=balance, income=inc, expense=exp, savings_rate=round(100 * (inc - exp) / inc, 1) if inc else 0,
                inc_delta=round(100 * (inc - pinc) / pinc, 1) if pinc else 0, exp_delta=round(100 * (exp - pexp) / pexp, 1) if pexp else 0,
                trend=trend, cats=dict(sorted(cats.items(), key=lambda kv: -kv[1])), anomalies=anomalies[:5],
                forecast=forecast, budgets=sorted(budgets, key=lambda b: -b["pct"]), recent=tx[::-1][:7], count=len(tx))


# ------------------------------------------------------------------ pages
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", s=summary(g.user["id"]), CATS=CATEGORIES, page_title="Dashboard")


@app.route("/transactions")
@login_required
def transactions():
    return render_template("transactions.html", CATS=CATEGORIES, INCOME=sorted(INCOME_CATS), page_title="Transactions")


@app.route("/budgets", methods=["GET", "POST"])
@login_required
def budgets():
    if request.method == "POST":
        for cat in CATEGORIES:
            v = request.form.get(cat, "").strip()
            if v:
                execute("INSERT INTO budgets VALUES(?,?,?) ON CONFLICT(user_id,category) DO UPDATE SET monthly_limit=excluded.monthly_limit",
                        (g.user["id"], cat, float(v)))
            else:
                execute("DELETE FROM budgets WHERE user_id=? AND category=?", (g.user["id"], cat))
        from flask import flash, redirect, url_for
        flash("Budgets saved.", "success")
        return redirect(url_for("budgets"))
    s = summary(g.user["id"])
    current = {b["category"]: b for b in s["budgets"]}
    return render_template("budgets.html", s=s, current=current, CATS=CATEGORIES, INCOME=INCOME_CATS, page_title="Budgets")


@app.route("/insights")
@login_required
def insights():
    uid = g.user["id"]
    tx = query("SELECT * FROM transactions WHERE user_id=? AND kind='expense'", (uid,))
    dow = [0] * 7
    for t in tx:
        dow[datetime.strptime(t["date"], "%Y-%m-%d").weekday()] += t["amount"]
    top = query("SELECT description, COUNT(*) n, SUM(amount) total FROM transactions WHERE user_id=? AND kind='expense' "
                "GROUP BY lower(description) ORDER BY total DESC LIMIT 8", (uid,))
    return render_template("insights.html", s=summary(uid), dow=dow, top=top, CATS=CATEGORIES, page_title="AI Insights")


# ------------------------------------------------------------------ JSON API (used live by the UI)
@app.get("/api/transactions")
@login_required
def api_list():
    q, cat = request.args.get("q", "").lower(), request.args.get("category", "")
    sql, args = "SELECT * FROM transactions WHERE user_id=?", [g.user["id"]]
    if q:
        sql += " AND lower(description) LIKE ?"
        args.append(f"%{q}%")
    if cat:
        sql += " AND category=?"
        args.append(cat)
    return jsonify(query(sql + " ORDER BY date DESC, id DESC LIMIT 500", args))


@app.post("/api/transactions")
@login_required
def api_add():
    d = request.get_json(force=True)
    desc = str(d.get("description", "")).strip()[:120]
    try:
        amount = abs(float(d.get("amount", 0)))
    except ValueError:
        amount = 0
    if not desc or amount <= 0:
        return jsonify(error="Description and a positive amount are required"), 400
    cat = d.get("category") or predict_category(g.user["id"], desc)[0][0]
    if cat not in CATEGORIES:
        cat = "Other"
    kind = "income" if cat in INCOME_CATS else "expense"
    conf = d.get("confidence")
    tid = execute("INSERT INTO transactions(user_id,date,description,amount,kind,category,ai_confidence) VALUES(?,?,?,?,?,?,?)",
                  (g.user["id"], d.get("date") or date.today().isoformat(), desc, amount, kind, cat, conf))
    return jsonify(query("SELECT * FROM transactions WHERE id=?", (tid,), one=True)), 201


@app.patch("/api/transactions/<int:tid>")
@login_required
def api_update(tid):
    cat = request.get_json(force=True).get("category")
    if cat not in CATEGORIES:
        return jsonify(error="Unknown category"), 400
    execute("UPDATE transactions SET category=?, kind=? WHERE id=? AND user_id=?",
            (cat, "income" if cat in INCOME_CATS else "expense", tid, g.user["id"]))
    return jsonify(ok=True, message="Category updated — the AI model will learn from this.")


@app.delete("/api/transactions/<int:tid>")
@login_required
def api_delete(tid):
    execute("DELETE FROM transactions WHERE id=? AND user_id=?", (tid, g.user["id"]))
    return jsonify(ok=True)


@app.get("/api/predict")
@login_required
def api_predict():
    text = request.args.get("text", "").strip()
    if len(text) < 2:
        return jsonify(predictions=[])
    return jsonify(predictions=[{"category": c, "confidence": round(p, 3), "icon": CATEGORIES[c][0], "color": CATEGORIES[c][1]}
                                for c, p in predict_category(g.user["id"], text)])


@app.get("/api/export.csv")
@login_required
def api_export():
    rows = query("SELECT date, description, amount, kind, category FROM transactions WHERE user_id=? ORDER BY date", (g.user["id"],))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["date", "description", "amount", "kind", "category"])
    w.writeheader()
    w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=finsight_transactions.csv"})


@app.post("/api/import")
@login_required
def api_import():
    f = request.files.get("file")
    if not f:
        return jsonify(error="Choose a CSV file"), 400
    reader = csv.DictReader(io.StringIO(f.read().decode("utf-8", errors="ignore")))
    n = 0
    for r in reader:
        try:
            desc, amount = r["description"].strip(), abs(float(r["amount"]))
        except (KeyError, ValueError):
            continue
        cat = r.get("category") if r.get("category") in CATEGORIES else predict_category(g.user["id"], desc)[0][0]
        execute("INSERT INTO transactions(user_id,date,description,amount,kind,category) VALUES(?,?,?,?,?,?)",
                (g.user["id"], r.get("date") or date.today().isoformat(), desc[:120], amount,
                 "income" if cat in INCOME_CATS else "expense", cat))
        n += 1
    return jsonify(ok=True, imported=n)


if __name__ == "__main__":
    app.run(debug=True)
