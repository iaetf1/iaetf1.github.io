# How I Turned a Sunday Side Project into a Deployed Machine Learning API

**The story of a weekend of struggle (and triumph) that taught me more about ML in production than 6 months of online courses.**

---

## The backstory — How it all started

A few months ago, on a Saturday morning, I was scrolling through LinkedIn over my coffee when I stumbled upon a post that hit a nerve:

> "Data science is 80% Jupyter notebooks that will never see the light of day."

The guy had 2,000 likes. And he was right.

I had just finished a small side project on wine quality prediction — a classic, nothing revolutionary. A Random Forest, physicochemical features, a solid R². My notebook was clean, my charts were pretty. I was feeling good about myself.

And then I asked myself the question: **if a winemaker friend asked me to test his wine with my model, how would I do it?** Email him my notebook? Ask him to install Python, Jupyter, scikit-learn? Obviously, that's absurd.

I decided to dedicate my weekend to a challenge: **take this model and make it accessible to anyone through a simple URL**. With a real API, a real database, real tests, real automated deployment.

What was supposed to be "a quick thing" took the entire weekend. But I learned more in those two days than in months of tutorials. And that's exactly the journey I'm going to walk you through here, step by step.

---

## What I built (and what you'll learn)

By the end of that weekend, I had:

- An ML model accessible via a **REST API** that anyone can call from a browser
- A **PostgreSQL database** that records every prediction (inputs, outputs, timestamps)
- **Automated tests** that verify nothing is broken with every change
- A **CI/CD pipeline**: with every `git push`, tests run automatically, and if everything passes, the app deploys itself
- The whole thing **live on Hugging Face Spaces**, publicly accessible

Here's the full architecture, which I scribbled on a sticky note before writing any code (always sketch before you code, always):

```
┌─────────────────────────────────────────────────────────┐
│                       USER                              │
│            (browser, curl, Postman, app)                │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP request (JSON)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI API                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Pydantic │  │   Endpoints  │  │   ML Model (.pkl) │  │
│  │(data     │→ │  /predict    │→ │  loaded at         │  │
│  │ valid.)  │  │  /health     │  │  startup           │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │ Log inputs/outputs
                     ▼
┌─────────────────────────────────────────────────────────┐
│              PostgreSQL (Database)                       │
│  ┌───────────────┐  ┌────────────────────────────────┐  │
│  │  wines table   │  │  predictions table             │  │
│  │  (dataset)     │  │  (input, output, timestamp)    │  │
│  └───────────────┘  └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 CI/CD Pipeline                           │
│  git push → GitHub Actions → Tests → Auto deployment    │
│                                    → Hugging Face Spaces │
└─────────────────────────────────────────────────────────┘
```

That's a lot of building blocks. But each one has a purpose, and we'll build them one by one. Don't skip steps — that's the mistake I made on Saturday morning, and I lost 3 hours debugging an issue caused by a poorly configured Git setup.

---

## Saturday morning — Laying the foundations with Git

### My beginner mistake

My first instinct was to create a `main.py` file and start coding the API right away. Bad idea. After 2 hours, I had code that "sort of worked", no history, no structure, and when I wanted to roll back after a major mistake... I couldn't.

I deleted everything and started over. This time, doing things in the right order.

### What is Git, and why can't I live without it?

If you've never used Git, here's the simplest explanation I know.

Imagine you're writing a thesis in Word. You save it. You edit a chapter. You save again. But a week later, you realize the old version of chapter 3 was better. Too late — you've overwritten the file.

**Git is a system that keeps ALL versions of ALL your files.** Each "save" (called a **commit**) is a complete snapshot of your project at a given point in time. You can travel through time, go back to any version, create parallel branches to test ideas risk-free, and merge everything when it's ready.

It's not just a backup tool. It's a **logbook** for your project. Each commit tells a story: "I added the /predict endpoint", "I fixed the validation bug", "I optimized the SQL query".

![Illustration of how Git works — each commit is a snapshot in time](https://blog.git-init.com/content/images/size/w1600/2021/08/commit-snapshots.001.jpeg)

> **Going further with Git:**
> - [The official Git book](https://git-scm.com/book/en/v2) — the reference
> - [Learn Git Branching](https://learngitbranching.js.org/) — an interactive game to understand branches (addictive)
> - [Oh Shit, Git!?!](https://ohshitgit.com/) — for when things go wrong (it happens to everyone)

### GitHub: Git, but in the cloud

**Git** is the tool running on your machine. **GitHub** is the online platform that hosts your code and enables collaboration. Think of Git as your local text editor, and GitHub as Google Docs — the shared version accessible from anywhere.

![](https://data-flair.training/blogs/wp-content/uploads/sites/2/2023/09/difference-git-vs-github.webp)

We'll need GitHub for two reasons:
1. Host our code (and make it accessible)
2. Use **GitHub Actions** to automate testing and deployment (we'll get to that later)

> Create an account at [github.com](https://github.com) if you haven't already.

### Initialize the project properly

```bash
# Create the project directory
mkdir wine-quality-api
cd wine-quality-api

# Initialize Git
git init

# Create the Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Create the basic structure
mkdir -p app tests data docs scripts .github/workflows
touch app/__init__.py app/main.py app/models.py app/database.py
touch app/schemas.py app/ml_model.py
touch tests/__init__.py tests/test_api.py tests/test_model.py
touch requirements.txt .gitignore README.md .env.example
```

### Why this structure?

```
wine-quality-api/
│
├── app/                    # Application source code
│   ├── __init__.py         # Marks the folder as a Python package
│   ├── main.py             # FastAPI entry point
│   ├── schemas.py          # Pydantic schemas (data validation)
│   ├── models.py           # SQLAlchemy models (DB tables)
│   ├── database.py         # PostgreSQL connection and config
│   └── ml_model.py         # ML model loading and inference
│
├── tests/                  # Automated tests
│   ├── __init__.py
│   ├── test_api.py         # API endpoint tests
│   └── test_model.py       # ML model tests
│
├── scripts/                # Utility scripts (DB init, etc.)
├── data/                   # Data (dataset, trained model)
├── docs/                   # Additional documentation
├── .github/workflows/      # GitHub Actions CI/CD pipeline
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Files ignored by Git
├── .env.example            # Environment variables template
└── README.md               # Main documentation
```

The principle here is **separation of concerns**. Business code lives in `app/`. Tests live in `tests/`. Utility scripts in `scripts/`. When someone discovers your project, they immediately know where to look for what.

### The .gitignore — Files Git should ignore

```gitignore
# Virtual environment and secrets
venv/
.env

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# IDE
.vscode/
.idea/

# Large data files and serialized models
*.pkl
*.joblib
data/*.csv

# OS
.DS_Store
Thumbs.db

# Test reports
htmlcov/
.coverage
```

**Critical point: NEVER commit your secrets.** Database passwords, API tokens, private keys — all of that goes in a `.env` file listed in `.gitignore`. If you commit a password to GitHub, even if you delete it afterwards, it stays in the Git history. Bots constantly scan GitHub for exposed credentials.

> **Going further:**
> - [gitignore.io](https://www.toptal.com/developers/gitignore) — a `.gitignore` generator based on your stack
> - [GitHub Security Best Practices](https://docs.github.com/en/code-security) — security best practices

### Dependencies

```txt
# requirements.txt
fastapi==0.115.0
uvicorn==0.30.0
pydantic==2.9.0
sqlalchemy==2.0.35
psycopg2-binary==2.9.9
scikit-learn==1.5.2
pandas==2.2.3
joblib==1.4.2
pytest==8.3.3
pytest-cov==5.0.0
httpx==0.27.2
python-dotenv==1.0.1
```

```bash
pip install -r requirements.txt
```

Why pin dependency versions (with `==`)? Because if you install `fastapi` without a version, pip will grab the latest. And in 6 months, when someone clones your project, the latest version might be incompatible with your code. A pinned `requirements.txt` guarantees your project works the same way on any machine.

### Conventional Commits — A readable history

```bash
git add .
git commit -m "feat: project initialization - structure and dependencies"
```

The `feat:` prefix isn't just cosmetic. It's a convention called [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) that makes your history scannable at a glance:

| Prefix | Usage | Example |
|--------|-------|---------|
| `feat:` | New feature | `feat: add /predict endpoint` |
| `fix:` | Bug fix | `fix: pH field validation` |
| `docs:` | Documentation | `docs: update README` |
| `test:` | Tests | `test: add input validation tests` |
| `refactor:` | Refactoring | `refactor: extract DB logic` |
| `ci:` | CI/CD | `ci: add GitHub Actions pipeline` |

### Branches — Working without risk

Before coding each new feature, I create a **dedicated branch**. It's like creating a parallel copy of the project to experiment with. If it works, I merge. If it's a disaster, I delete the branch, and `main` hasn't moved.

![](https://user-images.githubusercontent.com/1256329/117236177-33599100-adf6-11eb-967c-5ef7898b55dc.png)

```bash
# Create a branch for the first feature
git checkout -b feature/api-fastapi

# Work, commit...

# When ready, merge into main
git checkout main
git merge feature/api-fastapi

# Tag the version
git tag -a v0.1.0 -m "First version of the API"
```

> **Going further on branches:**
> - [GitFlow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) — the most widely used branching workflow in the industry

---

## Saturday late morning — Training and saving the model

### The Wine Quality dataset

For this side project, I used the [Wine Quality dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). It's a classic in the ML world: 1,599 Portuguese red wines, each described by 11 physicochemical measurements (acidity, residual sugar, alcohol, etc.) and a quality rating from 0 to 10 given by expert tasters.

Why this dataset? It's simple, understandable by anyone, free, and the model it produces is lightweight enough to be deployed for free on Hugging Face Spaces.

### Training the model (the prerequisite)

This training script isn't the core of the project — it's a prerequisite. In a real-world scenario, you'd already have a trained model that needs to be put into production. Let's get this done quickly:

```python
# train_model.py — run ONCE, outside of the API
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---- Load data ----
url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)
df = pd.read_csv(url, sep=";")
print(f"Dataset loaded: {df.shape[0]} wines, {df.shape[1]} columns")

# ---- Separate features and target ----
X = df.drop("quality", axis=1)
y = df["quality"]

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Train a Random Forest ----
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R²:  {r2_score(y_test, y_pred):.3f}")

# ---- Save the model AND the feature names ----
joblib.dump(model, "data/wine_model.joblib")
joblib.dump(list(X.columns), "data/feature_names.joblib")

print("Model saved to data/wine_model.joblib")
```

**One detail that cost me time**: I saved the model but not the list of feature names. The result? When the API received data as JSON (where key order isn't guaranteed), I couldn't put them in the right order for the model. Since then, I always save the model's metadata alongside the model itself.

```bash
python train_model.py
git add train_model.py
git commit -m "feat: Random Forest training script"
```

---

## Saturday afternoon — The API with FastAPI

### Demystifying REST APIs

This is the point in the day where I had to stop coding and step back. Because before building an API, you need to understand what one actually is.

#### What is an API, concretely?

The term API (Application Programming Interface) sounds intimidating, but the concept is dead simple.

**The restaurant analogy.** You're at a restaurant. You don't walk into the kitchen to cook your own meal. You place an order with the waiter, who passes it to the kitchen and brings back the result.

![](https://media.geeksforgeeks.org/wp-content/uploads/20230216170349/What-is-an-API.png)

- **You** = the client (a browser, a Python script, a mobile app)
- **The waiter** = the API (receives requests and returns results)
- **The kitchen** = the ML model (where the actual work happens)
- **The menu** = the API documentation (what you can order)
- **Your order** = an HTTP request
- **The dish served** = the response (in JSON format)

![Simplified diagram of a REST API — the client sends a request, the server processes it and returns a response](https://voyager.postman.com/illustration/diagram-what-is-an-api-postman-illustration.svg)

#### HTTP methods — The basic vocabulary

When you browse the Internet, your browser sends HTTP requests to servers. Each request has a **method** that indicates what it wants to do:

![](https://pbs.twimg.com/media/F-eVhBbaIAAECzd.png)

| Method | What it does | Restaurant analogy |
|--------|-------------|-------------------|
| `GET` | Retrieve data | "Show me the menu" |
| `POST` | Send data to create something | "I'll order this dish" |
| `PUT` | Update existing data | "Change my order" |
| `DELETE` | Remove data | "Cancel my order" |

For our project, we'll use:
- `GET /` → check that the API is alive (health check)
- `POST /predict` → send wine characteristics and receive the prediction

The exchange format is **JSON** (JavaScript Object Notation). It looks like a Python dictionary:

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "alcohol": 9.4
}
```

> **Going further on REST APIs:**
> - [What is a REST API?](https://www.redhat.com/en/topics/api/what-is-a-rest-api) — Red Hat, clear and comprehensive
> - [HTTP Status Codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status) — Mozilla's doc on response codes (200, 404, 500...)
> - [Postman](https://www.postman.com/) — the go-to tool for testing APIs (free)

#### Pydantic — Your API's bouncer

When someone sends data to your API, they can send anything. Text where you expect a number. A missing field. An absurd value (a pH of 47, seriously?).

**Pydantic** is the Python library that acts as a **bouncer at the door**. You define a "contract": which fields are expected, which types, which constraints. If the data doesn't meet the contract, the request is rejected with a clear error message — before your model is even touched.

```
Incoming data → Pydantic checks → ✅ OK → ML Model
                                → ❌ Rejected (422 error)
```

This is called **input data validation**, and it's absolutely non-negotiable in a production API. Without it, your model receives garbage data and produces nonsensical results — or worse, crashes.

> **Going further on Pydantic:**
> - [Official Pydantic v2 documentation](https://docs.pydantic.dev/latest/) — comprehensive and well-written
> - [Pydantic is all you need](https://blog.det.life/pydantic-is-all-you-need-f3ce1e15de5c) — a good introductory article on Medium

### Why FastAPI over other frameworks?

There are dozens of Python frameworks for building APIs: Flask, Django REST Framework, Falcon, Bottle... I chose FastAPI for this side project for four reasons:

1. **Concise syntax** — far less boilerplate than Flask or Django
2. **Built-in validation** — Pydantic is native, no need to add it manually
3. **Automatic documentation** — Swagger UI is generated automatically without writing any extra code (more on this below — it's a game-changer)
4. **Performance** — built on Starlette and Python's async coroutines, it's one of the fastest frameworks out there

> **Comparing frameworks:**
> - [FastAPI vs Flask — What's the difference?](https://fastapi.tiangolo.com/alternatives/) — by FastAPI's creator
> - [Official FastAPI documentation](https://fastapi.tiangolo.com/) — a progressive and very well-made tutorial

### The code — Step by step

#### 1. Pydantic schemas (the data contract)

This is the first thing I coded. Before even the API itself. Why? Because defining schemas forces you to think: what data will my API receive? What types? What constraints? It's a design exercise, not just code.

```python
# app/schemas.py
from pydantic import BaseModel, Field
from datetime import datetime


class WineFeatures(BaseModel):
    """
    Validation schema for wine physicochemical characteristics.
    Each field has a type, min/max bounds, and an example.
    If data doesn't match this contract, the request is rejected.
    """

    fixed_acidity: float = Field(
        ...,          # "..." means "required field"
        ge=0, le=20,  # ge = greater or equal, le = less or equal
        description="Fixed acidity (g tartaric acid/L)",
        examples=[7.4],
    )
    volatile_acidity: float = Field(
        ..., ge=0, le=2,
        description="Volatile acidity (g acetic acid/L)",
        examples=[0.7],
    )
    citric_acid: float = Field(
        ..., ge=0, le=1.5,
        description="Citric acid (g/L)",
        examples=[0.0],
    )
    residual_sugar: float = Field(
        ..., ge=0, le=20,
        description="Residual sugar (g/L)",
        examples=[1.9],
    )
    chlorides: float = Field(
        ..., ge=0, le=1,
        description="Chlorides (g sodium chloride/L)",
        examples=[0.076],
    )
    free_sulfur_dioxide: float = Field(
        ..., ge=0, le=100,
        description="Free sulfur dioxide (mg/L)",
        examples=[11.0],
    )
    total_sulfur_dioxide: float = Field(
        ..., ge=0, le=400,
        description="Total sulfur dioxide (mg/L)",
        examples=[34.0],
    )
    density: float = Field(
        ..., ge=0.9, le=1.1,
        description="Density (g/cm³)",
        examples=[0.9978],
    )
    ph: float = Field(
        ..., ge=2, le=5,
        description="Wine pH",
        examples=[3.51],
    )
    sulphates: float = Field(
        ..., ge=0, le=2.5,
        description="Sulphates (g potassium sulphate/L)",
        examples=[0.56],
    )
    alcohol: float = Field(
        ..., ge=8, le=16,
        description="Alcohol content (%vol)",
        examples=[9.4],
    )


class PredictionResponse(BaseModel):
    """What the API returns after a prediction."""
    predicted_quality: float = Field(
        ..., description="Predicted quality score (0-10)"
    )
    model_version: str = Field(
        ..., description="Model version used"
    )
    timestamp: datetime = Field(
        ..., description="Prediction timestamp"
    )


class HealthResponse(BaseModel):
    """What the API returns on the health check endpoint."""
    status: str
    model_loaded: bool
    database_connected: bool
```

Take a close look at the `alcohol` field:

```python
alcohol: float = Field(..., ge=8, le=16, ...)
```

This means: "the `alcohol` field is a required decimal number between 8 and 16". If someone sends `"alcohol": "lots"` or `"alcohol": 99`, Pydantic automatically rejects the request with a clear error message. You don't have to code anything for that.

#### 2. Loading the ML model

```python
# app/ml_model.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path("data/wine_model.joblib")
FEATURES_PATH = Path("data/feature_names.joblib")
MODEL_VERSION = "1.0.0"


class WineQualityModel:
    """
    Wraps the ML model.

    CRITICAL POINT: the model is loaded ONCE at API startup,
    then reused for all requests.

    Loading a model per request = performance disaster.
    A 200 MB model reloaded 100 times per second kills a server.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_loaded = False

    def load(self):
        """Loads the model from disk."""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.feature_names = joblib.load(FEATURES_PATH)
            self.is_loaded = True
            print(f"Model v{MODEL_VERSION} loaded ({MODEL_PATH})")
        except FileNotFoundError as e:
            print(f"ERROR: model file not found — {e}")
            self.is_loaded = False

    def predict(self, features: dict) -> float:
        """
        Prediction from a feature dictionary.

        1. Checks the model is loaded
        2. Converts dict to DataFrame in the correct column order
        3. Calls model.predict()
        4. Clips the result between 0 and 10
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        # Column order MUST match training
        df = pd.DataFrame([features])[self.feature_names]
        prediction = self.model.predict(df)[0]

        return float(np.clip(prediction, 0, 10))


# Singleton — shared across the entire application
wine_model = WineQualityModel()
```

**The mistake 90% of beginners make**: loading the model on every request. I almost did it too. When you write `model = joblib.load(...)` inside the prediction function, your API reloads the entire model on every call. If the model weighs 200 MB and you have 50 requests per second, you're loading 10 GB/second from disk. The server collapses.

The solution: load the model **once** at startup, and reuse the in-memory object.

#### 3. The API itself

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone

from app.schemas import WineFeatures, PredictionResponse, HealthResponse
from app.ml_model import wine_model, MODEL_VERSION
from app.database import log_prediction, check_db_connection

# ---- Create the application ----
app = FastAPI(
    title="Wine Quality Prediction API",
    description=(
        "Predicts red wine quality based on "
        "physicochemical properties. "
        "Model: Random Forest trained on the UCI Wine Quality dataset."
    ),
    version=MODEL_VERSION,
)


# ---- Startup event ----
@app.on_event("startup")
def startup_event():
    """The model loads once, when the API starts."""
    wine_model.load()


# ---- Endpoint: Health Check ----
@app.get("/", response_model=HealthResponse)
def health_check():
    """
    Checks that the API, model, and database are operational.
    Always the first endpoint to test.
    """
    return HealthResponse(
        status="healthy" if wine_model.is_loaded else "degraded",
        model_loaded=wine_model.is_loaded,
        database_connected=check_db_connection(),
    )


# ---- Endpoint: Prediction ----
@app.post("/predict", response_model=PredictionResponse)
def predict_quality(wine: WineFeatures):
    """
    Predicts the quality of a red wine.

    Full workflow:
    1. Pydantic validates the 11 features (automatically)
    2. The ML model makes the prediction
    3. The result is logged to the database
    4. The JSON response is sent back to the client
    """
    # Check model availability
    if not wine_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Try again later.",
        )

    # Convert Pydantic data to Python dict
    features = wine.model_dump()

    # Prediction
    try:
        predicted_quality = wine_model.predict(features)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}",
        )

    timestamp = datetime.now(timezone.utc)

    # Log to database (don't block if it fails)
    try:
        log_prediction(
            input_data=features,
            output_data=predicted_quality,
            timestamp=timestamp,
        )
    except Exception as e:
        # Log warning but don't crash the API
        print(f"Warning: Could not log to DB: {e}")

    return PredictionResponse(
        predicted_quality=round(predicted_quality, 2),
        model_version=MODEL_VERSION,
        timestamp=timestamp,
    )
```

### The "wow" moment — Swagger UI

Here's the moment that blew me away on Saturday afternoon. I launched the API:

```bash
uvicorn app.main:app --reload
```

And I opened `http://127.0.0.1:8000/docs` in my browser.

**Without writing a single extra line of code**, FastAPI had generated a complete interactive documentation page. Every endpoint was documented with its parameters, types, and examples. And best of all, I could test my endpoints directly from the browser — click "Try it out", fill in the fields, click "Execute", and see the response in real time.

This is **Swagger UI**, and it's built natively into FastAPI through the OpenAPI specification. All the information comes from your Pydantic schemas and your docstrings — which is why we took the time to write them well.

![FastAPI automatically generates interactive Swagger UI documentation accessible at /docs](https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png)

> **Going further:**
> - [FastAPI — Interactive tutorial](https://fastapi.tiangolo.com/tutorial/) — the best starting point
> - [Swagger/OpenAPI Specification](https://swagger.io/specification/) — the technical spec behind Swagger UI

```bash
git add .
git commit -m "feat: FastAPI API with /predict and / (health) endpoints"
```

---

## Saturday evening — The PostgreSQL database

### Why do I need a database?

At this point, my API was working. I could have stopped there. But I imagined a scenario: my API has been running for 3 months, and one day the predictions go haywire. Quality scores completely off the mark.

Without history, I'm blind. I don't know if the model has a problem, or if the input data has changed (what's called **data drift** — a fundamental concept in MLOps).

With a database recording **every request** (what went in, what came out, when), I can:

- **Audit** any past prediction
- **Detect** anomalies in the input data
- **Retrain** the model on real production data
- **Measure** actual API performance (response times, volume, etc.)

This is the difference between a weekend prototype and a system you can maintain long-term.

### SQL, PostgreSQL, SQLAlchemy — Untangling the terms

When I started, these three words blurred together in my head. Here's the distinction:

**SQL** (Structured Query Language): it's a **language**, like Python or JavaScript. But instead of manipulating variables and functions, it manipulates data in tables. `SELECT * FROM wines WHERE quality > 7` is a SQL query that retrieves all wines with a rating above 7.

**PostgreSQL**: it's a **software** — the database engine itself. It's what stores your data on disk, manages transactions, indexes, and access rights. Think of it as the engine of a car: you don't see it, but it powers everything. PostgreSQL is free, open source, and used by Instagram, Spotify, and thousands of companies.

**SQLAlchemy**: it's an **ORM** (Object-Relational Mapper) — a Python library that bridges Python and PostgreSQL. Instead of writing raw SQL in your Python code, you manipulate Python objects that represent your tables, and SQLAlchemy translates that into SQL. It's an interpreter between two worlds.

```
Without ORM:  cursor.execute("INSERT INTO predictions (quality) VALUES (6.5)")
With ORM:     db.add(Prediction(quality=6.5))  ← plain Python
```

> **Going further:**
> - [PostgreSQL — Official documentation](https://www.postgresql.org/docs/) — the reference
> - [SQLAlchemy — ORM Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/) — the official tutorial (very thorough)
> - [What is an ORM?](https://www.fullstackpython.com/object-relational-mappers-orms.html) — simple explanation

### Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install postgresql postgresql-contrib

# macOS (with Homebrew)
brew install postgresql@16 && brew services start postgresql@16

# Verify installation
psql --version

# Create the database and user
sudo -u postgres psql -c "CREATE DATABASE wine_quality_db;"
sudo -u postgres psql -c "CREATE USER wine_user WITH PASSWORD 'wine_secret_pwd';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE wine_quality_db TO wine_user;"
```

### The environment file

```bash
# .env.example (commit this — it's the TEMPLATE)
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# .env (DO NOT commit — this has the REAL secrets)
DATABASE_URL=postgresql://wine_user:wine_secret_pwd@localhost:5432/wine_quality_db
```

### Design the schema before coding

Before touching any code, I grabbed a sheet of paper and sketched my database schema. Two tables:

```
┌──────────────────────────────┐     ┌────────────────────────────────┐
│         wines                │     │        predictions             │
├──────────────────────────────┤     ├────────────────────────────────┤
│ id          SERIAL (PK)     │     │ id              SERIAL (PK)   │
│ fixed_acidity    FLOAT      │     │ input_data      JSONB         │
│ volatile_acidity FLOAT      │     │ predicted_quality FLOAT       │
│ citric_acid      FLOAT      │     │ model_version   VARCHAR(20)   │
│ residual_sugar   FLOAT      │     │ response_time_ms FLOAT        │
│ chlorides        FLOAT      │     │ created_at      TIMESTAMP     │
│ free_sulfur_dioxide FLOAT   │     └────────────────────────────────┘
│ total_sulfur_dioxide FLOAT  │
│ density          FLOAT      │
│ ph               FLOAT      │
│ sulphates        FLOAT      │
│ alcohol          FLOAT      │
│ quality          FLOAT      │
│ created_at       TIMESTAMP  │
└──────────────────────────────┘
```

- **wines**: the complete dataset, imported once. Useful for reference and future retraining.
- **predictions**: every prediction made by the API. Input is stored as JSON (flexible if features change), along with the result, model version, and timestamp.

> **Going further on DB design:**
> - [dbdiagram.io](https://dbdiagram.io/) — free tool to draw DB schemas online
> - [Lucidchart](https://www.lucidchart.com/) — for more elaborate UML diagrams

### Implementation with SQLAlchemy

```python
# app/database.py
import os
import json
from datetime import datetime, timezone
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, Float,
    String, DateTime, Text, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fallback.db")

# ---- Connection engine ----
# echo=True to see generated SQL queries (useful for debugging)
engine = create_engine(DATABASE_URL, echo=False)

# ---- Session factory ----
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ---- Base class for models ----
Base = declarative_base()


# =============================================
#  TABLE DEFINITIONS
# =============================================

class Wine(Base):
    """Table containing the full wine dataset."""
    __tablename__ = "wines"

    id = Column(Integer, primary_key=True, index=True)
    fixed_acidity = Column(Float, nullable=False)
    volatile_acidity = Column(Float, nullable=False)
    citric_acid = Column(Float, nullable=False)
    residual_sugar = Column(Float, nullable=False)
    chlorides = Column(Float, nullable=False)
    free_sulfur_dioxide = Column(Float, nullable=False)
    total_sulfur_dioxide = Column(Float, nullable=False)
    density = Column(Float, nullable=False)
    ph = Column(Float, nullable=False)
    sulphates = Column(Float, nullable=False)
    alcohol = Column(Float, nullable=False)
    quality = Column(Float, nullable=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class Prediction(Base):
    """Table that logs every prediction made by the API."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(Text, nullable=False)     # Serialized JSON
    predicted_quality = Column(Float, nullable=False)
    model_version = Column(String(20), nullable=False)
    response_time_ms = Column(Float, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


# =============================================
#  UTILITY FUNCTIONS
# =============================================

def init_db():
    """Creates all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")


@contextmanager
def get_db():
    """
    Context manager for DB sessions.
    Automatically handles commit, rollback on error,
    and connection closing.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def log_prediction(
    input_data: dict, output_data: float, timestamp: datetime
):
    """Records a prediction in the predictions table."""
    with get_db() as db:
        prediction = Prediction(
            input_data=json.dumps(input_data),
            predicted_quality=output_data,
            model_version="1.0.0",
            created_at=timestamp,
        )
        db.add(prediction)


def check_db_connection() -> bool:
    """Tests whether the DB connection works."""
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
```

### Database initialization script

```python
# scripts/init_db.py
"""
Run ONCE to:
1. Create tables in PostgreSQL
2. Insert the Wine Quality dataset into the 'wines' table
"""
import pandas as pd
from app.database import init_db, get_db, Wine


def seed_database():
    """Inserts the dataset into the wines table."""
    url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/wine-quality/winequality-red.csv"
    )
    df = pd.read_csv(url, sep=";")

    # Clean column names
    df.columns = [
        col.strip().replace(" ", "_") for col in df.columns
    ]

    with get_db() as db:
        for _, row in df.iterrows():
            wine = Wine(**row.to_dict())
            db.add(wine)
        print(f"{len(df)} wines inserted into database.")


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    seed_database()
    print("Done.")
```

```bash
python -m scripts.init_db
git add .
git commit -m "feat: PostgreSQL + SQLAlchemy — wines and predictions tables"
```

---

## Sunday morning — Testing with Pytest

### Why do I test? The trapeze artist's safety net.

Sunday morning. I have a working API, a database logging predictions. I could deploy. But Saturday night, I added validation on the `density` field, and it silently broke something. I spent 45 minutes debugging a problem I would have caught in 2 seconds with an automated test.

The analogy I always use: automated tests are the **safety net of a trapeze artist**. The artist can nail 100 tricks in a row. But the 101st time, when they slip, the net is there. Without a net, a single mistake is a catastrophe.

Every time you modify your code, you rerun the tests. All green? Keep going. One red? You know exactly what and where — before the bug reaches production.

### Unit tests vs functional tests

- **Unit test**: tests an isolated function with controlled inputs and outputs. "Does `model.predict()` return a float between 0 and 10?"
- **Functional test** (or integration test): tests the system end-to-end. "When I send a POST to `/predict` with valid data, do I get a correct JSON response with a 200 status?"

Both are necessary. Unit tests catch bugs in individual components. Functional tests catch assembly bugs — when each piece works alone but not together.

> **Going further on testing:**
> - [Pytest documentation](https://docs.pytest.org/) — the reference
> - [Testing Best Practices](https://realpython.com/pytest-python-testing/) — an excellent guide on Real Python

### API tests

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

# TestClient simulates a browser calling our API
client = TestClient(app)


# ---- Reusable test data ----

VALID_WINE = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


class TestHealthEndpoint:
    """The / endpoint should indicate the API is alive."""

    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_response_structure(self):
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_model_is_loaded(self):
        response = client.get("/")
        assert response.json()["model_loaded"] is True


class TestPredictEndpoint:
    """The /predict endpoint should return correct predictions."""

    def test_returns_200_with_valid_data(self):
        response = client.post("/predict", json=VALID_WINE)
        assert response.status_code == 200

    def test_response_contains_quality(self):
        response = client.post("/predict", json=VALID_WINE)
        data = response.json()
        assert "predicted_quality" in data
        assert isinstance(data["predicted_quality"], float)

    def test_quality_in_valid_range(self):
        response = client.post("/predict", json=VALID_WINE)
        quality = response.json()["predicted_quality"]
        assert 0 <= quality <= 10

    def test_response_contains_metadata(self):
        response = client.post("/predict", json=VALID_WINE)
        data = response.json()
        assert "model_version" in data
        assert "timestamp" in data


class TestInputValidation:
    """
    Pydantic should reject invalid data with a 422 status code.
    These tests verify the "bouncer" is doing its job.
    """

    def test_missing_field_rejected(self):
        incomplete = {
            k: v for k, v in VALID_WINE.items() if k != "alcohol"
        }
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_wrong_type_rejected(self):
        bad = {**VALID_WINE, "alcohol": "lots"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_out_of_range_rejected(self):
        bad = {**VALID_WINE, "alcohol": 99.9}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_negative_value_rejected(self):
        bad = {**VALID_WINE, "fixed_acidity": -5.0}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_empty_body_rejected(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422
```

### ML model tests

```python
# tests/test_model.py
import pytest
from app.ml_model import WineQualityModel

SAMPLE_INPUT = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


class TestWineModel:
    """Unit tests for the ML model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Loads the model once for the entire class."""
        self.model = WineQualityModel()
        self.model.load()

    def test_loads_successfully(self):
        assert self.model.is_loaded is True

    def test_prediction_is_float(self):
        result = self.model.predict(SAMPLE_INPUT)
        assert isinstance(result, float)

    def test_prediction_in_range(self):
        result = self.model.predict(SAMPLE_INPUT)
        assert 0 <= result <= 10

    def test_prediction_is_deterministic(self):
        """Same input → same output. Always."""
        r1 = self.model.predict(SAMPLE_INPUT)
        r2 = self.model.predict(SAMPLE_INPUT)
        assert r1 == r2

    def test_unloaded_model_raises_error(self):
        """An unloaded model should raise an explicit error."""
        empty = WineQualityModel()  # no .load()
        with pytest.raises(RuntimeError):
            empty.predict(SAMPLE_INPUT)
```

### Running tests and measuring coverage

```bash
# All tests, with coverage and details
pytest --cov=app --cov-report=term-missing --cov-report=html -v
```

Terminal output:

```
tests/test_api.py::TestHealthEndpoint::test_returns_200             PASSED
tests/test_api.py::TestHealthEndpoint::test_response_structure      PASSED
tests/test_api.py::TestPredictEndpoint::test_returns_200            PASSED
tests/test_api.py::TestPredictEndpoint::test_quality_in_valid_range PASSED
tests/test_api.py::TestInputValidation::test_missing_field_rejected PASSED
tests/test_api.py::TestInputValidation::test_wrong_type_rejected    PASSED
tests/test_model.py::TestWineModel::test_loads_successfully         PASSED
tests/test_model.py::TestWineModel::test_prediction_is_float        PASSED
tests/test_model.py::TestWineModel::test_prediction_is_deterministic PASSED
...

---------- coverage ----------
Name               Stmts   Miss  Cover   Missing
-------------------------------------------------
app/main.py           35      2    94%   48-49
app/ml_model.py       28      3    89%   38-40
app/schemas.py        22      0   100%
app/database.py       45      8    82%   67-74
-------------------------------------------------
TOTAL                130     13    90%
```

**90% coverage.** The HTML report (generated in `htmlcov/index.html`) visually shows which lines are tested (in green) and which aren't (in red). Open it in a browser — it's incredibly satisfying to see green everywhere.

> **Going further:**
> - [pytest-cov documentation](https://pytest-cov.readthedocs.io/) — coverage reporting
> - [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/) — the official testing guide with FastAPI

```bash
git add .
git commit -m "test: full unit and functional test suite (90% coverage)"
```

---

## Sunday afternoon — CI/CD with GitHub Actions

### CI/CD demystified

This is the concept that intimidated me the most when I started. **CI/CD** — Continuous Integration / Continuous Deployment. Sounds complex. In reality, it's the simplest idea in the world:

**Automate what you would do manually.**

Without CI/CD, here's your workflow:
1. Modify code
2. Remember to run the tests (spoiler: you forget one time out of three)
3. Manually verify everything works
4. Manually deploy to the server
5. One day, a bug slips into production because you skipped step 2

With CI/CD:
1. Modify code and `git push`
2. **Everything else is automatic**: tests run, and if everything passes, deployment happens on its own

It's like having an invisible assistant who systematically checks your work. You can't forget to test anymore.

![CI/CD pipeline diagram — from code to deployment through automated testing](https://www.synopsys.com/glossary/what-is-cicd/_jcr_content/root/synopsyscontainer/column_1946395452_c/colRight/image_copy.coreimg.svg/1714621378774/cicd.svg)

> **Going further on CI/CD:**
> - [GitHub Actions — Official documentation](https://docs.github.com/en/actions) — the complete guide
> - [What is CI/CD?](https://www.redhat.com/en/topics/devops/what-is-ci-cd) — Red Hat
> - [GitHub Actions step-by-step tutorial](https://docs.github.com/en/actions/quickstart) — the official quickstart

### YAML — The configuration format

GitHub Actions pipelines are described in **YAML** files. If you've never seen YAML, don't panic — it's just a text format structured by indentation (like Python). Each line is either a key-value pair or a list item (prefixed with `-`).

```yaml
# Minimal YAML example
name: "Test Pipeline"
version: 1
steps:
  - name: "Install Python"
    action: "setup-python"
  - name: "Run tests"
    command: "pytest"
```

Pretty readable, right? No brackets, curly braces, or semicolons. Just consistent indentation (2 spaces by convention).

> **Watch out for the classic trap:** tabs. YAML only accepts spaces. If you mix tabs and spaces, the file will be invalid and the error message will be cryptic.

### The pipeline

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

# ---- WHEN to trigger the pipeline ----
on:
  push:
    branches: [main, develop]    # On every push to main or develop
  pull_request:
    branches: [main]             # On every pull request to main

# ---- THE JOBS ----
jobs:

  # ============================
  # JOB 1: Run tests
  # ============================
  test:
    runs-on: ubuntu-latest

    env:
      DATABASE_URL: sqlite:///./test.db  # Lightweight DB for CI tests

    steps:
      # Pull the repo code
      - name: Checkout code
        uses: actions/checkout@v4

      # Install Python
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      # Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # Run tests with coverage
      - name: Run tests
        run: |
          pytest --cov=app --cov-report=xml -v

      # Verify coverage meets minimum threshold
      - name: Check minimum coverage
        run: |
          coverage report --fail-under=80

  # ============================
  # JOB 2: Deploy (only if tests pass)
  # ============================
  deploy:
    needs: test   # ← waits for the "test" job to succeed
    runs-on: ubuntu-latest
    # Only on push to main (not on PRs)
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for the push

      - name: Deploy to Hugging Face Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add hf https://huggingface_user:$HF_TOKEN@huggingface.co/spaces/${{ secrets.HF_SPACE_NAME }}
          git push hf main --force
```

Let's break down the key points:

**`needs: test`** — The `deploy` job only starts if the `test` job succeeded. This is the heart of CI/CD: no green tests, no deployment.

**`${{ secrets.HF_TOKEN }}`** — These are **GitHub secrets**. Your Hugging Face token is encrypted and stored in the repo settings. It never appears in logs, even in case of errors. To set them up:

1. GitHub → your repo → **Settings** → **Secrets and variables** → **Actions**
2. **New repository secret**
3. Add `HF_TOKEN` and `HF_SPACE_NAME`

> **Going further:**
> - [GitHub Actions — Encrypted secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets) — docs on secret management
> - [Understanding GitHub Actions environments](https://docs.github.com/en/actions/deployment/targeting-different-environments) — dev, staging, prod

### The three environments

The pipeline implicitly manages three environments:

| Environment | Database | When |
|-------------|----------|------|
| **dev** (local) | Local PostgreSQL | On your machine |
| **test** (CI) | Temporary SQLite | In GitHub Actions |
| **prod** (cloud) | PostgreSQL/SQLite | On Hugging Face Spaces |

The `DATABASE_URL` variable changes depending on the context. That's why we use a `.env` file — each environment has its own.

```bash
git add .
git commit -m "ci: CI/CD pipeline with GitHub Actions (tests + auto-deploy)"
```

---

## Sunday late afternoon — Deploying to Hugging Face Spaces

### What is Hugging Face Spaces?

[Hugging Face Spaces](https://huggingface.co/spaces) is a free platform that hosts ML applications. You push your code via Git, and the application is automatically built and deployed. It's like GitHub Pages, but for ML applications with a backend.

The steps:
1. Create an account at [huggingface.co](https://huggingface.co)
2. Create a new Space (click "New Space")
3. Choose a name (e.g., `wine-quality-api`) and select **Docker** as the SDK
4. Your Space is ready to receive code

> **Going further:**
> - [Hugging Face Spaces — Documentation](https://huggingface.co/docs/hub/spaces) — the complete guide
> - [Getting started with Docker](https://docs.docker.com/get-started/) — if Docker is new to you

### The Dockerfile

Docker is a tool that **packages your application with everything it needs** to run: Python, dependencies, code, the model. The result is an "image" that runs identically everywhere — on your machine, on a server, on Hugging Face.

The `Dockerfile` is the recipe for building that image:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies FIRST
# (to leverage Docker cache — if requirements.txt hasn't changed,
# this step is instant on rebuild)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Port 7860 is used by Hugging Face Spaces
EXPOSE 7860

# Launch the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Deploy

```bash
# Add the Hugging Face remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/wine-quality-api

# Push the code
git push hf main
```

Within a few minutes, your application is live. The URL will be something like:
`https://YOUR_USERNAME-wine-quality-api.hf.space/docs`

And Swagger UI will be publicly accessible. Anyone in the world can now test your model.

```bash
git add Dockerfile
git commit -m "feat: Dockerfile for Hugging Face Spaces deployment"
git tag -a v1.0.0 -m "v1.0.0 — First public deployment"
git push origin main --tags
```

---

## Sunday evening — Documentation

### The README: 5 minutes of work, months of time saved

I'll admit it — I almost didn't write a README. It was 7 PM, I was tired, the API was working, I wanted to close my laptop. But I remembered all the GitHub repos I've opened and immediately closed because there was zero documentation.

A good README answers four questions in under 2 minutes of reading:

```markdown
# Wine Quality Prediction API

Predicts red wine quality based on physicochemical properties.
Random Forest model, FastAPI API, deployed on Hugging Face Spaces.

## Live Demo
https://YOUR_USERNAME-wine-quality-api.hf.space/docs

## Local Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 16+

### Steps
```shell
git clone https://github.com/your-user/wine-quality-api.git
cd wine-quality-api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Edit with your credentials
python -m scripts.init_db
uvicorn app.main:app --reload
```

## Usage

### Via Swagger UI (browser)
Open http://127.0.0.1:8000/docs

### Via curl (terminal)
```shell
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

## Tests
```shell
pytest --cov=app -v
```

## Architecture
- `app/` — Source code (API, model, DB)
- `tests/` — Automated tests
- `scripts/` — Initialization scripts
- `.github/workflows/` — CI/CD pipeline

## Tech Stack
FastAPI · Pydantic · PostgreSQL · SQLAlchemy · Pytest ·
GitHub Actions · Docker · Hugging Face Spaces
```

> **Going further:**
> - [Make a README](https://www.makeareadme.com/) — guide to writing a good README
> - [Awesome README](https://github.com/matiassingers/awesome-readme) — inspiring examples

---

## What this weekend taught me — Summary

Sunday evening, 10 PM. I had an ML API in production, publicly accessible, with automated tests and continuous deployment. Starting from nothing.

Here's the recap of everything we built:

| Building Block | Technology | Why it's there |
|----------------|------------|----------------|
| Versioning | Git + GitHub | History, branches, collaboration |
| REST API | FastAPI + Pydantic | Expose the model, validate inputs |
| Database | PostgreSQL + SQLAlchemy | Track and audit predictions |
| Tests | Pytest + pytest-cov | Safety net against regressions |
| CI/CD | GitHub Actions (YAML) | Automate testing and deployment |
| Hosting | Hugging Face Spaces + Docker | Make it publicly accessible |
| Documentation | Swagger UI + README | Make it understandable by all |

**Every building block is essential.** Remove one:
- Without Git → no traceability, no way to roll back
- Without Pydantic → invalid data silently breaks the model
- Without the DB → no history, impossible to diagnose problems
- Without tests → you discover bugs in production
- Without CI/CD → you forget to test, a bug slips through, chaos ensues
- Without documentation → nobody understands your work, not even you in 3 months

### Technologies to remember

| Tool | What it does | Link |
|------|-------------|------|
| **Git** | Code versioning | [git-scm.com](https://git-scm.com/) |
| **GitHub** | Code hosting + CI/CD | [github.com](https://github.com/) |
| **FastAPI** | Python API framework | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| **Pydantic** | Data validation | [docs.pydantic.dev](https://docs.pydantic.dev/) |
| **PostgreSQL** | Relational database | [postgresql.org](https://www.postgresql.org/) |
| **SQLAlchemy** | Python ↔ SQL ORM | [sqlalchemy.org](https://www.sqlalchemy.org/) |
| **Pytest** | Python testing framework | [docs.pytest.org](https://docs.pytest.org/) |
| **GitHub Actions** | CI/CD pipeline | [docs.github.com/actions](https://docs.github.com/en/actions) |
| **Docker** | Containerization | [docker.com](https://www.docker.com/) |
| **Hugging Face Spaces** | ML app hosting | [huggingface.co/spaces](https://huggingface.co/spaces) |

---

## Going further

This tutorial covers the fundamentals. If you want to dig deeper, here are my suggestions:

- **API Authentication** — add JWT or OAuth2 tokens to secure access ([FastAPI Security guide](https://fastapi.tiangolo.com/tutorial/security/))
- **Monitoring** — track real-time performance with Prometheus + Grafana
- **Data drift detection** — alert when input data distribution shifts ([Evidently AI](https://www.evidentlyai.com/))
- **Model versioning** — manage multiple model versions with [MLflow](https://mlflow.org/)
- **Docker Compose** — orchestrate the API + PostgreSQL in a single `docker-compose.yml`

---

## Conclusion

This weekend convinced me of one thing: **data science without production deployment is R&D that serves no one**. A model sitting in a notebook, no matter how good it is, has no value if it's not accessible.

Going from notebook to production is a conceptual leap as much as a technical one. You move from "it works on my machine" to "it works for everyone, all the time, reliably, tested, and traceable".

It's not just about code. It's an engineering mindset. Thinking about maintainability. Robustness. Documentation. Automation. The person who'll pick up your code in 6 months (who will probably be you, and who will have forgotten everything).

You now have all the building blocks. Next weekend, build yours.

---

*If this article helped you, a clap on Medium is always appreciated. And if you have questions or feedback, the comments are open — I respond to everyone.*
