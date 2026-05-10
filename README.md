# DataFlow-AI
A Production-Grade Multi-Agent Framework for Autonomous ML Pipelines.
(Autonomous Data Science Platform)

A production-grade, modular ML platform that simulates a real-world data science organization. This system automates the end-to-end machine learning lifecycle—from data ingestion and cleaning to parallel model training and automated reporting—orchestrated by specialized **LangGraph** agents.

## 🚀 Key Features

* **Agentic Orchestration**: Managed by LangGraph 1.0.8 using a StateGraph pattern for robust multi-agent coordination.
* **Parallel Model Training**: Concurrent execution of Random Forest, Logistic Regression, and XGBoost using `ThreadPoolExecutor`.
* **Automated EDA**: Generates 5 distinct chart types, correlation matrices, and comprehensive dataset statistics.
* **Robust Data Pipeline**: Automated imputation, deduplication, IQR clipping, and feature engineering (scaling/encoding).
* **Professional Reporting**: Generates polished Markdown and HTML reports for every experiment.
* **Production Architecture**: FastAPI backend, PostgreSQL persistence, and Dockerized infrastructure.

## 🏗️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Orchestration** | LangGraph (StateGraph, Send API) |
| **Backend** | FastAPI + Uvicorn |
| **Machine Learning** | Scikit-learn, XGBoost, Pandas, NumPy |
| **Database** | PostgreSQL 16 + SQLAlchemy + Alembic |
| **Infrastructure** | Docker, Docker Compose |
| **Testing** | Pytest (350+ tests) |

## 📁 Project Structure

```text
├── app/
│   ├── agents/         # Phase-specific logic (Planner, EDA, Training, etc.) [cite: 5, 6, 7]
│   ├── api/            # FastAPI routers (Health, Datasets, Workflow, Reports) [cite: 7, 8]
│   ├── workflows/      # LangGraph StateGraph and Node definitions [cite: 12, 13]
│   ├── services/       # Business logic orchestration [cite: 11, 12]
│   └── models/         # SQLAlchemy ORM schemas [cite: 10, 11]
├── outputs/            # Trained .pkl models and generated PNG charts [cite: 17]
├── reports/            # Markdown and HTML experiment outputs [cite: 18]
└── tests/              # Extensive test suite for all pipeline phases [cite: 14, 15, 16]
```


## Quick Start

### With Docker (recommended)

```bash
cp .env.example .env          # review and adjust values
docker compose up --build     # starts PostgreSQL + API
```

API docs: http://localhost:8000/docs  
Health:   http://localhost:8000/health  
Readiness: http://localhost:8000/ready

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Point POSTGRES_HOST to localhost in .env, then:
python scripts/init_db.py
uvicorn app.main:app --reload --port 8000
```
