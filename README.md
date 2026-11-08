# 🛒 E-Commerce Product Recommendation Engine

A complete, modular recommendation system built in pure Python using `scikit-learn`, `scipy`, `numpy`, and `pandas`.

## Features

| Model | Method | Cold-Start Safe |
|---|---|---|
| **Content-Based** | TF-IDF (ngram 1-2) + Cosine Similarity on product features | ✅ |
| **User-CF** | User-user cosine similarity, K-nearest neighbours | ❌ |
| **Item-CF** | Item-item cosine similarity | ❌ |
| **SVD** | Truncated matrix factorization (rank-50, mean-centred) | ❌ |
| **Hybrid** | Weighted blend: α×CB + (1-α)×SVD | Partial |
| **Cold-Start** | Recency-weighted global & per-category popularity | ✅ |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Manage or inspect the database (optional)
python manage_db.py stats

# 3. Run the recommendation pipeline (loads from SQLite DB & generates dashboard report)
python main.py

# 4. Or launch the live FastAPI server & dashboard
python run_api.py
# Docs: http://127.0.0.1:8000/docs
# Dashboard: http://127.0.0.1:8000/dashboard/index.html
```

## Database Architecture & Management

The project uses an embedded **SQLite database** (`db/recommender.db`) managed through SQLAlchemy ORM.

### Tables & Schema
- **`products`**: `product_id` (PK), `title`, `category`, `brand`, `price`, `tags`, `description`, `image_url`, `features`. Composite index on `(category, brand)`.
- **`users`**: `user_id` (PK), `age`, `gender`, `preferred_categories`.
- **`interactions`**: `id` (PK), `user_id` (FK), `product_id` (FK), `interaction_type` (`view`, `wishlist`, `cart`, `purchase`), `strength` (1–4), `rating`, `timestamp`. Composite index on `(user_id, product_id)`.

### Database CLI (`manage_db.py`)
```bash
python manage_db.py stats      # View current product, user, and interaction counts
python manage_db.py inspect    # Preview sample rows from each table
python manage_db.py init       # Create tables and seed if database is empty
python manage_db.py reseed     # Wipe and re-populate with fresh synthetic data
python manage_db.py drop       # Drop all tables
```

### Pipeline Options (`main.py`)
```bash
python main.py                 # Default: loads directly from SQLite database
python main.py --source db     # Explicitly load from SQLite database
python main.py --reseed        # Re-seed the database before training
python main.py --source generate # In-memory generation without touching the database
```

## Project Structure

```
sample/
├── main.py                        # Entry-point — full pipeline (supports DB and simulated data)
├── manage_db.py                   # Database management CLI tool (init, stats, inspect, reseed)
├── run_api.py                     # FastAPI server runner
├── recommendation_engine.ipynb    # Jupyter notebook version
├── requirements.txt
├── db/
│   └── recommender.db             # Embedded SQLite database
├── database/
│   ├── connection.py              # SQLAlchemy engine, session maker & connection pool
│   ├── models.py                  # Product, User, and Interaction ORM models
│   ├── crud.py                    # Database CRUD functions & dataframe extractors
│   └── seed.py                    # Schema creation and automatic database seeding
├── api/
│   ├── app.py                     # FastAPI app with lifespan model training
│   ├── schemas.py                 # Pydantic request/response schemas
│   └── routers/                   # Products, Users, Interactions, Recs, Compare, Eval
├── data/
│   └── data_generator.py          # Simulates products, users, interactions
├── models/
│   ├── content_based.py           # TF-IDF + cosine similarity
│   ├── collaborative.py           # User-CF, Item-CF, SVD
│   └── hybrid.py                  # Weighted blend CB + SVD
├── utils/
│   └── cold_start.py              # Popularity-based fallback
├── evaluation/
│   └── metrics.py                 # Precision@K, Recall@K, NDCG@K, HitRate@K
└── dashboard/
    ├── report.html                # Auto-generated offline HTML dashboard
    ├── index.html                 # Interactive live web dashboard
    └── compare.html               # Multi-product comparison dashboard
```

## Evaluation Metrics

- **Precision@K** — Of top-K recommended items, what fraction are relevant?
- **Recall@K** — Of all relevant test items, what fraction appear in top-K?
- **NDCG@K** — Normalised Discounted Cumulative Gain (rewards top-ranked relevant items)
- **HitRate@K** — Fraction of users with ≥1 relevant item in their top-K list

Evaluated at K = 5, 10, 20 using a **temporal 80/20 train/test split**.

## Simulated Dataset

- **500 products** across 10 categories (Electronics, Fashion, Books, …)
- **200 user profiles** with preferred categories
- **~10,000 interactions** (view, wishlist, cart, purchase) biased by user preferences
