"""
main.py — E-Commerce Product Recommendation Engine
====================================================
Full pipeline:
  1. Generate / load simulated dataset
  2. Train all recommendation models
  3. Demo — similar products
  4. Demo — personalised recs for 3 user profiles
  5. Full evaluation across models
  6. Generate HTML dashboard report
"""

import os
import sys
import warnings
import json
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError) ─────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.data_generator    import load_dataset
from database.seed          import initialise_database, load_dataframes
from database.connection    import DB_PATH
from models.content_based   import ContentBasedRecommender
from models.collaborative   import UserBasedCF, ItemBasedCF, SVDRecommender
from models.hybrid          import HybridRecommender
from utils.cold_start       import ColdStartRecommender
from evaluation.metrics     import (
    temporal_train_test_split, compare_models
)

# ── helpers ───────────────────────────────────────────────────────────────────

def banner(text: str, char: str = "═") -> None:
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def fmt_product(pid: str, score: float, products: pd.DataFrame, rank: int) -> str:
    row = products[products["product_id"] == pid]
    if row.empty:
        return f"  {rank:2d}. {pid}  (score={score:.4f})"
    r = row.iloc[0]
    return (
        f"  {rank:2d}. [{pid}] {r['title']}\n"
        f"      Category: {r['category']} | Brand: {r['brand']} | "
        f"Price: ${r['price']:.2f}\n"
        f"      Tags: {r['tags']}  | Score: {score:.4f}"
    )


def print_recs(
    recs: list[tuple[str, float]],
    products: pd.DataFrame,
    label: str,
) -> None:
    print(f"\n  ── {label} ──")
    if not recs:
        print("  (no recommendations returned)")
        return
    for i, (pid, score) in enumerate(recs, 1):
        print(fmt_product(pid, score, products, i))


# ── CLI ARGS ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="E-Commerce Product Recommendation Engine"
    )
    parser.add_argument(
        "--source",
        choices=["db", "generate"],
        default="db",
        help="Data source: 'db' loads from SQLite database (auto-initialises & seeds if empty), 'generate' generates synthetic dataset in-memory (default: db)",
    )
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="Force re-seeding the SQLite database with fresh data",
    )
    return parser.parse_args()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(args=None):
    if args is None:
        args = parse_args()

    if args.source == "db":
        banner("STEP 1 — DATA LOADING FROM DATABASE", "═")
        print(f"  Database file: {DB_PATH}")
        initialise_database(force_reseed=args.reseed)
        products, users, interactions = load_dataframes()
    else:
        banner("STEP 1 — IN-MEMORY DATA GENERATION", "═")
        products, users, interactions = load_dataset(
            n_products=500, n_users=200, n_interactions=10_000
        )

    print("\n── Dataset Overview ──")
    print(f"  Products    : {len(products):,}")
    print(f"  Users       : {len(users):,}")
    print(f"  Interactions: {len(interactions):,}")
    print(f"  Categories  : {products['category'].nunique()}")
    print("\n  Category distribution:")
    for cat, cnt in products["category"].value_counts().items():
        print(f"    {cat:<25} {cnt} products")

    # ── train / test split ────────────────────────────────────────────────────
    banner("STEP 2 — TRAIN / TEST SPLIT", "═")
    train, test = temporal_train_test_split(interactions, test_ratio=0.20)

    # ── train all models ──────────────────────────────────────────────────────
    banner("STEP 3 — TRAINING MODELS", "═")

    print("\n[1/6] Content-Based Recommender")
    cb_model = ContentBasedRecommender(max_features=5_000)
    cb_model.fit(products)

    print("\n[2/6] User-Based CF")
    user_cf = UserBasedCF(k_neighbors=30)
    user_cf.fit(train)

    print("\n[3/6] Item-Based CF")
    item_cf = ItemBasedCF(k_neighbors=30)
    item_cf.fit(train)

    print("\n[4/6] SVD Recommender")
    svd_model = SVDRecommender(n_factors=50)
    svd_model.fit(train)

    print("\n[5/6] Hybrid Recommender (CB + SVD, α=0.4)")
    hybrid = HybridRecommender(alpha=0.4, n_factors=50)
    hybrid.fit(products, train)

    print("\n[6/6] Cold-Start Popularity Recommender")
    cold_start = ColdStartRecommender(min_interactions=3, recency_halflife=90)
    cold_start.fit(products, train)

    # ── demo: similar products ────────────────────────────────────────────────
    banner("STEP 4 — DEMO: SIMILAR PRODUCTS", "═")

    sample_products = products.sample(3, random_state=1)
    for _, row in sample_products.iterrows():
        pid = row["product_id"]
        print(f"\n  Query product: [{pid}] {row['title']}")
        print(f"  Category: {row['category']} | Brand: {row['brand']}")
        print(f"  Tags: {row['tags']}")

        print("\n  ── Content-Based Similar Products ──")
        cb_sims = cb_model.get_similar_products(pid, n=5)
        for i, (spid, score) in enumerate(cb_sims, 1):
            print(fmt_product(spid, score, products, i))

        print("\n  ── Item-CF Similar Products ──")
        cf_sims = item_cf.get_similar_items(pid, n=5)
        for i, (spid, score) in enumerate(cf_sims, 1):
            print(fmt_product(spid, score, products, i))

        print("─" * 70)

    # ── demo: personalised recommendations ───────────────────────────────────
    banner("STEP 5 — DEMO: PERSONALISED RECOMMENDATIONS", "═")

    # 1. Heavy user (many interactions)
    interaction_counts = train["user_id"].value_counts()
    heavy_uid = interaction_counts.index[0]

    # 2. Light user — bottom-quartile interaction count (but at least 1)
    threshold = max(1, int(np.percentile(interaction_counts.values, 25)))
    light_candidates = interaction_counts[interaction_counts <= threshold].index
    if len(light_candidates) == 0:
        light_candidates = interaction_counts.index  # fallback: use least-active
    light_uid = light_candidates[-1]  # last = lowest count in the filtered set

    # 3. Brand-new user — synthesise a user ID guaranteed absent from train
    #    (all 200 simulated users may already appear in the 80% training split)
    all_uid_in_train = set(train["user_id"])
    new_uid_candidates = [u for u in users["user_id"] if u not in all_uid_in_train]
    if new_uid_candidates:
        new_uid = new_uid_candidates[0]
    else:
        # all users have train history → create a virtual cold-start ID
        new_uid = "U_COLDSTART_DEMO"

    demo_users = [
        ("Heavy user",  heavy_uid, False),
        ("Light user",  light_uid, False),
        ("New user",    new_uid,   True),
    ]

    for label, uid, is_new in demo_users:
        print(f"\n{'─'*70}")
        n_hist = (train["user_id"] == uid).sum()
        user_row = users[users["user_id"] == uid]
        prefs = user_row["preferred_categories"].values[0] if not user_row.empty else "N/A"
        print(f"  User: {uid} ({label}) | interactions in train: {n_hist}"
              f" | prefs: {prefs}")

        if is_new:
            pref_list = prefs.split(",") if prefs != "N/A" else None
            cold_recs = cold_start.get_recommendations(uid, n=5, preferred_categories=pref_list)
            print_recs(cold_recs, products, "Cold-Start (Popularity) Recs")
        else:
            cb_recs     = cb_model.get_recommendations(uid, train, n=5)
            user_recs   = user_cf.get_recommendations(uid, n=5)
            svd_recs    = svd_model.get_recommendations(uid, n=5)
            hybrid_recs = hybrid.get_recommendations(uid, n=5)

            print_recs(cb_recs,     products, "Content-Based Recs")
            print_recs(user_recs,   products, "User-CF Recs")
            print_recs(svd_recs,    products, "SVD Recs")
            print_recs(hybrid_recs, products, "Hybrid Recs")

    # ── evaluation ────────────────────────────────────────────────────────────
    banner("STEP 6 — EVALUATION (Precision / Recall / NDCG / HitRate @ K)", "═")

    models_to_eval = {
        "Popularity":   lambda uid, n: cold_start.get_popular_products(n),
        "Content-Based": lambda uid, n: cb_model.get_recommendations(uid, train, n),
        "User-CF":      lambda uid, n: user_cf.get_recommendations(uid, n),
        "Item-CF":      lambda uid, n: item_cf.get_recommendations(uid, n),
        "SVD":          lambda uid, n: svd_model.get_recommendations(uid, n),
        "Hybrid":       lambda uid, n: hybrid.get_recommendations(uid, n),
    }

    print()
    results_df = compare_models(
        models_to_eval, train, test,
        k_values=[5, 10, 20],
        max_users=150,
    )

    # pretty-print comparison table
    print("\n" + "─" * 75)
    print(f"  {'Model':<16} {'K':>3}  {'Precision':>10} {'Recall':>10} "
          f"{'NDCG':>10} {'HitRate':>10}")
    print("─" * 75)
    for _, row in results_df.iterrows():
        print(
            f"  {row['Model']:<16} {int(row['K']):>3}  "
            f"{row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['NDCG']:>10.4f} {row['HitRate']:>10.4f}"
        )
    print("─" * 75)

    # ── generate HTML report ──────────────────────────────────────────────────
    banner("STEP 7 — GENERATING HTML DASHBOARD", "═")
    report_path = ROOT / "dashboard" / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    _generate_html_report(
        report_path   = report_path,
        products      = products,
        users         = users,
        interactions  = interactions,
        train         = train,
        test          = test,
        results_df    = results_df,
        cb_model      = cb_model,
        svd_model     = svd_model,
        hybrid        = hybrid,
        cold_start    = cold_start,
        demo_users    = demo_users,
        demo_train    = train,
    )
    print(f"\n  ✓ Dashboard saved → {report_path}")
    print(f"    Open in browser: file:///{report_path.as_posix()}")

    banner("DONE ✓", "═")
    return results_df


# ── HTML report generator ─────────────────────────────────────────────────────

def _generate_html_report(
    report_path, products, users, interactions, train, test,
    results_df, cb_model, svd_model, hybrid, cold_start,
    demo_users, demo_train,
):
    """Render a rich, self-contained HTML dashboard."""

    # ── chart data ────────────────────────────────────────────────────────────
    all_models  = results_df["Model"].unique().tolist()
    k_values    = sorted(results_df["K"].unique().tolist())

    def metric_series(metric):
        """Returns {model: [val_at_k5, val_at_k10, val_at_k20]}"""
        d = {}
        for m in all_models:
            sub = results_df[results_df["Model"] == m].sort_values("K")
            d[m] = [round(v, 4) for v in sub[metric].tolist()]
        return d

    prec_data = metric_series("Precision")
    rec_data  = metric_series("Recall")
    ndcg_data = metric_series("NDCG")
    hr_data   = metric_series("HitRate")

    colors = [
        "#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6", "#a855f7"
    ]

    def js_datasets(data, metric_label):
        lines = []
        for i, (m, vals) in enumerate(data.items()):
            c = colors[i % len(colors)]
            lines.append(
                f"{{ label: '{m}', data: {vals}, "
                f"borderColor: '{c}', backgroundColor: '{c}22', "
                f"borderWidth: 2.5, pointRadius: 5, tension: 0.35 }}"
            )
        return ",\n".join(lines)

    # category distribution
    cat_counts = products["category"].value_counts()
    cat_labels = json.dumps(cat_counts.index.tolist())
    cat_values = json.dumps(cat_counts.values.tolist())

    # interaction type distribution
    itype_counts = interactions["interaction_type"].value_counts()
    itype_labels = json.dumps(itype_counts.index.tolist())
    itype_values = json.dumps(itype_counts.values.tolist())

    # evaluation table rows
    table_rows = ""
    prev_model = None
    for _, row in results_df.sort_values(["Model", "K"]).iterrows():
        model_cell = f'<td rowspan="3" class="model-cell">{row["Model"]}</td>' \
                     if row["Model"] != prev_model else ""
        prev_model = row["Model"]
        table_rows += (
            f'<tr>'
            f'{model_cell}'
            f'<td>{int(row["K"])}</td>'
            f'<td>{row["Precision"]:.4f}</td>'
            f'<td>{row["Recall"]:.4f}</td>'
            f'<td>{row["NDCG"]:.4f}</td>'
            f'<td>{row["HitRate"]:.4f}</td>'
            f'</tr>\n'
        )

    # best model per metric @ K=10
    k10 = results_df[results_df["K"] == 10]
    best = {
        "Precision": k10.loc[k10["Precision"].idxmax(), "Model"],
        "Recall":    k10.loc[k10["Recall"].idxmax(), "Model"],
        "NDCG":      k10.loc[k10["NDCG"].idxmax(), "Model"],
        "HitRate":   k10.loc[k10["HitRate"].idxmax(), "Model"],
    }

    # sample recommendations panels
    def make_rec_panel(uid, label, train_data, products_df):
        n_hist = (train_data["user_id"] == uid).sum()
        if n_hist == 0:
            recs = cold_start.get_recommendations(uid, n=6)
            model_label = "Cold-Start (Popularity)"
        else:
            recs = hybrid.get_recommendations(uid, n=6)
            model_label = "Hybrid (CB + SVD)"

        cards = ""
        for pid, score in recs:
            row = products_df[products_df["product_id"] == pid]
            if row.empty:
                continue
            r = row.iloc[0]
            cat_color = {
                "Electronics": "#6366f1", "Fashion": "#ec4899",
                "Books": "#f59e0b", "Home & Kitchen": "#10b981",
                "Sports & Outdoors": "#3b82f6", "Beauty": "#a855f7",
                "Toys & Games": "#ef4444", "Automotive": "#64748b",
                "Health": "#14b8a6", "Music & Instruments": "#f97316",
            }.get(r["category"], "#6366f1")
            cards += f"""
            <div class="rec-card">
              <div class="rec-cat-badge" style="background:{cat_color}22;color:{cat_color}">
                {r['category']}
              </div>
              <div class="rec-title">{r['title']}</div>
              <div class="rec-meta">
                <span class="rec-brand">{r['brand']}</span>
                <span class="rec-price">${r['price']:.2f}</span>
              </div>
              <div class="rec-score">Score: {score:.4f}</div>
            </div>"""
        return f"""
        <div class="user-panel">
          <div class="user-header">
            <span class="user-id">{uid}</span>
            <span class="user-label">{label}</span>
            <span class="user-hist">{n_hist} train interactions</span>
            <span class="model-badge">{model_label}</span>
          </div>
          <div class="rec-grid">{cards}</div>
        </div>"""

    rec_panels = ""
    for label, uid, _ in demo_users:
        rec_panels += make_rec_panel(uid, label, demo_train, products)

    # ── HTML template ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>E-Commerce Recommendation Engine Dashboard</title>
  <meta name="description" content="Interactive dashboard for the E-Commerce Product Recommendation Engine — comparing content-based, collaborative, SVD, and hybrid approaches." />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:        #0a0e1a;
      --surface:   #111827;
      --surface2:  #1f2937;
      --border:    #ffffff12;
      --text:      #f1f5f9;
      --muted:     #94a3b8;
      --accent:    #6366f1;
      --accent2:   #a855f7;
      --green:     #10b981;
      --amber:     #f59e0b;
      --red:       #ef4444;
      --blue:      #3b82f6;
    }}

    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.6;
    }}

    /* ── hero ── */
    .hero {{
      background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 100%);
      padding: 3.5rem 2rem 3rem;
      text-align: center;
      position: relative;
      overflow: hidden;
    }}
    .hero::before {{
      content: '';
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 50% 0%, #6366f140 0%, transparent 70%);
    }}
    .hero h1 {{
      font-size: clamp(1.8rem, 4vw, 3rem);
      font-weight: 800;
      background: linear-gradient(90deg, #e0e7ff, #c4b5fd, #a78bfa);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
      position: relative;
    }}
    .hero p {{
      color: #c4b5fd; font-size: 1.05rem; margin-top: .6rem; position: relative;
    }}
    .hero-badges {{
      display: flex; flex-wrap: wrap; justify-content: center;
      gap: .6rem; margin-top: 1.5rem; position: relative;
    }}
    .badge {{
      background: #ffffff18; border: 1px solid #ffffff25;
      border-radius: 2rem; padding: .3rem .9rem;
      font-size: .82rem; font-weight: 500; color: #e0e7ff;
      backdrop-filter: blur(8px);
    }}

    /* ── layout ── */
    .container {{ max-width: 1280px; margin: 0 auto; padding: 0 1.5rem; }}
    .section    {{ padding: 3rem 0; }}
    .section-title {{
      font-size: 1.4rem; font-weight: 700; margin-bottom: 1.5rem;
      display: flex; align-items: center; gap: .6rem;
    }}
    .section-title::before {{
      content: ''; display: inline-block; width: 4px; height: 1.4rem;
      background: linear-gradient(180deg, var(--accent), var(--accent2));
      border-radius: 2px;
    }}

    /* ── stat cards ── */
    .stat-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1.25rem;
    }}
    .stat-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 1rem; padding: 1.5rem;
      position: relative; overflow: hidden;
      transition: transform .2s, box-shadow .2s;
    }}
    .stat-card:hover {{ transform: translateY(-4px); box-shadow: 0 12px 40px #0008; }}
    .stat-card::after {{
      content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
      background: var(--accent-grad, linear-gradient(90deg, var(--accent), var(--accent2)));
    }}
    .stat-num {{ font-size: 2.2rem; font-weight: 800; line-height: 1; }}
    .stat-label {{ color: var(--muted); font-size: .88rem; margin-top: .4rem; }}
    .stat-icon {{ position: absolute; top: 1rem; right: 1rem; font-size: 1.8rem; opacity: .4; }}

    /* ── chart containers ── */
    .chart-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 1.5rem;
    }}
    .chart-card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 1rem; padding: 1.5rem;
    }}
    .chart-card h3 {{ font-size: 1rem; font-weight: 600; margin-bottom: 1.25rem; color: var(--muted); }}
    .chart-wrap {{ position: relative; height: 280px; }}

    /* ── best model badges ── */
    .best-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem; margin-bottom: 2rem;
    }}
    .best-card {{
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: .75rem; padding: 1rem 1.25rem;
    }}
    .best-metric {{ font-size: .8rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
    .best-model  {{ font-size: 1.1rem; font-weight: 700; color: var(--green); margin-top: .25rem; }}

    /* ── evaluation table ── */
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%; border-collapse: collapse;
      font-size: .9rem;
    }}
    thead th {{
      background: var(--surface2); padding: .85rem 1rem;
      text-align: left; font-weight: 600;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}
    tbody td {{
      padding: .75rem 1rem; border-bottom: 1px solid var(--border);
      vertical-align: middle;
    }}
    tbody tr:hover td {{ background: var(--surface2); }}
    .model-cell {{
      font-weight: 600; color: var(--accent);
      border-right: 1px solid var(--border);
    }}

    /* ── recommendation panels ── */
    .user-panel {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem;
    }}
    .user-header {{
      display: flex; flex-wrap: wrap; align-items: center; gap: .75rem;
      margin-bottom: 1.25rem;
    }}
    .user-id    {{ font-size: 1.05rem; font-weight: 700; color: var(--accent); }}
    .user-label {{ background: var(--accent)22; color: var(--accent); border-radius: .5rem;
                   padding: .2rem .7rem; font-size: .82rem; font-weight: 600; }}
    .user-hist  {{ color: var(--muted); font-size: .85rem; }}
    .model-badge {{ margin-left: auto; background: var(--surface2); border: 1px solid var(--border);
                    border-radius: .5rem; padding: .2rem .75rem; font-size: .8rem; color: var(--muted); }}

    .rec-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 1rem;
    }}
    .rec-card {{
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: .75rem; padding: 1rem;
      transition: transform .2s, box-shadow .2s;
    }}
    .rec-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px #0006; }}
    .rec-cat-badge {{
      display: inline-block; border-radius: .4rem;
      padding: .15rem .6rem; font-size: .72rem; font-weight: 600; margin-bottom: .5rem;
    }}
    .rec-title  {{ font-size: .9rem; font-weight: 600; margin-bottom: .5rem; line-height: 1.3; }}
    .rec-meta   {{ display: flex; justify-content: space-between; font-size: .8rem; }}
    .rec-brand  {{ color: var(--muted); }}
    .rec-price  {{ color: var(--green); font-weight: 600; }}
    .rec-score  {{ color: var(--muted); font-size: .75rem; margin-top: .4rem; }}

    /* ── footer ── */
    footer {{
      text-align: center; padding: 2rem; color: var(--muted);
      font-size: .85rem; border-top: 1px solid var(--border);
    }}

    /* ── scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: #ffffff20; border-radius: 3px; }}
  </style>
</head>
<body>

<!-- ═══ HERO ═══════════════════════════════════════════════════════════════ -->
<div class="hero">
  <h1>🛒 E-Commerce Recommendation Engine</h1>
  <p>Content-Based · Collaborative Filtering · SVD · Hybrid · Cold-Start</p>
  <div class="hero-badges">
    <span class="badge">TF-IDF + Cosine Similarity</span>
    <span class="badge">User-Based CF</span>
    <span class="badge">Item-Based CF</span>
    <span class="badge">SVD Matrix Factorization</span>
    <span class="badge">Hybrid Model</span>
    <span class="badge">Precision @ K · Recall @ K · NDCG</span>
  </div>
</div>

<div class="container">

<!-- ═══ DATASET OVERVIEW ═══════════════════════════════════════════════════ -->
<section class="section">
  <div class="section-title">Dataset Overview</div>
  <div class="stat-grid">
    <div class="stat-card" style="--accent-grad: linear-gradient(90deg,#6366f1,#a855f7)">
      <div class="stat-num" style="color:#6366f1">{len(products):,}</div>
      <div class="stat-label">Products in Catalog</div>
      <div class="stat-icon">📦</div>
    </div>
    <div class="stat-card" style="--accent-grad: linear-gradient(90deg,#3b82f6,#06b6d4)">
      <div class="stat-num" style="color:#3b82f6">{len(users):,}</div>
      <div class="stat-label">User Profiles</div>
      <div class="stat-icon">👤</div>
    </div>
    <div class="stat-card" style="--accent-grad: linear-gradient(90deg,#10b981,#14b8a6)">
      <div class="stat-num" style="color:#10b981">{len(interactions):,}</div>
      <div class="stat-label">Total Interactions</div>
      <div class="stat-icon">🔗</div>
    </div>
    <div class="stat-card" style="--accent-grad: linear-gradient(90deg,#f59e0b,#f97316)">
      <div class="stat-num" style="color:#f59e0b">{products['category'].nunique()}</div>
      <div class="stat-label">Product Categories</div>
      <div class="stat-icon">🏷️</div>
    </div>
    <div class="stat-card" style="--accent-grad: linear-gradient(90deg,#ef4444,#ec4899)">
      <div class="stat-num" style="color:#ef4444">{len(train):,}</div>
      <div class="stat-label">Train Interactions (80%)</div>
      <div class="stat-icon">🎓</div>
    </div>
    <div class="stat-card" style="--accent-grad: linear-gradient(90deg,#a855f7,#8b5cf6)">
      <div class="stat-num" style="color:#a855f7">{len(test):,}</div>
      <div class="stat-label">Test Interactions (20%)</div>
      <div class="stat-icon">🧪</div>
    </div>
  </div>
</section>

<!-- ═══ DATA DISTRIBUTION CHARTS ═══════════════════════════════════════════ -->
<section class="section">
  <div class="section-title">Data Distribution</div>
  <div class="chart-grid">

    <div class="chart-card">
      <h3>Products by Category</h3>
      <div class="chart-wrap">
        <canvas id="catChart"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <h3>Interaction Type Distribution</h3>
      <div class="chart-wrap">
        <canvas id="itypeChart"></canvas>
      </div>
    </div>

  </div>
</section>

<!-- ═══ BEST MODELS ══════════════════════════════════════════════════════════ -->
<section class="section">
  <div class="section-title">Best Model per Metric (K=10)</div>
  <div class="best-grid">
    <div class="best-card">
      <div class="best-metric">Precision@10</div>
      <div class="best-model">🏆 {best["Precision"]}</div>
    </div>
    <div class="best-card">
      <div class="best-metric">Recall@10</div>
      <div class="best-model">🏆 {best["Recall"]}</div>
    </div>
    <div class="best-card">
      <div class="best-metric">NDCG@10</div>
      <div class="best-model">🏆 {best["NDCG"]}</div>
    </div>
    <div class="best-card">
      <div class="best-metric">HitRate@10</div>
      <div class="best-model">🏆 {best["HitRate"]}</div>
    </div>
  </div>
</section>

<!-- ═══ PERFORMANCE CHARTS ═══════════════════════════════════════════════════ -->
<section class="section">
  <div class="section-title">Model Performance Comparison</div>
  <div class="chart-grid">

    <div class="chart-card">
      <h3>Precision @ K</h3>
      <div class="chart-wrap"><canvas id="precChart"></canvas></div>
    </div>

    <div class="chart-card">
      <h3>Recall @ K</h3>
      <div class="chart-wrap"><canvas id="recChart"></canvas></div>
    </div>

    <div class="chart-card">
      <h3>NDCG @ K</h3>
      <div class="chart-wrap"><canvas id="ndcgChart"></canvas></div>
    </div>

    <div class="chart-card">
      <h3>Hit Rate @ K</h3>
      <div class="chart-wrap"><canvas id="hrChart"></canvas></div>
    </div>

  </div>
</section>

<!-- ═══ FULL RESULTS TABLE ════════════════════════════════════════════════════ -->
<section class="section">
  <div class="section-title">Full Evaluation Results</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Model</th><th>K</th>
          <th>Precision@K</th><th>Recall@K</th>
          <th>NDCG@K</th><th>HitRate@K</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>
</section>

<!-- ═══ PERSONALISED RECOMMENDATIONS ════════════════════════════════════════ -->
<section class="section">
  <div class="section-title">Personalised Recommendations — Sample Users</div>
  {rec_panels}
</section>

</div><!-- /container -->

<footer>
  E-Commerce Recommendation Engine · Built with Python (scikit-learn, scipy, numpy, pandas) · Chart.js
</footer>

<script>
const K_LABELS = {json.dumps([str(k) for k in k_values])};

const CHART_DEFAULTS = {{
  type: 'line',
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8', boxWidth: 12, font: {{ size: 12 }} }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#ffffff08' }}, title: {{ display: true, text: 'K', color: '#94a3b8' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#ffffff08' }}, min: 0 }}
    }}
  }}
}};

function lineChart(id, datasets) {{
  new Chart(document.getElementById(id), {{
    ...CHART_DEFAULTS,
    data: {{ labels: K_LABELS, datasets }}
  }});
}}

// ── data distribution charts ──────────────────────────────────────────────
new Chart(document.getElementById('catChart'), {{
  type: 'bar',
  data: {{
    labels: {cat_labels},
    datasets: [{{ label: 'Products', data: {cat_values},
      backgroundColor: ['#6366f1','#a855f7','#3b82f6','#10b981','#f59e0b',
                        '#ef4444','#ec4899','#64748b','#14b8a6','#f97316'],
      borderRadius: 6, borderSkipped: false }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8', maxRotation: 30 }}, grid: {{ color: '#ffffff08' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#ffffff08' }} }}
    }}
  }}
}});

new Chart(document.getElementById('itypeChart'), {{
  type: 'doughnut',
  data: {{
    labels: {itype_labels},
    datasets: [{{ data: {itype_values},
      backgroundColor: ['#6366f1','#f59e0b','#10b981','#ef4444'],
      borderWidth: 0, hoverOffset: 8 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }}
  }}
}});

// ── performance line charts ───────────────────────────────────────────────
lineChart('precChart', [
  {js_datasets(prec_data, "Precision")}
]);
lineChart('recChart', [
  {js_datasets(rec_data, "Recall")}
]);
lineChart('ndcgChart', [
  {js_datasets(ndcg_data, "NDCG")}
]);
lineChart('hrChart', [
  {js_datasets(hr_data, "HitRate")}
]);
</script>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
