"""
manage_db.py
Command-line tool for database management.

Commands
--------
  python manage_db.py init       — create tables + seed if empty
  python manage_db.py reseed     — drop all data and re-generate from scratch
  python manage_db.py stats      — print row counts
  python manage_db.py inspect    — show sample rows from each table
  python manage_db.py drop       — drop all tables (destructive!)

Usage
-----
  python manage_db.py stats
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from database.connection import SessionLocal, engine, DB_PATH
from database.models     import Product, User, Interaction, Base
from database.crud       import get_stats, is_seeded
from database.seed       import create_tables, seed_database


# ── helpers ───────────────────────────────────────────────────────────────────

def cmd_init(args):
    print(f"Database: {DB_PATH}")
    create_tables()
    with SessionLocal() as db:
        if is_seeded(db) and not args.force:
            stats = get_stats(db)
            print(f"Already seeded — {stats['n_products']} products, "
                  f"{stats['n_users']} users, {stats['n_interactions']} interactions")
            print("Use --force to re-seed.")
        else:
            seed_database(db, force=True)


def cmd_reseed(_args):
    print("Re-seeding database (drops existing data)…")
    Base.metadata.drop_all(bind=engine)
    create_tables()
    with SessionLocal() as db:
        seed_database(db, force=True)


def cmd_stats(_args):
    print(f"Database: {DB_PATH}")
    if not DB_PATH.exists():
        print("Database file does not exist. Run: python manage_db.py init")
        return
    with SessionLocal() as db:
        stats = get_stats(db)
        print(f"  Products    : {stats['n_products']:,}")
        print(f"  Users       : {stats['n_users']:,}")
        print(f"  Interactions: {stats['n_interactions']:,}")


def cmd_inspect(_args):
    if not DB_PATH.exists():
        print("Database file does not exist. Run: python manage_db.py init")
        return
    with SessionLocal() as db:
        print("\n── Sample Products (first 5) ──")
        for p in db.query(Product).limit(5).all():
            print(f"  {p.product_id}  {p.title:<50}  ${p.price:.2f}  [{p.category}]")

        print("\n── Sample Users (first 5) ──")
        for u in db.query(User).limit(5).all():
            print(f"  {u.user_id}  age={u.age}  gender={u.gender}  prefs={u.preferred_categories}")

        print("\n── Sample Interactions (first 5) ──")
        for ix in db.query(Interaction).limit(5).all():
            print(f"  {ix.user_id} → {ix.product_id}  "
                  f"[{ix.interaction_type}]  strength={ix.strength}  "
                  f"ts={ix.timestamp}")


def cmd_drop(_args):
    confirm = input("This will DROP all tables. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        Base.metadata.drop_all(bind=engine)
        print("All tables dropped.")
    else:
        print("Aborted.")


# ── CLI ───────────────────────────────────────────────────────────────────────

COMMANDS = {
    "init":    cmd_init,
    "reseed":  cmd_reseed,
    "stats":   cmd_stats,
    "inspect": cmd_inspect,
    "drop":    cmd_drop,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recommendation Engine — Database Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k}" for k in COMMANDS),
    )
    parser.add_argument("command", choices=COMMANDS,
                        help="Management command to run")
    parser.add_argument("--force", action="store_true",
                        help="Force re-seed even if DB is populated (init only)")
    args = parser.parse_args()
    COMMANDS[args.command](args)
