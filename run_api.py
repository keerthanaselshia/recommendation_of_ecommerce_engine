"""
run_api.py
Entry-point for the Recommendation Engine API.

Usage:
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 8080 --reload

The server trains all models on startup (~30 seconds), then serves requests.
"""

import sys
import argparse
from pathlib import Path

# ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="E-Commerce Recommendation Engine API Server"
    )
    parser.add_argument("--host",    default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port",    default=8000, type=int, help="Bind port (default: 8000)")
    parser.add_argument("--reload",  action="store_true",  help="Enable hot-reload (dev mode)")
    parser.add_argument("--workers", default=1, type=int,  help="Number of worker processes")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║    E-Commerce Recommendation Engine API                  ║
╠══════════════════════════════════════════════════════════╣
║  Host    : {args.host:<45} ║
║  Port    : {args.port:<45} ║
║  Reload  : {str(args.reload):<45} ║
║  Docs    : http://{args.host}:{args.port}/docs{' '*(27-len(str(args.port)))} ║
╚══════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "api.app:app",
        host        = args.host,
        port        = args.port,
        reload      = args.reload,
        workers     = args.workers if not args.reload else 1,
        log_level   = "info",
    )
