#!/usr/bin/env python3
"""Read-only lookup utility for RequestForwarding audit JSONL indexes."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="trace", help="trace root directory")
    parser.add_argument("--date", help="YYYY-MM-DD; defaults to all retained dates")
    parser.add_argument("--port", choices=("8077", "8080"))
    parser.add_argument("--trace-id")
    parser.add_argument("--response-id")
    parser.add_argument("--session-id")
    parser.add_argument("--thread-id")
    parser.add_argument("--final-state", choices=("accepted", "rejected", "failed", "disconnected"))
    parser.add_argument("--error-code")
    args = parser.parse_args()
    filters = {key.replace("-", "_"): value for key, value in vars(args).items()
               if key in {"trace_id", "response_id", "session_id", "thread_id", "final_state", "error_code"}
               and value is not None}
    root = Path(args.root) / "audit"
    dates = [args.date] if args.date else sorted(path.name for path in root.glob("????-??-??") if path.is_dir())
    matched = set()
    for day in dates:
        ports = [args.port] if args.port else ("8077", "8080")
        for port in ports:
            index = root / day / port / "events.jsonl"
            if not index.is_file():
                continue
            for line in index.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "terminal" or not all(event.get(k) == v for k, v in filters.items()):
                    continue
                key = (event.get("trace_id"), event.get("terminal_at"))
                if key not in matched:
                    matched.add(key)
                    print(json.dumps(event, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
