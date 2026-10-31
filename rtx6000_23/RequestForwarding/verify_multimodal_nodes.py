"""Read-only admission check. Run before deployment and before adding PD nodes."""
import argparse
import asyncio
import base64
import hashlib
import json
import struct
import zlib
from datetime import datetime, timezone

import httpx


def sample_request(model):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\xff\xff" * 64) * 64))
           + chunk(b"IEND", b""))
    return {
        "model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe the image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
        ]}],
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"},
        "add_special_tokens": False,
    }


async def inspect_node(client, role, root, model, payload):
    root = root.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    result = {"role": role, "node": root, "ok": False}
    try:
        models, tokenize = await asyncio.gather(
            client.get(root + "/v1/models"), client.post(root + "/tokenize", json=payload),
        )
        models.raise_for_status()
        tokenize.raise_for_status()
        ids = sorted(item["id"] for item in models.json()["data"])
        body = tokenize.json()
        tokens, count = body["tokens"], body["count"]
        if model not in ids or not isinstance(tokens, list) or isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count != len(tokens):
            raise ValueError("invalid model/count")
        result.update(
            ok=True, models=ids, count=count, max_model_len=body["max_model_len"],
            visual_tokens=tokens.count(129264),
            tokens_sha256=hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
        )
    except Exception as exc:
        # Never print response bodies, which can contain the image URL.
        result["error"] = type(exc).__name__
    return result


async def verify(prefillers, decoders, model, timeout=30):
    payload = sample_request(model)
    async with httpx.AsyncClient(trust_env=False, timeout=timeout, limits=httpx.Limits(max_connections=8)) as client:
        results = await asyncio.gather(
            *(inspect_node(client, "prefill", root, model, payload) for root in prefillers),
            *(inspect_node(client, "decode", root, model, payload) for root in decoders),
        )
    # Same-role renderers must agree. Context capacity may differ between P/D.
    consistent = all(result["ok"] and result.get("visual_tokens", 0) > 0 for result in results)
    for role in ("prefill", "decode"):
        signatures = {
            (tuple(r["models"]), r["count"], r["max_model_len"], r["tokens_sha256"])
            for r in results if r["ok"] and r["role"] == role
        }
        consistent = consistent and len(signatures) == 1
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "consistent": consistent, "nodes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefillers", nargs="+", required=True, help="HTTP roots of all existing and candidate P nodes")
    parser.add_argument("--decoders", nargs="+", required=True, help="HTTP roots of all existing and candidate D nodes")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    report = asyncio.run(verify(args.prefillers, args.decoders, args.model, args.timeout))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["consistent"] else 1)


if __name__ == "__main__":
    main()
