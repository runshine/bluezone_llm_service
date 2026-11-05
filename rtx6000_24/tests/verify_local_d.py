#!/usr/bin/env python3
"""Check the deployable D tree in isolated, CPU-only copies of its image."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def inside():
    # Execute only the reviewed preparation portion, never the model launcher.
    startup = Path("/opt/workspace/run.sh").read_text()
    prefix, separator, _ = startup.partition("export TRITON_PTXAS_BLACKWELL_PATH=")
    if not separator or "launch_online_dp.py" in prefix or "vllm serve" in prefix:
        raise RuntimeError("Cannot isolate the startup preparation safely")
    subprocess.run(["bash", "-c", prefix], check=True)
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("Offline tests must not have GPU access")
    runtime = Path("/opt/venv/lib/python3.12/site-packages/vllm")
    for relative in ("v1/core/kv_cache_utils.py", "v1/engine/core.py"):
        assert "# bluezone pr61589 lookahead" in (runtime / relative).read_text()
    env = dict(os.environ, VLLM_MOONCAKE_LOOKAHEAD_HASHES="1")
    subprocess.run([sys.executable, "/opt/kv-tests/kv_cache_isolation_regression_test.py"],
                   check=True, env=env)


def inspect_service(root, name):
    directory = root / f"DeepSeek-V4.1-Flash-1017-{name}"
    rendered = subprocess.check_output(
        ["docker", "compose", "-f", str(directory / "docker-compose.yaml"),
         "config", "--format", "json"], text=True)
    service = json.loads(rendered)["services"][f"DeepSeek-V4.1-Flash-1017-{name}"]
    template = (directory / "run_dp_template.sh").read_text()
    match = re.search(r"--kv-transfer-config\s+'([^']+)'", template, re.S)
    if not match:
        raise RuntimeError(f"Missing KV configuration: {name}")
    config = json.loads(match.group(1))
    children = config["kv_connector_extra_config"]["connectors"]
    assert config["kv_role"] == "kv_consumer"
    assert [c["kv_connector"] for c in children] == ["MooncakeConnector", "MooncakeStoreConnector"]
    assert all(c["kv_role"] == "kv_consumer" for c in children)
    assert children[1]["kv_connector_extra_config"] == {"enable_mixed_group_store": True}
    assert "cache_prefix" not in config["kv_connector_extra_config"]
    assert all("cache_prefix" not in child.get("kv_connector_extra_config", {})
               for child in children)
    assert service["environment"]["VLLM_MOONCAKE_LOOKAHEAD_HASHES"] == "1"
    assert "--disable-hybrid-kv-cache-manager" not in template
    assert "--prefix-match-unit 8" in template and "--block-size 128" in template
    expected = {"D0": ("0,1,2,3", "7300", "9100", "mlx5_2"),
                "D1": ("4,5,6,7", "7301", "9101", "mlx5_4")}[name]
    gpu, port, bootstrap, device = expected
    assert f'CUDA_VISIBLE_DEVICES="{gpu}"' in template
    assert f"--vllm-start-port {port}" in (directory / "run.sh").read_text()
    assert f"VLLM_MOONCAKE_BOOTSTRAP_PORT={bootstrap}" in template
    assert config["kv_connector_extra_config"]["device_name"] == device
    assert service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == gpu.split(",")
    for script in ("run.sh", "run_dp_template.sh"):
        subprocess.run(["bash", "-n", str(directory / script)], check=True)
    for volume in service["volumes"]:
        source = Path(volume["source"])
        assert source.exists(), f"Missing mount source: {source}"
        if volume["target"] != "/data/models":
            assert source.resolve().is_relative_to(root.resolve()), f"External dependency: {source}"
    return service


def run_offline(root, name, service, output):
    cmd = ["docker", "run", "--rm", "--network", "none", "--cpus", "2",
           "--memory", "8g", "-e", "NVIDIA_VISIBLE_DEVICES=void",
           "-e", "CUDA_VISIBLE_DEVICES=", "-e", "PYTHONHASHSEED=0",
           "--entrypoint", "/opt/venv/bin/python"]
    # Use exactly the deployment's source->target mappings, but read-only and
    # with no privileges, devices, host network, or GPU reservations.
    for volume in service["volumes"]:
        cmd += ["--mount", f"type=bind,src={volume['source']},dst={volume['target']},readonly"]
    cmd += ["--mount", f"type=bind,src={root / 'tests'},dst=/opt/kv-tests,readonly",
            service["image"], "/opt/kv-tests/verify_local_d.py", "--inside"]
    print(f"{name}: running isolated image regression", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    log = result.stdout + result.stderr
    (output / f"{name}_offline.log").write_text(log)
    print(f"{name}: offline exit={result.returncode}", flush=True)
    return {"exit_code": result.returncode, "log": str(output / f"{name}_offline.log"),
            "image": service["image"], "network": "none", "gpu_access": False,
            "mounts_checked": len(service["volumes"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--output", default="tests/results/local_d_20261104")
    args = parser.parse_args()
    if args.inside:
        inside()
        return
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=False)
    services = {name: inspect_service(root, name) for name in ("D0", "D1")}
    manifest = {}
    for name, service in services.items():
        files = [root / f"DeepSeek-V4.1-Flash-1017-{name}" / script
                 for script in ("docker-compose.yaml", "run.sh", "run_dp_template.sh")]
        files += [Path(v["source"]) for v in service["volumes"] if Path(v["source"]).is_file()]
        for path in files:
            relative = str(path.resolve().relative_to(root))
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "SHA256SUMS").write_text("".join(
        f"{digest}  {path}\n" for path, digest in sorted(manifest.items())))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(run_offline, root, name, svc, output)
                   for name, svc in services.items()}
        results = {name: future.result() for name, future in futures.items()}
    report = {"checked_at": datetime.datetime.now().astimezone().isoformat(),
              "scope": "local files and disposable CPU-only containers; no live service or remote changes",
              "results": results, "file_count": len(manifest),
              "passed": all(r["exit_code"] == 0 for r in results.values())}
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
