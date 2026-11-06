"""Fail closed before vLLM starts if the selected IPv4 RoCE-v2 GID is wrong."""
import argparse
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import socket
import struct


def validate(device, store_config, env=os.environ, sysfs=Path("/sys")):
    config = json.loads(Path(store_config).read_text())
    if config.get("protocol", "rdma") != "rdma":
        raise ValueError("Expected rdma Store transport")
    if config.get("device_name") != device:
        raise ValueError(f'Store HCA {config.get("device_name")!r} != P/D HCA {device!r}')
    index = int(env["MC_GID_INDEX"])
    if index < 0:
        raise ValueError("MC_GID_INDEX must be nonnegative")
    port = sysfs / "class/infiniband" / device / "ports/1"
    state = (port / "state").read_text().strip()
    if state.split(":")[0] != "4":
        raise ValueError(f"HCA is not ACTIVE: {device}: {state}")
    gid = ipaddress.IPv6Address((port / "gids" / str(index)).read_text().strip())
    kind = (port / "gid_attrs/types" / str(index)).read_text().strip()
    nic = (port / "gid_attrs/ndevs" / str(index)).read_text().strip()
    if kind != "RoCE v2" or gid.ipv4_mapped is None:
        raise ValueError(f"Expected IPv4-mapped RoCE v2: {device} index={index} gid={gid} type={kind}")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        addr = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", nic.encode()[:15]))
    ipv4 = ipaddress.IPv4Address(addr[20:24])
    if gid.ipv4_mapped != ipv4:
        raise ValueError(f"GID IPv4 {gid.ipv4_mapped} != {nic} IPv4 {ipv4}")
    verbs = [p for p in (sysfs / "class/infiniband_verbs").glob("uverbs*")
             if (p / "ibdev").read_text().strip() == device]
    if not any(os.access("/dev/infiniband/" + p.name, os.R_OK | os.W_OK) for p in verbs):
        raise ValueError(f"No accessible verbs device for {device}")
    print(f"ROCE_GID_OK device={device} port=1 index={index} gid={gid} ipv4={ipv4} netdev={nic}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True, choices=["mlx5_2", "mlx5_4"])
    p.add_argument("--store-config", default=os.environ.get("MOONCAKE_CONFIG_PATH", "/opt/workspace/mooncake_config.json"))
    args = p.parse_args()
    validate(args.device, args.store_config)
