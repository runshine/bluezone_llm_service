import argparse
import multiprocessing
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size-local", type=int, default=-1)
    parser.add_argument("--dp-rank-start", type=int, default=0)
    parser.add_argument("--dp-address", type=str, required=True)
    parser.add_argument("--dp-rpc-port", type=str, default="12345")
    parser.add_argument("--vllm-start-port", type=int, default=9000)
    return parser.parse_args()


args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = args.dp_size if args.dp_size_local == -1 else args.dp_size_local


def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        "./run_dp_template.sh",
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        args.dp_address,
        args.dp_rpc_port,
        str(tp_size),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    if not os.path.exists("./run_dp_template.sh"):
        print("Template file ./run_dp_template.sh does not exist.")
        sys.exit(1)

    processes = []
    for i in range(dp_size_local):
        dp_rank = args.dp_rank_start + i
        vllm_engine_port = args.vllm_start_port + i
        visible_devices = ",".join(
            str(device) for device in range(i * tp_size, (i + 1) * tp_size)
        )
        process = multiprocessing.Process(
            target=run_command,
            args=(visible_devices, dp_rank, vllm_engine_port),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()