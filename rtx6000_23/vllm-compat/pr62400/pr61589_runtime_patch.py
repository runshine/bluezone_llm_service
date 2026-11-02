from pathlib import Path


def patch_runtime(root: Path) -> None:
    kv_path = root / "v1/core/kv_cache_utils.py"
    kv = kv_path.read_text()
    if "# bluezone pr61589 lookahead" not in kv:
        kv = kv.replace(
            "    caching_hash_fn: Callable[[Any], bytes],\n"
            ") -> Callable[[Request], list[BlockHash]]:",
            "    caching_hash_fn: Callable[[Any], bytes],\n"
            "    num_hash_lookahead_tokens: int = 1,\n"
            ") -> Callable[[Request], list[BlockHash]]:",
            1,
        )
        kv = kv.replace(
            "        num_tokens = request.num_tokens\n\n"
            "        if start_token_idx + hash_block_size > num_tokens:",
            "        # bluezone pr61589 lookahead\n"
            "        num_tokens = request.num_tokens - num_hash_lookahead_tokens\n\n"
            "        if start_token_idx + hash_block_size > num_tokens:",
            1,
        )
        old = (
            "            # MM and LoRA requests need extra keys for block-hash computation.\n"
            "            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(\n"
            "                request, start_token_idx, end_token_idx, curr_mm_idx\n"
            "            )\n"
        )
        new = (
            "            # Include the proof token in the hash key, matching #61589.\n"
            "            hash_end_token_idx = end_token_idx + num_hash_lookahead_tokens\n"
            "            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(\n"
            "                request, start_token_idx, hash_end_token_idx, curr_mm_idx\n"
            "            )\n"
            "            extra_keys = (\n"
            "                extra_keys,\n"
            "                *request.all_token_ids[end_token_idx:hash_end_token_idx],\n"
            "            )\n"
        )
        if old not in kv:
            raise RuntimeError("cannot patch request block hasher")
        kv_path.write_text(kv.replace(old, new, 1))

    engine_path = root / "v1/engine/core.py"
    engine = engine_path.read_text()
    if "# bluezone pr61589 lookahead" not in engine:
        old = (
            "            self.request_block_hasher = get_request_block_hasher(\n"
            "                hash_block_size, caching_hash_fn\n"
            "            )\n"
        )
        new = (
            "            # bluezone pr61589 lookahead\n"
            "            self.request_block_hasher = get_request_block_hasher(\n"
            "                hash_block_size, caching_hash_fn, 1\n"
            "            )\n"
        )
        if old not in engine:
            raise RuntimeError("cannot patch engine hasher call")
        engine_path.write_text(engine.replace(old, new, 1))


if __name__ == "__main__":
    patch_runtime(Path("/opt/venv/lib/python3.12/site-packages/vllm"))
