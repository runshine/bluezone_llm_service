#!/usr/bin/env python3
"""CPU-only checks, run with the deployment's vLLM interpreter.

Exercise the repaired D-node Store path with deterministic CPU fixtures.
Partial-group hits must fail closed; complete groups can be restored safely.
No live service, GPU, or external Store is mutated.
"""
import hashlib
import ast
import inspect
import json
import textwrap
import unittest
import threading
import ctypes
from dataclasses import replace
from types import SimpleNamespace

import torch
from unittest.mock import patch
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    MooncakeStoreWorker,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    RankLocalStoreLayout, KeyMetadata, RequestTracker, ReqMeta, LoadSpec,
    ChunkedTokenDatabase,
    PoolKey,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash


class CacheGroupContractTest(unittest.TestCase):
    def test_transfer_hotpaths_do_not_format_hashes_for_diagnostics(self):
        for method in (KVCacheStoreSendingThread._handle_request,
                       KVCacheStoreRecvingThread._load_request,
                       MooncakeStoreWorker.lookup):
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "logger"):
                    continue
                self.assertFalse(any(
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Attribute)
                    and arg.func.attr == "hex"
                    for arg in ast.walk(node)), method.__qualname__)
                self.assertFalse(any(
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and "lookup diagnosis:" in arg.value
                    for arg in node.args), method.__qualname__)

    def test_default_namespace_shares_keys_without_collapsing_group_or_rank(self):
        producer = KeyMetadata(model_name="shared_model", tp_rank=0,
                               pcp_rank=0, dcp_rank=0, pp_rank=0, group_id=1)
        consumer = replace(producer)
        self.assertEqual(producer.cache_prefix, "")
        producer_key = PoolKey.build_prefix(producer)
        self.assertEqual(producer_key, PoolKey.build_prefix(consumer))
        self.assertTrue(producer_key.startswith("shared_model@"))
        self.assertNotEqual(producer_key, PoolKey.build_prefix(replace(consumer, group_id=2)))
        self.assertNotEqual(producer_key, PoolKey.build_prefix(replace(consumer, tp_rank=1)))

    def test_proof_token_and_salt_are_part_of_d_node_hash_contract(self):
        hash_fn = lambda value: hashlib.sha256(repr(value).encode()).digest()
        # Normally initialized by EngineCore; this fixture intentionally does
        # not start an engine or touch a GPU.
        init_none_hash(hash_fn)
        for block_size in (8, 32):
            def request(name, proof, salt="shared"):
                return Request(request_id=name,
                               prompt_token_ids=list(range(block_size)) + proof,
                               sampling_params=SamplingParams(max_tokens=16),
                               pooling_params=None, cache_salt=salt)
            hasher = get_request_block_hasher(block_size, hash_fn, 1)
            self.assertEqual(hasher(request("without_proof", [])), [])
            a = hasher(request("a", [100]))
            b = hasher(request("b", [101]))
            self.assertEqual(len(a), 1)
            self.assertNotEqual(a, b)
            self.assertEqual(a, hasher(request("different_request_id", [100])))
            self.assertNotEqual(a, hasher(request("salted", [100], "other")))
            legacy = get_request_block_hasher(block_size, hash_fn, 0)
            self.assertEqual(legacy(request("old_a", [100])),
                             legacy(request("old_b", [101])))

    def test_namespace_requires_child_extra_config(self):
        marker = "snapshot_fix_20261104_load_snapshot_v1"
        children = [{"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"},
                    {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}]
        config = SimpleNamespace(kv_transfer_config=KVTransferConfig(
            kv_connector="MultiConnector", kv_role="kv_producer",
            kv_connector_extra_config={"cache_prefix": marker, "connectors": children}))
        target = ("vllm.distributed.kv_transfer.kv_connector.v1.multi_connector."
                  "KVConnectorFactory.get_connector_class")
        with patch(target, return_value=object):
            configs = MultiConnector._get_connector_classes_and_configs(config)
            self.assertNotIn("cache_prefix", configs[1][1].kv_transfer_config.kv_connector_extra_config)
            children[1]["kv_connector_extra_config"] = {"cache_prefix": marker}
            configs = MultiConnector._get_connector_classes_and_configs(config)
            self.assertEqual(configs[1][1].kv_transfer_config.kv_connector_extra_config["cache_prefix"], marker)

    def test_removing_override_uses_group_gcd_not_scheduler_block(self):
        groups = [
            KVCacheGroupSpec(layer_names=["full"], kv_cache_spec=FullAttentionSpec(
                block_size=128, num_kv_heads=1, head_size=64, dtype=torch.float16)),
            KVCacheGroupSpec(layer_names=["swa"], kv_cache_spec=SlidingWindowMLASpec(
                block_size=32, num_kv_heads=1, head_size=64, dtype=torch.float16,
                sliding_window=128)),
        ]
        config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128, prefix_match_unit=None,
                                         enable_prefix_caching=True),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            kv_transfer_config=object(),
        )
        self.assertEqual(resolve_kv_cache_block_sizes(
            SimpleNamespace(kv_cache_groups=groups), config), (128, 32))
        config.cache_config.prefix_match_unit = 8
        self.assertEqual(resolve_kv_cache_block_sizes(
            SimpleNamespace(kv_cache_groups=groups), config), (128, 8))

    def test_old_xargs_skip_control_is_not_a_real_cache_bypass(self):
        request = ChatCompletionRequest(
            model="deepseek-v4.1-flash", messages=[{"role": "user", "content": "test"}],
            vllm_xargs={"skip_reading_prefix_cache": True})
        params = request.to_sampling_params(64, {})
        self.assertFalse(params.skip_reading_prefix_cache)
        self.assertTrue(params.extra_args["skip_reading_prefix_cache"])

    def test_worker_rejects_missing_swa_at_both_hash_granularities(self):
        for hash_size in (8, 32):
            # SWA has 32-token blocks; global attention has 128-token blocks.
            # Both groups explicitly participate in prefix caching.
            specs = [
                FullAttentionSpec(block_size=128, num_kv_heads=1,
                                  head_size=64, dtype=torch.float16),
                SlidingWindowMLASpec(block_size=32, num_kv_heads=1,
                                     head_size=64, dtype=torch.float16,
                                     sliding_window=128),
            ]
            groups = [KVCacheGroupSpec(layer_names=[f"layer{i}"], kv_cache_spec=s)
                      for i, s in enumerate(specs)]
            coord = MooncakeStoreCoordinator(groups, 128, hash_size)
            hashes = [hashlib.sha256(str(i).encode()).digest()
                      for i in range(512 // hash_size)]
            full_hashes = coord.block_hashes_for_spec(hashes, specs[0])
            exists = {(0, bytes(h)) for h in full_hashes}
            pool = ExternalCachedBlockPool(hash_size, exists)
            _, reconciled, _ = coord.find_longest_cache_hit(hashes, 512, pool)
            masks, partial, _ = coord.find_longest_cache_hit(
                hashes, 512, pool, active_group_ids={0})
            self.assertEqual(reconciled, 0)
            self.assertEqual(partial, 512)
            self.assertEqual(masks[1], [])
            # Exercise the real worker lookup with a deterministic in-memory
            # Store: no SWA key exists, but the current worker never asks for it.
            worker = object.__new__(MooncakeStoreWorker)
            worker._capacity_only = False
            worker.coord = coord
            worker.hash_block_size = hash_size
            worker.block_size = 128
            worker._kv_cache_groups = groups
            worker.token_dbs = [SimpleNamespace(block_size=s.block_size) for s in specs]
            worker._lookup_key_prefixes = [("full",), ("swa",)]
            worker._record_kv_connector_operation = lambda *a, **kw: None
            worker.store = SimpleNamespace(batch_is_exist=lambda keys: [
                int(key.startswith("full")) for key in keys])
            result = worker.lookup(513, hashes)
            self.assertEqual(result.hit_length, 0)
            worker.enable_mixed_group_store = True
            self.assertEqual(worker.lookup(513, hashes).hit_length, 0)
            worker.store.batch_is_exist = lambda keys: [1] * len(keys)
            self.assertEqual(worker.lookup(513, hashes).hit_length, 512)
            # A missing rank shard must invalidate the participating group,
            # even if another rank has every SWA page.
            worker._lookup_key_prefixes[1] = ("swa", "missing_rank")
            worker.store.batch_is_exist = lambda keys: [
                int(not key.startswith("missing_rank")) for key in keys]
            self.assertEqual(worker.lookup(513, hashes).hit_length, 0)
            print(json.dumps({"hash_size": hash_size,
                              "all_groups_reconciled_hit": reconciled,
                              "worker_reported_hit": result.hit_length,
                              "unsafe_partial_group_hit": partial,
                              "missing_prefix_cacheable_group": 1}), flush=True)
            # An explicitly non-prefix-cacheable scratch group is different:
            # it must not disable Store reuse for the supported group.
            groups[1] = replace(groups[1], kv_cache_spec=replace(
                specs[1], allow_prefix_caching=False))
            worker.coord = MooncakeStoreCoordinator(groups, 128, hash_size)
            self.assertEqual(worker.lookup(513, hashes).hit_length, 512)

    def test_group_layout_uses_only_own_layers_and_physical_blocks(self):
        metadata = KeyMetadata(model_name="test", tp_rank=0, pcp_rank=0,
                               dcp_rank=0, pp_rank=0)
        layouts = [RankLocalStoreLayout(metadata, bs, 8) for bs in (128, 32)]
        full = torch.zeros((8, 128, 2), dtype=torch.uint8)
        swa = torch.zeros((8, 32, 2), dtype=torch.uint8)
        worker = object.__new__(MooncakeStoreWorker)
        worker.num_blocks = 8
        worker._kv_cache_groups = [SimpleNamespace(layer_names=["full"]),
                                   SimpleNamespace(layer_names=["swa"])]
        worker.token_dbs = [SimpleNamespace(store_layout=l, block_size=bs)
                            for l, bs in zip(layouts, (128, 32))]
        worker._register_group_layouts({"full": full, "swa": swa})
        for layout, tensor, bs in zip(layouts, (full, swa), (128, 32)):
            addrs, sizes, bids = layout.prepare_values([(bs, bs * 2)], [1, 3], [0])
            self.assertEqual(addrs, [[tensor.data_ptr() + 3 * bs * 2]])
            self.assertEqual(sizes, [[bs * 2]])
            self.assertEqual(bids, [3])
            for bad_bid in (0, -1, 8):
                with self.assertRaises(ValueError):
                    layout.prepare_values([(0, bs)], [bad_bid], [0])
        with self.assertRaises(ValueError):
            worker._register_group_layouts({"full": full})

    def test_interleaved_layers_do_not_copy_other_groups_or_padding(self):
        metadata = KeyMetadata(model_name="interleaved", tp_rank=0,
                               pcp_rank=0, dcp_rank=0, pp_rank=0)
        # Two layers share a block-interleaved allocation. Each layer view's
        # stride includes the other layer, but its payload must not include it.
        backing = torch.zeros((8, 2, 32, 2), dtype=torch.uint8)
        layouts = [RankLocalStoreLayout(metadata, 32, 8) for _ in range(2)]
        for index, layout in enumerate(layouts):
            view = backing[:, index]
            layout.register_kv_caches([view], 8)
            addrs, sizes, _ = layout.prepare_values([(0, 32)], [3], [0])
            self.assertEqual(addrs, [[backing.data_ptr() + 3 * 128 + index * 64]])
            self.assertEqual(sizes, [[64]])
            self.assertEqual(layout.prepare_value_for_block(3), (addrs[0], sizes[0]))

    def test_metadata_owns_block_hash_and_load_snapshots(self):
        hashes = [b"a"] * 16
        tracker = RequestTracker(req_id="snapshot", token_len=128,
                                 allocated_block_ids=([1, 2],))
        load = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=128, can_load=True)
        meta = ReqMeta.from_request_tracker(tracker, 128, load_spec=load,
                                           block_hashes=hashes)
        tracker.allocated_block_ids[0][0] = 99
        hashes[0] = b"changed"
        load.kvpool_cached_tokens = 999
        self.assertEqual(meta.block_ids, ([1, 2],))
        self.assertEqual(meta.block_hashes[0], b"a")
        self.assertEqual(meta.load_spec.kvpool_cached_tokens, 128)
        self.assertFalse(meta.can_save)

    def test_load_mapping_failure_invalidates_entire_request_and_completes(self):
        receiver = KVCacheStoreRecvingThread(
            SimpleNamespace(), SimpleNamespace(), [], 128, 0, threading.Event())
        meta = ReqMeta(req_id="bad_load", token_len_chunk=128,
                       block_ids=([0, 2], [3]), block_hashes=[])
        receiver.request_queue.put(meta)
        with patch.object(receiver, "_load_request", side_effect=ValueError("bad mapping")):
            receiver._handle_request(meta)
        self.assertEqual(receiver.get_and_clear_block_ids_with_load_errors(), {2, 3})
        self.assertEqual(receiver.get_and_clear_finished_requests(), {"bad_load"})
        self.assertEqual(receiver.request_queue.unfinished_tasks, 0)

    def test_mixed_group_save_lookup_load_roundtrip(self):
        # Exercise real save/load descriptor generation against host memory;
        # no CUDA, live Mooncake data, or service state is involved.
        for hash_size in (8, 32):
            payloads = {}
            def put(keys, addresses, sizes, config):
                for key, regions, lengths in zip(keys, addresses, sizes, strict=True):
                    payloads[key] = [ctypes.string_at(p, n) for p, n in zip(regions, lengths)]
                return [0] * len(keys)
            def get(keys, addresses, sizes):
                for key, regions, lengths in zip(keys, addresses, sizes, strict=True):
                    self.assertEqual([len(b) for b in payloads[key]], lengths)
                    for p, n, data in zip(regions, lengths, payloads[key], strict=True):
                        ctypes.memmove(p, data, n)
                return [0] * len(keys)
            store = SimpleNamespace(batch_is_exist=lambda keys: [int(k in payloads) for k in keys],
                                    batch_put_from_multi_buffers=put, batch_get_into_multi_buffers=get)
            specs = [FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=64, dtype=torch.float16),
                     SlidingWindowMLASpec(block_size=32, num_kv_heads=1, head_size=64,
                                          dtype=torch.float16, sliding_window=128)]
            groups = [KVCacheGroupSpec(layer_names=[f"g{i}"], kv_cache_spec=s)
                      for i, s in enumerate(specs)]
            coord = MooncakeStoreCoordinator(groups, 128, hash_size)
            def databases():
                return [ChunkedTokenDatabase(KeyMetadata(model_name="roundtrip", tp_rank=0,
                        pcp_rank=0, dcp_rank=0, pp_rank=0, group_id=i), s.block_size, hash_size)
                        for i, s in enumerate(specs)]
            source_dbs, target_dbs = databases(), databases()
            sources = [torch.arange(64 * s.block_size * 2, dtype=torch.int64).to(torch.uint8)
                       .reshape(64, s.block_size, 2) for s in specs]
            sources[1].add_(73)
            targets = [torch.zeros_like(t) for t in sources]
            for db, tensor in zip(source_dbs, sources):
                db.store_layout.register_kv_caches([tensor], 64)
            for db, tensor in zip(target_dbs, targets):
                db.store_layout.register_kv_caches([tensor], 64)
            hashes = [hashlib.sha256(str(i).encode()).digest() for i in range(512 // hash_size)]
            source_blocks = tuple(list(range(1, 512 // s.block_size + 1)) for s in specs)
            target_blocks = tuple(list(range(33, 33 + 512 // s.block_size)) for s in specs)
            sender = KVCacheStoreSendingThread(store, coord, source_dbs, 128, 0, [1, 1],
                                               "kv_both", threading.Event())
            save = ReqMeta(req_id="roundtrip", token_len_chunk=512, block_ids=source_blocks,
                           block_hashes=hashes, can_save=True, store_job_id=1)
            sender.add_request(save)
            sender._handle_request(save)
            self.assertGreater(len(payloads), 4)
            lookup = object.__new__(MooncakeStoreWorker)
            lookup._capacity_only = False
            lookup.enable_mixed_group_store = True
            lookup.coord, lookup.hash_block_size, lookup.block_size = coord, hash_size, 128
            lookup._kv_cache_groups, lookup.token_dbs, lookup.store = groups, source_dbs, store
            lookup._lookup_key_prefixes = [db.store_layout.lookup_key_prefixes([(0, 0, 0, 0)]) for db in source_dbs]
            lookup._record_kv_connector_operation = lambda *a, **kw: None
            self.assertEqual(lookup.lookup(513, hashes).hit_length, 512)
            receiver = KVCacheStoreRecvingThread(store, coord, target_dbs, 128, 0, threading.Event())
            load = ReqMeta(req_id="roundtrip_load", token_len_chunk=512, block_ids=target_blocks,
                           block_hashes=hashes, load_spec=LoadSpec(0, 512, True, token_len=512))
            receiver.add_request(load)
            receiver._handle_request(load)
            self.assertEqual(receiver.get_and_clear_block_ids_with_load_errors(), set())
            masks = coord.load_mask(hashes, 512)
            for group_id, (source, target) in enumerate(zip(sources, targets)):
                for index, (sb, tb) in enumerate(zip(source_blocks[group_id], target_blocks[group_id])):
                    if masks[group_id][index]:
                        self.assertTrue(torch.equal(source[sb], target[tb]))


if __name__ == "__main__":
    unittest.main()
