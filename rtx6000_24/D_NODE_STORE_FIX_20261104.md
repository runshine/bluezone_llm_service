# D0/D1 Store 修复：本地交付与手动上线

## 当前状态

本地 D0/D1 已同步完整 Store 修复及相关调度、哈希依赖。
前一版于 2026-11-04 21:26（UTC+08:00）完成离线检查：D0、D1 各 10 项回归全部通过。
随后按要求移除 P0/P1/D0/D1 的全部部署配置 `cache_prefix`，改用默认空命名空间；
新增默认命名空间共享键、保留 group/rank 区分的回归，最新结果见下文 noprefix 目录。
随后完成本地性能清理，最终交付结果以 `local_d_20261104_perfclean_final` 为准：
移除逐块 save/load 映射及 hash 格式化、每次 lookup 的诊断日志、初始化 group 诊断，
以及 P 节点临时 A/B 开关；保留异步元数据快照、proof-token 哈希、内存边界检查、
CUDA 完成事件、异常处理和 Store 操作指标。未改变 Router 路由策略，也未删除历史记录。
新增热路径诊断开销回归后，共 12 项测试。
测试使用各自 Compose 声明的镜像和全部 16 个挂载映射，运行在无网络、
无 GPU、非特权的临时容器中；模型没有加载，也没有连接线上 Store。

**本次没有同步远端文件、没有重启 P/D 或 Router，也没有执行新 D 版本的线上 GPU 验证。**
此前线上 P 侧的 1264 请求验证不能代替本次 D 侧上线后的验证。

## 必须一起同步的目录

在远端保留相同的 `/data/bluezone_llm_service/rtx6000_24` 路径，并一起更新：

```text
rtx6000_24/
  DeepSeek-V4.1-Flash-1017-D0/    # 代码、run.sh、模板、Compose 等整套目录
  DeepSeek-V4.1-Flash-1017-D1/
  vllm-compat/                  # 新增共享依赖，不能遗漏
    multi_connector.py
    deepseek_v41.py
    pr61589_runtime_patch.py
  tests/                       # 可选：离线复验和结果、文件校验和
```

可以同步整个 `rtx6000_24`，但无需改动其中其他模型目录。
不能只同步 `worker.py`，也不能只同步两个 D 目录而漏掉共享 `vllm-compat`。
Compose 已不再依赖远端 `rtx6000_23/vllm-compat` 中的文件。
模型目录 `/data/models` 和原有 RDMA/NIC 条件仍需在远端存在。

## 修复内容

- Store lookup、save、partial-tail save、load 使用一致的全组语义；
  不再以 block_size 小于调度块为理由跳过必要 SWA 组。
- 每组仅映射自身层的张量；区分物理块步长与有效载荷长度，避免拷贝其他层或 padding。
- block table、hash 列表和 load spec 独立快照；加载异常使整条请求失效并完成队列收尾。
- 同步核心 scheduler 和 single-type cache manager 中与多组、proof hash 配套的修复。
- D 启动时应用与 P 一致的 lookahead/proof-token 哈希补丁；
  `VLLM_MOONCAKE_LOOKAHEAD_HASHES=1`，`PYTHONHASHSEED=0`，prefix-match-unit 仍为 8。
- P0/P1/D0/D1 的 Store 子连接器均不配置 `cache_prefix`，使用默认空命名空间共享兼容 KV。
  保留 `enable_mixed_group_store=true` 及模型、group、rank、hash 等必要的原生键字段。
  删除这些原生键字段会造成错误混用，不属于共享缓存的正确方式。
- 移除 D 的 `--disable-hybrid-kv-cache-manager`，与已修复 P 的混合组策略对齐；
  因此上线后需要重新检查实际 group layout、内存预算和 P/D 传输。
- 修正 D1 的 `TP_SOCKET_IFNAME` 拼写。

保留原来的 consumer 角色、dspark 配置和节点差异：

| 节点 | GPU | API 端口 | bootstrap 端口 | device_name |
| --- | --- | --- | --- | --- |
| D0 | 0,1,2,3 | 7300 | 9100 | mlx5_2 |
| D1 | 4,5,6,7 | 7301 | 9101 | mlx5_4 |

## 上线前复验

从 `rtx6000_24` 目录执行，校验 31 个关键运行文件：

```bash
sha256sum -c tests/results/local_d_20261104_perfclean_final/SHA256SUMS
```

需要重新运行离线回归时，从仓库根目录执行；输出路径必须是尚不存在的新目录：

```bash
python3 rtx6000_24/tests/verify_local_d.py --output tests/results/manual_recheck_01
```

结果和日志：

```text
tests/results/local_d_20261104_perfclean_final/summary.json
tests/results/local_d_20261104_perfclean_final/D0_offline.log
tests/results/local_d_20261104_perfclean_final/D1_offline.log
```

日志中的注入式 `ValueError: bad mapping` 是预期的负向测试，最终以
`Ran 12 tests ... OK` 和 `summary.json` 的 `passed=true` 为准。
早期两个失败结果目录被保留：一个是离线只读 workspace 下的子挂载冲突，
现已改到 `/opt/kv-compat/pr61589_runtime_patch.py`；另一个是新测试未模拟
EngineCore 的 NONE_HASH 初始化，已修正测试夹具。不是线上模型异常。

## 手动上线顺序

**空命名空间迁移注意事项：** 相同的非空前缀原本也能在 P/D 之间共享，
本次是按要求改为默认空前缀，而不是关闭模型/group/rank 隔离。
历史未修复版本曾使用空前缀写入不同载荷格式；不能假设这些旧键已经过期。
上线前需在停止相关流量、确认影响范围后清理旧空前缀数据，或切换至干净的 Store。
该操作影响共享服务，本文不自动执行清库，也不建议在有请求时直接重启共享 Master。
P0/P1 也需要同步最新启动模板并重启；只更新 D 不会与仍运行旧前缀配置的 P 共享这些键。

1. 先备份远端原 D0/D1 和共享配置；确认现有容器使用的 Compose project 名。
   如以前使用了 `-p`，后续必须继续使用同一个 project，避免启动第二套 GPU 实例。
2. 同步上面的目录；检查 Compose 的全部 host mount source 都实际存在。
3. 在 **172.31.29.24** 使用原 project 重建 D 容器。
   本轮改变了挂载和环境变量，**仅 `docker restart` 不会应用这些 Compose 变更**。

以下命令适用于原来使用默认 project 的情况；自定义 project 时加上对应 `-p`：

```bash
cd /data/bluezone_llm_service/rtx6000_24
docker compose -f DeepSeek-V4.1-Flash-1017-D0/docker-compose.yaml config --quiet
docker compose -f DeepSeek-V4.1-Flash-1017-D1/docker-compose.yaml config --quiet
docker compose -f DeepSeek-V4.1-Flash-1017-D0/docker-compose.yaml up -d --force-recreate
docker compose -f DeepSeek-V4.1-Flash-1017-D1/docker-compose.yaml up -d --force-recreate
```

如果 Docker 提示旧 GPU 容器没有退出事件，先检查原容器及 GPU 子进程是否真正退出，
不要创建重复实例，也不要自动重置 GPU 或宿主机。

4. 等待所有 P/D 健康探测通过：

```bash
curl -f http://172.31.29.23:7100/health
curl -f http://172.31.29.23:7101/health
curl -f http://172.31.29.24:7300/health
curl -f http://172.31.29.24:7301/health
```

5. 然后在 **172.31.29.23** 重启现有 Router，并核对 bootstrap 和 worker 注册：

```bash
docker restart vllm-router-VllmRouter-1
curl -f http://172.31.29.23:8077/health
curl -f http://172.31.29.23:8077/workers
```

确认四个 worker 全部健康，Router bootstrap 中的 P engine ID 与当前 P 节点一致。
不要在 D 尚未就绪时先启动压测。无需为本次变更清空 Master 或重启未修改的 API 层。

## 上线后验收边界

- 先做 P0/D0、P1/D1，再做 P0/D1、P1/D0 的混合长度回归。
- 要证明 **D 自己读取 Store**，不能只测正常 Router 的 P/D 请求。
  D 的第一个子连接器仍是 Mooncake P/D；正常请求可能只走 P 恢复 Store 后传给 D 的路径。
- 单独验证 D Store 时，先通过 P 用长前缀和新共享 cache_salt 预热 Store，
  再以相同 model、模板和盐值直接请求 D0/D1 的 `/v1/chat/completions`，
  不携带远端 P/D `kv_transfer_params`。D 是 consumer，不能指望 D 自己写入预热数据。
- 分别检查 D0/D1 的成功 `load_get` 操作、加载字节、external hit/source token 增量和回答；
  HTTP 200、lookup exists 或 P/D successful transfer 都不能单独证明 D Store 恢复正确。
- thinking/chat 模板必须各自预热；此前 `/tokenize` 已证明它们可能从开头就不同。
- 检查加载失败、越界、OOM 和 GPU 内存预算；首次上线建议先小批量，再提高并发。

如需回退，恢复备份的整套代码与 Compose 配置并重建 D；等 P/D 全部就绪后再次重启 Router。
不要让旧载荷格式与当前默认空命名空间中的新载荷混用。
