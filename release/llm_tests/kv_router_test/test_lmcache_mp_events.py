"""Real replica-local MP reuse; requires the native-event LMCache dependency.

Set RAY_LMCACHE_MP_TWO_NODE=1 for two nodes advertising one GPU each.
"""

import asyncio
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlsplit
import uuid

import pytest
import requests
import msgspec
import zmq
from vllm.distributed.kv_events import KVEventBatch

import ray
from ray import serve
from ray._common.test_utils import async_wait_for_condition
from ray.serve.config import RequestRouterConfig
from ray.serve.llm import LLMConfig
from ray.serve.llm.request_router import KVAwareRouter

from lmcache_mp_utils import MPTestServer, MPWorkerExtension, observe_retrieves
from test_kv_events import (
    BLOCK_SIZE,
    MODEL_ID,
    MODEL_SOURCE,
    post_chat,
    tokenize_prompt,
)
from utils import build_kv_app, discover_replica_endpoints, patch_ingress

APP_NAME = "lmcache_mp_gpu_test"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TARGET = [
    {
        "role": "user",
        "content": "MP routing target. " + "orchid amber birch indigo " * 140,
    }
]
# 5s RPC timeout + up to 5s subscription backoff + 1s heartbeat + 0.1s idle poll.
INVALIDATION_TIMEOUT = 15


def two_node_gpu_owners(nodes):
    gpu_nodes = [
        node for node in nodes if node["Alive"] and node["Resources"].get("GPU", 0) > 0
    ]
    assert len(gpu_nodes) == 2, gpu_nodes
    assert all(node["Resources"]["GPU"] == 1 for node in gpu_nodes), gpu_nodes
    owners = {node["NodeID"]: node["NodeManagerAddress"] for node in gpu_nodes}
    assert len(owners) == len(set(owners.values())) == 2, owners
    return owners


def assert_mp_owners(endpoints, reports, node_ips):
    assert len(reports) == 2, reports
    for report in reports:
        assert endpoints[report["replica_id"]][0] == report["node_ip"], report
    if node_ips:
        assert {r["node_id"]: r["node_ip"] for r in reports} == node_ips, reports
        # Loopback addresses and OS PIDs are unique only within their owner node.
        assert len({(r["node_id"], r["server_url"]) for r in reports}) == 2, reports
        assert len({(r["node_id"], r["server_pid"]) for r in reports}) == 2, reports
    else:
        assert len({host for host, _ in endpoints.values()}) == 1, endpoints
        assert len({r["node_id"] for r in reports}) == 1, reports
        assert len({r["server_url"] for r in reports}) == 2, reports
        assert len({r["server_pid"] for r in reports}) == 2, reports


def mp_report(endpoint):
    host, port = endpoint
    response = requests.get(f"http://{host}:{port}/test/mp", timeout=45)
    response.raise_for_status()
    return response.json()


def mp_status(endpoint):
    """Read MP state without invoking the model engine's utility queue."""
    host, port = endpoint
    response = requests.get(f"http://{host}:{port}/test/mp/status", timeout=15)
    response.raise_for_status()
    return response.json()


def tier_hashes(snapshot, worker, tier):
    return {
        block["block_hash"]
        for partition in snapshot.values()
        for event in partition["events"]
        if event["worker_id"] == worker and event["storage_tier"] == tier
        for block in event["event"]["data"].get("stored", {}).get("blocks", [])
    }


@pytest.fixture(scope="module")
def mp_runtime():
    if ray.is_initialized():
        serve.shutdown()
        ray.shutdown()
    ray.init(
        address="auto",
        runtime_env={
            "py_modules": [str(Path(__file__).with_name("lmcache_mp_utils.py"))]
        },
    )
    try:
        two_node = os.environ.get("RAY_LMCACHE_MP_TWO_NODE", "0")
        assert two_node in ("0", "1"), two_node
        yield two_node_gpu_owners(ray.nodes()) if two_node == "1" else {}
    finally:
        serve.shutdown()
        ray.shutdown()


@pytest.fixture(params=[16, 256], ids=["equal-16", "unequal-256"])
def mp_app(request, mp_runtime):
    serve.shutdown()
    chunk_size = request.param
    config = LLMConfig(
        model_loading_config=dict(model_id=MODEL_ID, model_source=MODEL_SOURCE),
        server_cls=MPTestServer,
        deployment_config=dict(
            num_replicas=2,
            request_router_config=RequestRouterConfig(
                request_router_class=KVAwareRouter
            ),
        ),
        engine_kwargs=dict(
            revision=MODEL_REVISION,
            tokenizer_revision=MODEL_REVISION,
            tensor_parallel_size=1,
            block_size=BLOCK_SIZE,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            num_gpu_blocks_override=65,
            enforce_eager=True,
            gpu_memory_utilization=0.4,
            enable_prefix_caching=True,
            enable_prompt_tokens_details=True,
            enable_force_include_usage=True,
            worker_extension_cls="lmcache_mp_utils.MPWorkerExtension",
            # __lmcache_mp_config_start__
            kv_transfer_config=dict(
                kv_connector="LMCacheMPConnector",
                kv_connector_module_path="lmcache.integration.vllm.lmcache_mp_connector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "lmcache.mp.autostart": True,
                    "lmcache.mp.autostart.server_args": (
                        f"--chunk-size {chunk_size} --l1-size-gb 1 --no-l1-use-lazy "
                        "--eviction-policy LRU --supported-transfer-mode lmcache_driven"
                    ),
                    "lmcache.mp.mq_timeout": 5,
                    "lmcache.mp.heartbeat_interval": 1,
                },
            ),
            # __lmcache_mp_config_end__
        ),
        experimental_configs={"KV_EVENTS_PORT_BASE": 23000},
        runtime_env=dict(
            env_vars={
                "RAY_SERVE_ENABLE_DIRECT_INGRESS": "1",
                "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
            },
        ),
        log_engine_metrics=False,
    )
    try:
        with patch_ingress():
            handle = serve.run(build_kv_app(config), name=APP_NAME)
        yield handle, chunk_size, mp_runtime
    finally:
        serve.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_lmcache_mp_routes_to_cpu_prefix_and_reloads(mp_app):
    handle, chunk_size, node_ips = mp_app
    endpoints = await discover_replica_endpoints(handle, 2)
    if not node_ips:
        assert len({host for host, _ in endpoints.values()}) == 1, endpoints
    router = serve.get_deployment_handle("LLMRouter", app_name=APP_NAME)

    async def registered():
        states = await router.broadcast("get_kv_routing_state").results_async()
        assert states
        for state in states:
            assert set(state["replicas"].values()) == set(endpoints), state
        if node_ips:
            ingresses = await router.broadcast("get_token_push_report").results_async()
            assert set(node_ips.values()) <= {
                r["node_ip"] for r in ingresses
            }, ingresses
        return True

    await async_wait_for_condition(registered, timeout=90)
    state = (await router.broadcast("get_kv_routing_state").results_async())[0]
    cached, cold = sorted(state["replicas"])
    a, b = (endpoints[state["replicas"][w]] for w in (cached, cold))
    reports = [mp_report(endpoint) for endpoint in (a, b)]
    assert_mp_owners(endpoints, reports, node_ips)
    assert all(r["model_revision"] == MODEL_REVISION for r in reports), reports
    assert all(r["chunk_size"] == chunk_size for r in reports), reports
    assert all(r["healthy"] for r in reports), reports
    tokens = tokenize_prompt(a, TARGET)
    blocks = len(tokens) // chunk_size * chunk_size // BLOCK_SIZE
    assert 16 <= blocks < 64, (len(tokens), blocks)
    post_chat(a, TARGET, max_tokens=2)

    async def indexed(cpu_only=False):
        if not cpu_only:
            snapshots = await router.broadcast("get_kv_index_snapshot").results_async()
            for snapshot in snapshots:
                for worker, expected in ((cached, blocks), (cold, 0)):
                    placements = {}
                    for tier in ("device", "host_pinned"):
                        placements[tier] = {
                            block["block_hash"]
                            for partition in snapshot.values()
                            for event in partition["events"]
                            if event["worker_id"] == worker
                            and event["storage_tier"] == tier
                            for block in event["event"]["data"]
                            .get("stored", {})
                            .get("blocks", [])
                        }
                    assert len(placements["host_pinned"]) == expected, snapshot
                    assert placements["host_pinned"] <= placements["device"], snapshot
                    gpu_blocks = len(tokens) // BLOCK_SIZE if worker == cached else 0
                    assert len(placements["device"]) == gpu_blocks, snapshot
        scores = await router.broadcast("get_kv_overlap_scores", tokens).results_async()
        for score in scores:
            assert score[cold]["host_pinned_blocks"] == 0, score
            assert score[cold]["device_blocks"] == 0, score
            if cpu_only:
                assert score[cached]["host_pinned_blocks"] == blocks, score
                assert score[cached]["device_blocks"] == 0, score
                assert score[cached]["host_pinned_extension_blocks"] == blocks, score
            else:
                # Host overlap includes GPU hits; the raw index above proves CPU storage.
                assert (
                    score[cached]["device_blocks"] == len(tokens) // BLOCK_SIZE
                ), score
                assert score[cached]["host_pinned_extension_blocks"] == 0, score
                assert (
                    score[cached]["host_pinned_blocks"]
                    == score[cached]["device_blocks"]
                ), score
                assert (
                    score[cached]["router_credit_blocks"]
                    == score[cached]["device_blocks"]
                ), score
        return True

    await async_wait_for_condition(indexed, timeout=60)
    initial_index = await router.broadcast("get_kv_index_snapshot").results_async()
    assert (
        mp_report(a)["status"]["storage_manager"]["l1_manager"]["total_object_count"]
        > 0
    )
    assert (
        mp_report(b)["status"]["storage_manager"]["l1_manager"]["total_object_count"]
        == 0
    )
    for i in range(3):
        filler = [
            {"role": "user", "content": f"GPU eviction {i}. " + f"filler-{i} " * 200}
        ]
        post_chat(a, filler, max_tokens=1)

    async def cpu_only_indexed():
        return await indexed(cpu_only=True)

    await async_wait_for_condition(cpu_only_indexed, timeout=60)

    async def idle():
        for worker in (cached, cold):
            loads = await router.broadcast("get_worker_load", worker).results_async()
            assert all(
                load is not None
                and load["active_requests"] == 0
                and load["potential_prefill_tokens"] == 0
                and load["potential_decode_blocks"] == 0
                for load in loads
            ), loads
        return True

    await async_wait_for_condition(idle, timeout=30)
    cpu_only_scores = await router.broadcast(
        "get_kv_overlap_scores", tokens
    ).results_async()
    idle_loads = {
        worker: await router.broadcast("get_worker_load", worker).results_async()
        for worker in (cached, cold)
    }
    before = [mp_report(endpoint) for endpoint in (a, b)]
    await router.broadcast("reset_token_pushes").results_async()
    request_id = f"lmcache-mp-{uuid.uuid4().hex}"
    response = post_chat(
        (b[0] if node_ips else "127.0.0.1", 8000),
        TARGET,
        max_tokens=2,
        request_id=request_id,
    )
    routes = await router.broadcast("get_token_push_report").results_async()
    picks = [
        p
        for report in routes
        for p in report["pushes"]
        if p["request_id"] == request_id
    ]
    assert picks and all(
        p["replica_id"] == state["replicas"][cached] for p in picks
    ), routes
    if node_ips:
        # A first nonblocking push may fall back; the MP load is checked below.
        assert all(
            report["node_ip"] == b[0]
            and urlsplit(push["endpoint"]).hostname == a[0] != b[0]
            for report in routes
            for push in report["pushes"]
            if push["request_id"] == request_id
        ), routes
    after = [mp_report(endpoint) for endpoint in (a, b)]
    loads = [r for r in after[0]["loads"] if r["request_id"].startswith(response["id"])]
    assert loads and all(r["success"] for r in loads), after
    assert sum(r["blocks"] for r in loads) >= blocks, loads
    assert after[1]["loads"] == before[1]["loads"], after
    assert (
        response["usage"]["prompt_tokens_details"]["cached_tokens"]
        >= blocks * BLOCK_SIZE - 1
    ), response

    async def gpu_restored():
        scores = await router.broadcast("get_kv_overlap_scores", tokens).results_async()
        for score in scores:
            assert score[cached]["device_blocks"] == len(tokens) // BLOCK_SIZE, score
            assert score[cold]["device_blocks"] == 0, score
        return True

    await async_wait_for_condition(gpu_restored, timeout=60)
    restored_scores = await router.broadcast(
        "get_kv_overlap_scores", tokens
    ).results_async()
    print(
        json.dumps(
            dict(
                chunk_size=chunk_size,
                request_id=request_id,
                response=response,
                routes=picks,
                route_reports=routes,
                node_ips=node_ips,
                initial_index=initial_index,
                cpu_only_scores=cpu_only_scores,
                idle_loads=idle_loads,
                restored_scores=restored_scores,
                before=before,
                after=after,
            ),
            default=str,
        )
    )


def test_retrieve_observer_rejects_failed_and_dropped_loads():
    adapter = SimpleNamespace(
        retrieve_futures={"ok": (None, [1, 2]), "failed": (None, [3])},
        _failed_request_ids={"failed", "dropped"},
    )
    result = (set(), {"ok", "failed", "dropped"})
    reports = []
    assert observe_retrieves(adapter, lambda _: result, reports, set()) is result
    assert {r["request_id"]: (r["blocks"], r["success"]) for r in reports} == {
        "ok": (2, True),
        "failed": (1, False),
    }


def test_stop_owned_server_without_observer(monkeypatch):
    launcher = SimpleNamespace(_process=Mock(), shutdown=Mock())
    launcher._process.poll.return_value = 0
    connector = SimpleNamespace(
        worker_adapter=SimpleNamespace(_mp_server_launcher=launcher)
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.get_kv_transfer_group", lambda: connector
    )
    MPWorkerExtension.mp_stop(SimpleNamespace())
    launcher.shutdown.assert_called_once_with()


def test_two_node_gpu_owners_requires_two_distinct_one_gpu_nodes():
    nodes = [
        dict(
            Alive=True,
            NodeID=f"node-{i}",
            NodeManagerAddress=f"10.0.0.{i}",
            Resources={"GPU": 1},
        )
        for i in (1, 2)
    ]
    assert two_node_gpu_owners(nodes) == {
        "node-1": "10.0.0.1",
        "node-2": "10.0.0.2",
    }
    for invalid in (
        nodes[:1],
        [nodes[0], dict(nodes[1], Resources={"GPU": 2})],
        [nodes[0], dict(nodes[1], Alive=False)],
        [nodes[0], dict(nodes[1], NodeManagerAddress="10.0.0.1")],
    ):
        with pytest.raises(AssertionError):
            two_node_gpu_owners(invalid)


def test_mp_owners_allow_loopback_and_pid_reuse_only_across_nodes():
    endpoints = {"a": ("10.0.0.1", 8001), "b": ("10.0.0.2", 8001)}
    reports = [
        dict(
            replica_id=replica_id,
            node_id=f"node-{i}",
            node_ip=f"10.0.0.{i}",
            server_url="tcp://127.0.0.1:21800",
            server_pid=123,
        )
        for i, replica_id in enumerate(endpoints, 1)
    ]
    owners = {"node-1": "10.0.0.1", "node-2": "10.0.0.2"}
    assert_mp_owners(endpoints, reports, owners)
    with pytest.raises(AssertionError):
        assert_mp_owners(
            endpoints, [reports[0], dict(reports[1], node_id="node-1")], owners
        )
    endpoints["b"] = ("10.0.0.1", 8002)
    reports[1].update(node_id="node-1", node_ip="10.0.0.1")
    with pytest.raises(AssertionError):
        assert_mp_owners(endpoints, reports, owners)
    with pytest.raises(AssertionError):
        assert_mp_owners(endpoints, reports, {})
    reports[1]["server_url"] = "tcp://127.0.0.1:21801"
    with pytest.raises(AssertionError):
        assert_mp_owners(endpoints, reports, {})
    reports[1]["server_pid"] = 124
    assert_mp_owners(endpoints, reports, {})


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mp_app", [256], indirect=True, ids=["chunk-256"])
@pytest.mark.parametrize("operation", ["clear", "restart"])
async def test_lmcache_mp_idle_invalidation_preserves_live_replicas(mp_app, operation):
    """Real cache changes withdraw CPU hints while model/ingress identities live."""
    handle, chunk_size, node_ips = mp_app
    endpoints = await discover_replica_endpoints(handle, 2)
    router = serve.get_deployment_handle("LLMRouter", app_name=APP_NAME)

    async def registered():
        states = await router.broadcast("get_kv_routing_state").results_async()
        assert states and all(
            set(state["replicas"].values()) == set(endpoints) for state in states
        ), states
        for state in states:
            assert {w["worker_id"] for w in state["workers"]} == set(
                state["replicas"]
            ), state
            assert all(w["lifecycle"] == "schedulable" for w in state["workers"]), state
        if node_ips:
            ingresses = await router.broadcast("get_token_push_report").results_async()
            assert set(node_ips.values()) <= {
                r["node_ip"] for r in ingresses
            }, ingresses
        return True

    await async_wait_for_condition(registered, timeout=90)
    identities = await router.broadcast("get_kv_routing_state").results_async()
    state = identities[0]
    cached, cold = sorted(state["replicas"])
    a, b = (endpoints[state["replicas"][w]] for w in (cached, cold))
    models = [mp_report(endpoint) for endpoint in (a, b)]
    assert_mp_owners(endpoints, models, node_ips)
    tokens = tokenize_prompt(a, TARGET)
    blocks = len(tokens) // chunk_size * chunk_size // BLOCK_SIZE
    gpu_target = [{"role": "user", "content": "GPU survivor. " + "cobalt pine " * 150}]
    gpu_tokens = tokenize_prompt(a, gpu_target)
    fresh = [
        {"role": "user", "content": "Fresh MP generation. " + "silver maple " * 140}
    ]
    fresh_tokens = tokenize_prompt(a, fresh)
    fresh_blocks = len(fresh_tokens) // chunk_size * chunk_size // BLOCK_SIZE
    worker = next(w for w in state["workers"] if w["worker_id"] == cached)
    [event_endpoint] = worker["kv_events_endpoints"].values()
    if node_ips:
        assert urlsplit(event_endpoint).hostname == a[0], event_endpoint
    subscriber = zmq.Context.instance().socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect(event_endpoint)
    decoder = msgspec.msgpack.Decoder(KVEventBatch)
    published = []

    def drain_published():
        while subscriber.poll(0):
            _, sequence, payload = subscriber.recv_multipart()
            batch = decoder.decode(payload)
            published.append(
                dict(
                    received=time.time(),
                    published=batch.ts,
                    sequence=int.from_bytes(sequence, "big"),
                    events=msgspec.to_builtins(batch.events),
                )
            )

    try:
        post_chat(a, TARGET, max_tokens=2)

        async def target_stored():
            snapshots = await router.broadcast("get_kv_index_snapshot").results_async()
            assert all(
                len(tier_hashes(s, cached, "host_pinned")) == blocks for s in snapshots
            ), snapshots
            return True

        await async_wait_for_condition(target_stored, timeout=60)
        initial = (await router.broadcast("get_kv_index_snapshot").results_async())[0]
        target_hashes = tier_hashes(initial, cached, "host_pinned")
        for i in range(3):
            post_chat(
                a,
                [
                    {
                        "role": "user",
                        "content": f"Evict GPU {i}. " + f"filler-{i} " * 200,
                    }
                ],
                max_tokens=1,
            )

        async def cpu_only():
            scores = await router.broadcast(
                "get_kv_overlap_scores", tokens
            ).results_async()
            assert all(
                s[cached]["device_blocks"] == 0
                and s[cached]["host_pinned_extension_blocks"] == blocks
                for s in scores
            ), scores
            return True

        await async_wait_for_condition(cpu_only, timeout=60)
        post_chat(a, gpu_target, max_tokens=2)

        async def prepared():
            snapshots = await router.broadcast("get_kv_index_snapshot").results_async()
            for snapshot in snapshots:
                assert target_hashes <= tier_hashes(
                    snapshot, cached, "host_pinned"
                ), snapshot
                assert tier_hashes(snapshot, cached, "device"), snapshot
            for worker_id in (cached, cold):
                loads = await router.broadcast(
                    "get_worker_load", worker_id
                ).results_async()
                assert all(load["active_requests"] == 0 for load in loads), loads
            status = mp_status(a)
            assert status["kv_events"]["pending_stores"] == 0, status
            assert (
                status["storage_manager"]["l1_manager"]["read_locked_count"] == 0
            ), status
            return True

        await async_wait_for_condition(prepared, timeout=60)
        before = await router.broadcast("get_kv_index_snapshot").results_async()
        gpu_hashes = tier_hashes(before[0], cached, "device")
        assert gpu_hashes & tier_hashes(before[0], cached, "host_pinned"), before
        gpu_scores = await router.broadcast(
            "get_kv_overlap_scores", gpu_tokens
        ).results_async()
        assert all(s[cached]["device_blocks"] > 0 for s in gpu_scores), gpu_scores
        source_before = mp_status(a)
        drain_published()
        published.clear()
        host, port = a
        # Restart also budgets two 10s shutdown waits, 10s to observe unhealthy,
        # and the launcher's configured startup deadline before delivery.
        timeout = INVALIDATION_TIMEOUT
        if operation == "restart":
            timeout += 30 + models[0]["server_start_timeout"]
        operation_started = time.monotonic()
        response = requests.post(
            f"http://{host}:{port}/test/mp/{operation}", timeout=155
        )
        response.raise_for_status()
        mutation = response.json()
        idle_trace = []

        async def withdrawn():
            drain_published()
            status = mp_status(a)
            snapshots = await router.broadcast("get_kv_index_snapshot").results_async()
            idle_trace.append(
                dict(
                    ts=time.time(),
                    elapsed=time.monotonic() - operation_started,
                    source=status["kv_events"],
                    objects=status["storage_manager"]["l1_manager"][
                        "total_object_count"
                    ],
                    cpu=[
                        sorted(tier_hashes(s, cached, "host_pinned")) for s in snapshots
                    ],
                    gpu=[sorted(tier_hashes(s, cached, "device")) for s in snapshots],
                )
            )
            assert (
                status["storage_manager"]["l1_manager"]["total_object_count"] == 0
            ), status
            for snapshot in snapshots:
                assert not tier_hashes(snapshot, cached, "host_pinned"), snapshot
                assert tier_hashes(snapshot, cached, "device") == gpu_hashes, snapshot
            removed = {
                h
                for batch in published
                for event in batch["events"]
                if event["type"] == "BlockRemoved" and event["medium"] == "CPU"
                for h in event["block_hashes"]
            }
            assert target_hashes <= removed, published
            return True

        # Only MP HTTP status, ZMQ reception, and router snapshots until convergence.
        await async_wait_for_condition(
            withdrawn,
            timeout=max(0.1, timeout - (time.monotonic() - operation_started)),
        )
        assert idle_trace[-1]["elapsed"] < timeout, idle_trace[-1]
        recovery = mp_status(a)
        if operation == "restart":
            assert mutation["new_pid"] != mutation["old_pid"], mutation
            assert (
                recovery["kv_events"]["incarnation"]
                != source_before["kv_events"]["incarnation"]
            )

            async def reconnected():
                status = mp_status(a)
                assert (
                    status["registered_gpu_ids"] == source_before["registered_gpu_ids"]
                ), status
                assert status["kv_events"]["subscribers"] == 1, status
                await withdrawn()
                return True

            await async_wait_for_condition(reconnected, timeout=30)
        else:
            assert (
                recovery["kv_events"]["incarnation"]
                == source_before["kv_events"]["incarnation"]
            )

        # Observe a full reconnect-backoff window for delayed stale replay.
        replay_deadline = time.monotonic() + 5.1
        while time.monotonic() < replay_deadline:
            await withdrawn()
            await asyncio.sleep(0.1)
        after_identities = await router.broadcast(
            "get_kv_routing_state"
        ).results_async()
        assert {tuple(s["identity"]) for s in after_identities} == {
            tuple(s["identity"]) for s in identities
        }
        assert all(s["replicas"] == state["replicas"] for s in after_identities)
        before_serving = [mp_report(endpoint) for endpoint in (a, b)]
        for old, new in zip(models, before_serving):
            keys = ("node_id", "node_ip", "worker_pid", "instance_id", "replica_id")
            assert tuple(new[k] for k in keys) == tuple(old[k] for k in keys), (
                old,
                new,
            )
        scores = await router.broadcast("get_kv_overlap_scores", tokens).results_async()
        assert all(
            s[cached]["host_pinned_extension_blocks"] == 0 for s in scores
        ), scores
        gpu_credit = (
            max(s[w]["device_blocks"] for s in scores for w in (cached, cold))
            * BLOCK_SIZE
        )
        assert gpu_credit < blocks * BLOCK_SIZE - 1, scores
        result = post_chat(
            (b[0] if node_ips else "127.0.0.1", 8000), TARGET, max_tokens=2
        )
        assert (
            result["usage"]["prompt_tokens_details"]["cached_tokens"] <= gpu_credit
        ), result
        reports = [mp_report(endpoint) for endpoint in (a, b)]
        assert not any(
            r["request_id"].startswith(result["id"])
            for report in reports
            for r in report["loads"]
        ), reports

        # Public serving may choose B; establish fresh storage specifically on A.
        fresh_source = mp_status(a)
        pre_fresh = (await router.broadcast("get_kv_index_snapshot").results_async())[0]
        drain_published()
        fresh_event_offset = len(published)
        post_chat(a, fresh, max_tokens=2)
        fresh_placement = {}

        async def fresh_on_recovered_a():
            drain_published()
            status = mp_status(a)
            assert (
                status["kv_events"]["incarnation"]
                == recovery["kv_events"]["incarnation"]
            ), status
            assert (
                status["kv_events"]["next_seq"] > fresh_source["kv_events"]["next_seq"]
            ), status
            snapshots = await router.broadcast("get_kv_index_snapshot").results_async()
            events = [
                event
                for batch in published[fresh_event_offset:]
                for event in batch["events"]
                if event["type"] == "BlockStored" and event["medium"] == "CPU"
            ]
            expected = set()
            parent = None
            matched = []
            for start in range(0, fresh_blocks * BLOCK_SIZE, chunk_size):
                matches = [
                    event
                    for event in events
                    if event["token_ids"] == fresh_tokens[start : start + chunk_size]
                    and event["parent_block_hash"] == parent
                ]
                assert matches, events
                event = matches[0]
                assert event["block_size"] == BLOCK_SIZE, event
                assert len(event["block_hashes"]) == chunk_size // BLOCK_SIZE, event
                expected.update(event["block_hashes"])
                parent = event["block_hashes"][-1]
                matched.append(event)
            assert expected - tier_hashes(pre_fresh, cached, "host_pinned"), expected
            for snapshot in snapshots:
                assert expected <= tier_hashes(
                    snapshot, cached, "host_pinned"
                ), snapshot
            fresh_placement.update(hashes=sorted(expected), events=matched)
            return True

        await async_wait_for_condition(fresh_on_recovered_a, timeout=60)
        print(
            json.dumps(
                dict(
                    operation=operation,
                    node_ips=node_ips,
                    mutation=mutation,
                    timeout=timeout,
                    models=models,
                    before=before,
                    source_before=source_before,
                    idle_trace=idle_trace,
                    published=published,
                    recovery=recovery,
                    response=result,
                    fresh_source=mp_status(a),
                    fresh_placement=fresh_placement,
                    identities=after_identities,
                ),
                default=str,
            )
        )
    finally:
        subscriber.close(linger=0)
