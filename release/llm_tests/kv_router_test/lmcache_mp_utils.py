"""Test-only MP ownership and completed-load observations on the model node."""

import httpx

from ray import serve
from ray.serve.llm import LLMServer


def observe_retrieves(adapter, original, reports, finished_ids):
    pending = {
        key: len(blocks) for key, (_, blocks) in adapter.retrieve_futures.items()
    }
    result = original(finished_ids)
    for request_id in result[1] or ():
        if request_id in pending:
            reports.append(
                dict(
                    request_id=request_id,
                    blocks=pending[request_id],
                    success=request_id not in adapter._failed_request_ids,
                )
            )
    return result


class MPWorkerExtension:
    """vLLM's existing worker extension hook keeps control on the GPU owner."""

    def mp_observe(self):
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group()
        assert (
            type(connector).__module__
            == "lmcache.integration.vllm.lmcache_mp_connector"
        )
        self._mp_adapter = connector.worker_adapter
        self._mp_loads = []
        original = self._mp_adapter.get_finished
        self._mp_adapter.get_finished = lambda ids: observe_retrieves(
            self._mp_adapter, original, self._mp_loads, ids
        )
        assert self._mp_adapter._mp_server_launcher._process is not None
        return self.mp_report()

    def mp_report(self):
        from importlib.metadata import version

        import lmcache
        import lmcache.cuda_ops
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group()
        adapter = self._mp_adapter
        process = adapter._mp_server_launcher._process
        return dict(
            server_url=adapter._server_url,
            server_pid=process.pid if process else None,
            instance_id=adapter.instance_id,
            healthy=adapter.is_healthy,
            chunk_size=adapter.lmcache_tokens_per_chunk,
            model_revision=self.model_config.hf_config._commit_hash,
            connector_class=f"{type(connector).__module__}.{type(connector).__qualname__}",
            lmcache_module=lmcache.__file__,
            cuda_extension=lmcache.cuda_ops.__file__,
            lmcache_version=version("lmcache"),
            loads=list(self._mp_loads),
        )

    def mp_stop(self):
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        adapter = get_kv_transfer_group().worker_adapter
        process = adapter._mp_server_launcher._process
        adapter._mp_server_launcher.shutdown()
        assert process is None or process.poll() is not None


class MPTestServer(LLMServer):
    async def __init__(self, llm_config, **kwargs):
        llm_config = llm_config.model_copy(deep=True)
        rank = serve.get_replica_context().rank.local_rank
        extra = llm_config.engine_kwargs["kv_transfer_config"][
            "kv_connector_extra_config"
        ]
        extra["lmcache.mp.server_urls"] = [f"tcp://127.0.0.1:{21800 + rank}"]
        self._mp_http_port = 21900 + rank
        extra[
            "lmcache.mp.autostart.server_args"
        ] += f" --http-port {self._mp_http_port} --prometheus-port {22000 + rank}"
        llm_config.update_engine_kwargs(
            kv_transfer_config=llm_config.engine_kwargs["kv_transfer_config"]
        )
        self._mp_started = False
        await super().__init__(llm_config, **kwargs)
        self._mp_started = True
        try:
            await self.collective_rpc("mp_observe", timeout=30)
        except BaseException:
            await self.__del__()
            raise

    async def __serve_build_asgi_app__(self):
        app = await super().__serve_build_asgi_app__()

        @app.get("/test/mp")
        async def report():
            [worker] = await self.collective_rpc("mp_report", timeout=30)
            worker[
                "replica_id"
            ] = serve.get_replica_context().replica_id.to_full_id_str()
            worker["status"] = await self.mp_status()
            return worker

        @app.get("/test/mp/status")
        async def status():
            # This read must never wake the model engine during idle-invalidation tests.
            return await self.mp_status()

        return app

    async def mp_status(self):
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"http://127.0.0.1:{self._mp_http_port}/status")
            response.raise_for_status()
            return response.json()

    async def __del__(self):
        try:
            if getattr(self, "_mp_started", False):
                self._mp_started = False
                await self.collective_rpc("mp_stop", timeout=30)
        finally:
            await super().__del__()
