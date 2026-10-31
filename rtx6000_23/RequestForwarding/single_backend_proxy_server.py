"""Single-vLLM OpenAI proxy that reuses the PD proxy's Responses adapter."""

import argparse
import asyncio
import copy
import json
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import load_balance_proxy_server as responses
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass
class SingleBackendStats:
    requests: int = 0
    active_streams: int = 0
    completed_streams: int = 0
    errors: int = 0
    response_headers_received: int = 0
    first_sse_received: int = 0

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


class SingleVllmBackendRunner:
    """Direct Chat Completions transport for one complete vLLM instance."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        backend_model_name: str,
        max_retries: int,
        retry_delay: float,
        stats: SingleBackendStats,
    ):
        self.client = client
        self.backend_model_name = backend_model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stats = stats

    def _request_data(self, chat_request: dict[str, Any]) -> dict[str, Any]:
        request_data = copy.deepcopy(chat_request)
        request_data["model"] = self.backend_model_name
        return request_data

    @staticmethod
    def _error_response(response: httpx.Response) -> JSONResponse:
        try:
            payload = response.json()
        except ValueError:
            payload = {
                "error": {
                    "type": "backend_error",
                    "code": f"backend_http_{response.status_code}",
                    "message": "The upstream vLLM returned a non-JSON error response.",
                }
            }
        if not isinstance(payload, dict):
            payload = {"error": {"message": "The upstream vLLM returned an invalid error response."}}
        return JSONResponse(status_code=response.status_code, content=payload)

    def _headers(self) -> dict[str, str]:
        headers = {"X-Request-Id": uuid.uuid4().hex}
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def __call__(
        self,
        _request: Request,
        chat_request: dict[str, Any],
        _state: responses.ResponseRequestState,
        trace: responses.WireTrace,
        _raw_body: bytes,
    ) -> JSONResponse | StreamingResponse:
        trace_token = responses._active_wire_trace.set(trace)
        try:
            return await self.request_chat(chat_request)
        finally:
            responses._active_wire_trace.reset(trace_token)

    async def request_chat(self, chat_request: dict[str, Any]) -> JSONResponse | StreamingResponse:
        request_data = self._request_data(chat_request)
        self.stats.requests += 1
        if bool(request_data.get("stream", False)):
            return await self._stream_chat(request_data)
        return await self._post_chat(request_data)

    async def _post_chat(self, request_data: dict[str, Any]) -> JSONResponse:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.post(
                    "/chat/completions", json=request_data, headers=self._headers()
                )
                self.stats.response_headers_received += 1
                if response.status_code >= 500 and attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue
                if response.status_code >= 400:
                    self.stats.errors += 1
                    return self._error_response(response)
                try:
                    return JSONResponse(status_code=response.status_code, content=response.json())
                except ValueError:
                    self.stats.errors += 1
                    return JSONResponse(
                        status_code=502,
                        content={"error": {"code": "invalid_upstream_response", "message": "vLLM returned invalid JSON."}},
                    )
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue
        self.stats.errors += 1
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "code": "upstream_transport_error",
                    "message": "The upstream vLLM connection failed.",
                }
            },
        )

    async def _stream_chat(self, request_data: dict[str, Any]) -> JSONResponse | StreamingResponse:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            stream_context = self.client.stream(
                "POST", "/chat/completions", json=request_data, headers=self._headers()
            )
            try:
                response = await stream_context.__aenter__()
                self.stats.response_headers_received += 1
                if response.status_code >= 500 and attempt < self.max_retries:
                    await stream_context.__aexit__(None, None, None)
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue
                if response.status_code >= 400:
                    await response.aread()
                    error = self._error_response(response)
                    await stream_context.__aexit__(None, None, None)
                    self.stats.errors += 1
                    return error

                async def body_iterator():
                    first_chunk = True
                    self.stats.active_streams += 1
                    try:
                        async for chunk in response.aiter_bytes():
                            if first_chunk and chunk:
                                first_chunk = False
                                self.stats.first_sse_received += 1
                            yield chunk
                        self.stats.completed_streams += 1
                    except Exception:
                        self.stats.errors += 1
                        raise
                    finally:
                        self.stats.active_streams -= 1
                        with suppress(Exception):
                            await stream_context.__aexit__(None, None, None)

                return StreamingResponse(
                    body_iterator(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            except httpx.RequestError as exc:
                last_error = exc
                with suppress(Exception):
                    await stream_context.__aexit__(None, None, None)
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue
        self.stats.errors += 1
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "code": "upstream_transport_error",
                    "message": "The upstream vLLM connection failed.",
                }
            },
        )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8078)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--backend-model-name", required=True)
    parser.add_argument("--multimodal-backend-url", default=None)
    parser.add_argument("--multimodal-backend-model", default=None)
    parser.add_argument("--multimodal-max-image-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=0.2)
    parser.add_argument("--responses-sse-heartbeat-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    args.backend_url = args.backend_url.rstrip("/")
    if not args.backend_url.endswith("/v1"):
        args.backend_url = f"{args.backend_url}/v1"
    if bool(args.multimodal_backend_url) != bool(args.multimodal_backend_model):
        raise ValueError(
            "--multimodal-backend-url and --multimodal-backend-model must be provided together."
        )
    if args.multimodal_backend_url:
        args.multimodal_backend_url = args.multimodal_backend_url.rstrip("/")
        if not args.multimodal_backend_url.endswith("/v1"):
            args.multimodal_backend_url = f"{args.multimodal_backend_url}/v1"
    if args.multimodal_max_image_bytes <= 0:
        raise ValueError("--multimodal-max-image-bytes must be positive.")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1.")
    if args.retry_delay < 0:
        raise ValueError("--retry-delay must be >= 0.")
    if args.responses_sse_heartbeat_seconds < 0:
        raise ValueError("--responses-sse-heartbeat-seconds must be >= 0.")
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    # The shared Responses adapter uses these fields for image conversion and
    # route affinity. The injected runner still owns the only upstream client.
    responses.global_args = SimpleNamespace(
        responses_sse_heartbeat_seconds=args.responses_sse_heartbeat_seconds,
        multimodal_backend_url=args.multimodal_backend_url,
        multimodal_backend_model=args.multimodal_backend_model,
        multimodal_max_image_bytes=args.multimodal_max_image_bytes,
    )
    stats = SingleBackendStats()
    client = httpx.AsyncClient(base_url=args.backend_url, timeout=None)
    app.state.stats = stats
    app.state.client = client
    app.state.runner = SingleVllmBackendRunner(
        client, args.backend_model_name, args.max_retries, args.retry_delay, stats
    )
    try:
        response = await client.get("/models", headers=app.state.runner._headers())
        response.raise_for_status()
    except httpx.HTTPError as exc:
        await client.aclose()
        raise RuntimeError(f"Single vLLM /v1/models check failed: {exc}") from exc
    yield
    await client.aclose()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    try:
        request_data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(status_code=400, content={"error": {"code": "invalid_json", "message": "Invalid JSON request."}})
    if not isinstance(request_data, dict):
        return JSONResponse(status_code=400, content={"error": {"code": "invalid_request", "message": "Request body must be a JSON object."}})
    return await request.app.state.runner.request_chat(request_data)


@app.post("/v1/responses")
async def handle_responses(request: Request):
    return await responses.handle_responses(request, backend_runner=request.app.state.runner)


@app.get("/v1/responses/{response_id}")
async def retrieve_response(response_id: str):
    return await responses.retrieve_response(response_id)


@app.get("/v1/models")
async def list_models(request: Request):
    try:
        response = await request.app.state.client.get(
            "/models", headers=request.app.state.runner._headers()
        )
    except httpx.RequestError:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "code": "upstream_transport_error",
                    "message": "The upstream vLLM connection failed.",
                }
            },
        )
    if response.status_code >= 400:
        return SingleVllmBackendRunner._error_response(response)
    try:
        return JSONResponse(content=response.json())
    except ValueError:
        return JSONResponse(status_code=502, content={"error": {"code": "invalid_upstream_response", "message": "vLLM returned invalid JSON."}})


@app.get("/healthcheck")
async def healthcheck(request: Request):
    try:
        response = await request.app.state.client.get(
            "/models", headers=request.app.state.runner._headers()
        )
        upstream_status = response.status_code
    except httpx.RequestError:
        upstream_status = None
    return {
        "status": "ok" if upstream_status == 200 else "degraded",
        "backend_url": request.app.state.args.backend_url,
        "backend_model_name": request.app.state.args.backend_model_name,
        "upstream_models_status": upstream_status,
        "stats": request.app.state.stats.snapshot(),
    }


@app.get("/metrics")
async def metrics(request: Request):
    return request.app.state.stats.snapshot()


if __name__ == "__main__":
    import uvicorn

    app.state.args = parse_args()
    uvicorn.run(app, host=app.state.args.host, port=app.state.args.port)
