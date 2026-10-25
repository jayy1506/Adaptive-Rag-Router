import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


@pytest.mark.anyio
async def test_benchmark_api(client):
    response = await client.get("/api/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 5
    systems = {d["system"] for d in data}
    assert "adaptive_router" in systems


@pytest.mark.anyio
async def test_ablation_api(client):
    response = await client.get("/api/ablation")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 6


@pytest.mark.anyio
async def test_predict_endpoint(client):
    payload = {"query": "Who directed Inception?"}
    response = await client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["predicted_k"] in [0, 3, 8, 15]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "router_latency_ms" in data
    assert data["router_latency_ms"] >= 0.0


@pytest.mark.anyio
async def test_route_endpoint(client):
    payload = {"query": "When was Arthur's Magazine first published?"}
    response = await client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["executed_k"] in [0, 3, 8, 15]
    assert isinstance(data["fallback_triggered"], bool)
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.anyio
async def test_query_endpoint_and_headers(client):
    payload = {"query": "When was Arthur's Magazine first published?", "use_cache": True}
    response = await client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "passages" in data
    assert data["executed_k"] in [0, 3, 8, 15]
    assert data["total_tokens"] >= 0

    # Verify custom latency waterfall headers
    assert "x-router-latency-ms" in response.headers
    assert "x-retrieval-latency-ms" in response.headers
    assert "x-generation-latency-ms" in response.headers
    assert "x-total-latency-ms" in response.headers

    assert float(response.headers["x-router-latency-ms"]) >= 0.0
    assert float(response.headers["x-retrieval-latency-ms"]) >= 0.0
    assert float(response.headers["x-generation-latency-ms"]) >= 0.0
    assert float(response.headers["x-total-latency-ms"]) >= 0.0
