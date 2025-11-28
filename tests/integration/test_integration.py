# tests/integration/test_api.py
import pytest
from httpx import AsyncClient
from src.api.main import app

@pytest.mark.asyncio
async def test_quiz_endpoint_valid():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/", json={
            "email": "test@example.com",
            "secret": "test_secret",
            "url": "https://example.com/quiz"
        })
        assert response.status_code == 200
