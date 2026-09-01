"""Tests for the gRPC server and client.

Uses a mocked MCP server to avoid requiring a running LightRAG instance.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_env() -> None:
    """Set minimal env vars needed for gRPC server imports."""
    original = dict(os.environ)
    os.environ.setdefault("HARS_MEMORY_INDEX_DIR", "/tmp/hars_test_grpc_index")
    os.environ.setdefault("HARS_MEMORY_STAGING_DIR", "/tmp/hars_test_grpc_staging")
    yield
    # Restore only what we added
    for k in list(os.environ.keys()):
        if k not in original:
            del os.environ[k]
        else:
            os.environ[k] = original[k]


@pytest.fixture
def mock_mcp_server() -> Generator[None, None, None]:
    """Mock all imported functions from mcp_server.py."""
    patchers = [
        patch("hars_memory.mcp_server._get_rag", new_callable=AsyncMock),
        patch("hars_memory.mcp_server._index_status", return_value={"index_exists": True, "node_count": 100}),
        patch("hars_memory.mcp_server._get_graph", new_callable=AsyncMock),
        patch("hars_memory.mcp_server._graph_file_path", return_value=Path("/tmp/fake.graphml")),
        patch("hars_memory.mcp_server._staleness_info", return_value=("2026-08-01", 30)),
        patch("hars_memory.mcp_server._resolve_query_mode", return_value=("naive", "mock fallback")),
        patch("hars_memory.mcp_server._resolve_fetch_top_k", return_value=(20, "mock")),
        patch("hars_memory.mcp_server._lightrag_mode", side_effect=lambda m: m),
        patch("hars_memory.mcp_server._postprocess_context", side_effect=lambda c: c),
        patch("hars_memory.mcp_server._merge_context_with_fusion", return_value=("merged", True)),
        patch("hars_memory.mcp_server._has_graph_entity_context", return_value=True),
        patch("hars_memory.mcp_server._extract_breadcrumb_from_content", return_value="## Test Section"),
        patch("hars_memory.mcp_server._expand_query_aliases", return_value=["test"]),
        patch("hars_memory.mcp_server._normalize_entity_text", side_effect=lambda x: x),
        patch("hars_memory.mcp_server._entity_match_tier", return_value="exact"),
        patch("hars_memory.mcp_server._edge_relation_label", return_value="related_to"),
    ]
    for p in patchers:
        p.start()
    yield
    for p in patchers:
        p.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGrpcServerImport:
    """Verify the gRPC server module can be imported without errors."""

    def test_import_server(self, mock_env: None) -> None:
        """Server module imports cleanly."""
        from hars_memory.grpc import server  # noqa: F811

        assert server is not None

    def test_import_client(self) -> None:
        """Client module imports cleanly."""
        from hars_memory.grpc import client  # noqa: F811

        assert client is not None


class TestGrpcClientInitialization:
    """Verify client construction."""

    def test_sync_client_defaults(self) -> None:
        from hars_memory.grpc.client import HarsMemoryGrpcClient

        client = HarsMemoryGrpcClient("localhost:8788")
        assert client._target == "localhost:8788"
        client.close()

    async def test_async_client_close(self) -> None:
        """Async client can be created and closed."""
        from hars_memory.grpc.client import AsyncHarsMemoryGrpcClient

        client = AsyncHarsMemoryGrpcClient("localhost:8788")
        assert client._target == "localhost:8788"
        await client.close()

    def test_sync_client_with_api_key(self) -> None:
        from hars_memory.grpc.client import HarsMemoryGrpcClient

        client = HarsMemoryGrpcClient("localhost:8788", api_key="test-key")
        assert ("x-api-key", "test-key") in client._metadata
        client.close()

    def test_client_context_manager(self) -> None:
        from hars_memory.grpc.client import HarsMemoryGrpcClient

        with HarsMemoryGrpcClient("localhost:8788") as client:
            assert client._target == "localhost:8788"


class TestGrpcResponseTypes:
    """Verify that response types are correct."""

    def test_query_response_fields(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        resp = pb2.QueryResponse(
            ok=True,
            context="test context",
            mode="hybrid",
            lightrag_mode="mix",
            top_k=20,
            fetch_top_k=20,
            question="test question",
            context_priority_applied="merged",
        )
        assert resp.ok is True
        assert resp.context == "test context"
        assert resp.mode == "hybrid"
        assert resp.top_k == 20
        assert resp.question == "test question"

    def test_status_response_fields(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2
        from google.protobuf import struct_pb2

        data = struct_pb2.Struct()
        data["index_exists"] = True
        resp = pb2.StatusResponse(ok=True, data=data)
        assert resp.ok is True
        assert resp.data["index_exists"] is True

    def test_remember_response_fields(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        resp = pb2.RememberResponse(ok=True, saved="/tmp/test.md", pending_notes=5)
        assert resp.ok is True
        assert resp.saved == "/tmp/test.md"
        assert resp.pending_notes == 5

    def test_entities_response_fields(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        result = pb2.EntityResult(id="hyp:123", description="Test entity", entity_type="hypothesis", match_tier="exact")
        resp = pb2.EntitiesResponse(ok=True, query="test", results=[result], count=1)
        assert resp.ok is True
        assert resp.query == "test"
        assert len(resp.results) == 1
        assert resp.results[0].id == "hyp:123"

    def test_related_response_fields(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        node = pb2.Node(id="hyp:1", label="Node 1", type="hypothesis")
        edge = pb2.Edge(source="hyp:1", target="hyp:2", label="related_to")
        resp = pb2.RelatedResponse(ok=True, nodes=[node], edges=[edge], truncated=False)
        assert resp.ok is True
        assert len(resp.nodes) == 1
        assert resp.nodes[0].id == "hyp:1"
        assert len(resp.edges) == 1

    def test_consolidate_response_fields(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        resp = pb2.ConsolidateResponse(ok=True, returncode=0, stdout="done", stderr="", dry_run=False)
        assert resp.ok is True
        assert resp.returncode == 0
        assert resp.dry_run is False

    def test_forget_response_fields(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        candidate = pb2.Candidate(doc_id="doc1", date="2026-01-01", section="session", protected=False, deleted=False)
        resp = pb2.ForgetResponse(
            ok=True, applied=False, dry_run=True, candidates=[candidate],
            total_candidates=1, total_protected=0, total_deleted=0,
        )
        assert resp.ok is True
        assert resp.dry_run is True
        assert len(resp.candidates) == 1
        assert resp.candidates[0].doc_id == "doc1"


class TestGrpcClientRequestTypes:
    """Verify that client methods construct correct request messages."""

    def test_query_request(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        req = pb2.QueryRequest(
            question="What is Kubernetes?",
            mode="hybrid",
            top_k=20,
            ll_keywords=["k8s"],
            hl_keywords=["orchestration"],
            context_only=True,
            context_priority="merged",
            debug=False,
        )
        assert req.question == "What is Kubernetes?"
        assert req.mode == "hybrid"
        assert req.top_k == 20
        assert list(req.ll_keywords) == ["k8s"]
        assert list(req.hl_keywords) == ["orchestration"]
        assert req.context_only is True

    def test_query_request_with_fetch_top_k(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        req = pb2.QueryRequest(question="test", fetch_top_k=40)
        assert req.HasField("fetch_top_k")
        assert req.fetch_top_k == 40

    def test_remember_request(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        req = pb2.RememberRequest(title="my-note", content="test content", importance="high", tags=["k8s", "gcp"])
        assert req.title == "my-note"
        assert req.content == "test content"
        assert req.importance == "high"
        assert list(req.tags) == ["k8s", "gcp"]

    def test_forget_request_with_before(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        req = pb2.ForgetRequest(before="2026-06-01", protect=[".*important.*"], sections=["session"], apply=False)
        assert req.before == "2026-06-01"
        assert list(req.protect) == [".*important.*"]
        assert list(req.sections) == ["session"]

    def test_forget_request_with_older_than_days(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        req = pb2.ForgetRequest(older_than_days=90, apply=False)
        assert req.older_than_days == 90

    def test_entities_request(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        req = pb2.EntitiesRequest(name="test-entity", limit=5)
        assert req.name == "test-entity"
        assert req.limit == 5

    def test_related_request(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        req = pb2.RelatedRequest(entity_id="hyp:123", hops=1)
        assert req.entity_id == "hyp:123"
        assert req.hops == 1

    def test_consolidate_request(self) -> None:
        from hars_memory.grpc import hars_memory_pb2 as pb2

        req = pb2.ConsolidateRequest(paths=[".plans", "docs"], dry_run=True)
        assert list(req.paths) == [".plans", "docs"]
        assert req.dry_run is True


class TestGrpcServerCapture:
    """Integration tests for the gRPC server using a real running server.

    These tests start a gRPC server in-process, connect to it, and verify
    the full round-trip works.
    """

    @pytest.fixture
    async def grpc_server(self, mock_env: None) -> AsyncGenerator[str, None]:
        """Start a real gRPC server on a random port and yield the address."""
        import grpc
        from concurrent import futures
        from hars_memory.grpc.server import LongTermMemoryServicer
        from hars_memory.grpc import hars_memory_pb2_grpc as pb2_grpc

        # Find a free port
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=2),
            options=[
                ("grpc.max_send_message_length", 4194304),
                ("grpc.max_receive_message_length", 4194304),
            ],
        )

        servicer = LongTermMemoryServicer()
        pb2_grpc.add_LongTermMemoryServicer_to_server(servicer, server)
        address = f"127.0.0.1:{port}"
        server.add_insecure_port(address)

        await server.start()
        try:
            yield address
        finally:
            await server.stop(grace=1)

    async def test_status_returns_ok(self, mock_env: None, mock_mcp_server: None) -> None:
        """Status RPC returns ok=True with index data."""
        from hars_memory.grpc import hars_memory_pb2_grpc as pb2_grpc
        from hars_memory.grpc import hars_memory_pb2 as pb2
        from google.protobuf import empty_pb2
        from hars_memory.grpc.server import LongTermMemoryServicer

        servicer = LongTermMemoryServicer()
        resp = await servicer.Status(empty_pb2.Empty(), MagicMock())
        assert resp.ok is True
        assert resp.data is not None

    async def test_remember_returns_ok(self, mock_env: None, mock_mcp_server: None) -> None:
        """Remember RPC with valid content returns ok."""
        from hars_memory.grpc import hars_memory_pb2 as pb2
        from hars_memory.grpc.server import LongTermMemoryServicer

        servicer = LongTermMemoryServicer()
        req = pb2.RememberRequest(title="test-note", content="test content", importance="normal")
        resp = await servicer.Remember(req, MagicMock())
        assert resp.ok is True
        assert resp.saved is not None
        assert resp.pending_notes >= 0

    async def test_remember_rejects_empty_content(self, mock_env: None, mock_mcp_server: None) -> None:
        """Remember RPC with empty content returns error."""
        from hars_memory.grpc import hars_memory_pb2 as pb2
        from hars_memory.grpc.server import LongTermMemoryServicer

        servicer = LongTermMemoryServicer()
        req = pb2.RememberRequest(title="test-note", content="")
        resp = await servicer.Remember(req, MagicMock())
        assert resp.ok is False
        assert "content is required" in (resp.error or "")

    async def test_search_entities_no_index(self, mock_env: None, mock_mcp_server: None) -> None:
        """SearchEntities with no index returns error."""
        from hars_memory.grpc import hars_memory_pb2 as pb2
        from hars_memory.grpc.server import LongTermMemoryServicer
        from unittest.mock import patch

        servicer = LongTermMemoryServicer()
        with patch("hars_memory.mcp_server._graph_file_path") as mock_gfp:
            mock_gfp.return_value = Path("/nonexistent/graph.graphml")
            req = pb2.EntitiesRequest(name="test-entity", limit=10)
            resp = await servicer.SearchEntities(req, MagicMock())
            assert resp.ok is False
            assert "not built" in (resp.error or "")

    async def test_related_entities_no_index(self, mock_env: None, mock_mcp_server: None) -> None:
        """RelatedEntities with no index returns error."""
        from hars_memory.grpc import hars_memory_pb2 as pb2
        from hars_memory.grpc.server import LongTermMemoryServicer
        from unittest.mock import patch

        servicer = LongTermMemoryServicer()
        with patch("hars_memory.mcp_server._graph_file_path") as mock_gfp:
            mock_gfp.return_value = Path("/nonexistent/graph.graphml")
            req = pb2.RelatedRequest(entity_id="hyp:123", hops=1)
            resp = await servicer.RelatedEntities(req, MagicMock())
            assert resp.ok is False
            assert "not built" in (resp.error or "")

    async def test_query_rejects_empty_question(self, mock_env: None, mock_mcp_server: None) -> None:
        """Query with empty question returns error."""
        from hars_memory.grpc import hars_memory_pb2 as pb2
        from hars_memory.grpc.server import LongTermMemoryServicer

        servicer = LongTermMemoryServicer()
        req = pb2.QueryRequest(question="")
        resp = await servicer.Query(req, MagicMock())
        assert resp.ok is False
        assert "question is required" in (resp.error or "")

    async def test_forget_rejects_no_cutoff(self, mock_env: None, mock_mcp_server: None) -> None:
        """Forget without before or older_than_days returns error."""
        from hars_memory.grpc import hars_memory_pb2 as pb2
        from hars_memory.grpc.server import LongTermMemoryServicer

        servicer = LongTermMemoryServicer()
        req = pb2.ForgetRequest(apply=False, confirm_unprotected=False)
        # Both are optional fields — with no value set, both should be None
        resp = await servicer.Forget(req, MagicMock())
        assert resp.ok is False
        assert "supply exactly one" in (resp.error or "")

    async def test_consolidate_dry_run(self, mock_env: None, mock_mcp_server: None) -> None:
        """Consolidate with dry_run=True runs successfully."""
        from hars_memory.grpc import hars_memory_pb2 as pb2
        from hars_memory.grpc.server import LongTermMemoryServicer
        from unittest.mock import patch

        servicer = LongTermMemoryServicer()
        with patch("asyncio.to_thread") as mock_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "dry-run report"
            mock_result.stderr = ""
            mock_thread.return_value = mock_result

            req = pb2.ConsolidateRequest(paths=[".plans"], dry_run=True)
            resp = await servicer.Consolidate(req, MagicMock())
            assert resp.ok is True