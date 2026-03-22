"""Tests for vec_inject types — VecInjectMatch, VecInjectResult, VecInjectMeta."""

from __future__ import annotations

from chuk_lazarus.inference.context.vec_inject._types import (
    SourceType,
    VecInjectMatch,
    VecInjectMeta,
    VecInjectResult,
)


class TestSourceType:
    def test_values(self):
        assert SourceType.DOCUMENT == 0
        assert SourceType.GENERATED == 1

    def test_is_int(self):
        assert isinstance(SourceType.DOCUMENT, int)


class TestVecInjectMatch:
    def test_required_fields(self):
        m = VecInjectMatch(
            token_id=42,
            coefficient=0.5,
            score=0.8,
            window_id=3,
            position=10,
        )
        assert m.token_id == 42
        assert m.coefficient == 0.5
        assert m.score == 0.8
        assert m.window_id == 3
        assert m.position == 10

    def test_defaults(self):
        m = VecInjectMatch(
            token_id=1,
            coefficient=0.1,
            score=0.5,
            window_id=0,
            position=0,
        )
        assert m.distinctive is True
        assert m.source_id == 0
        assert m.source_type == SourceType.DOCUMENT

    def test_distinctive_false(self):
        m = VecInjectMatch(
            token_id=1,
            coefficient=0.1,
            score=0.5,
            window_id=0,
            position=0,
            distinctive=False,
        )
        assert m.distinctive is False

    def test_source_type_generated(self):
        m = VecInjectMatch(
            token_id=1,
            coefficient=0.1,
            score=0.5,
            window_id=0,
            position=0,
            source_type=SourceType.GENERATED,
            source_id=2,
        )
        assert m.source_type == SourceType.GENERATED
        assert m.source_id == 2

    def test_serialisation_roundtrip(self):
        m = VecInjectMatch(
            token_id=42,
            coefficient=-0.3,
            score=0.99,
            window_id=5,
            position=20,
            distinctive=False,
            source_id=3,
            source_type=SourceType.GENERATED,
        )
        d = m.model_dump()
        m2 = VecInjectMatch(**d)
        assert m2 == m


class TestVecInjectResult:
    def test_empty_defaults(self):
        r = VecInjectResult()
        assert r.matches == []
        assert r.retrieval_ms == 0.0
        assert r.injection_layer == 30
        assert r.routing_confident is True
        assert r.top_score == 0.0
        assert r.routing_stage == "kspace"

    def test_with_matches(self):
        m = VecInjectMatch(
            token_id=1,
            coefficient=0.5,
            score=0.9,
            window_id=0,
            position=5,
        )
        r = VecInjectResult(
            matches=[m],
            retrieval_ms=1.5,
            injection_layer=30,
            routing_confident=True,
            top_score=0.9,
            routing_stage="h4",
        )
        assert len(r.matches) == 1
        assert r.routing_stage == "h4"

    def test_fallback_stage(self):
        r = VecInjectResult(
            routing_confident=False,
            routing_stage="fallback",
        )
        assert r.routing_confident is False
        assert r.routing_stage == "fallback"


class TestVecInjectMeta:
    def test_fields(self):
        m = VecInjectMeta(
            retrieval_layer=29,
            kv_head=1,
            query_head=4,
            injection_layer=30,
        )
        assert m.retrieval_layer == 29
        assert m.kv_head == 1
        assert m.query_head == 4
        assert m.injection_layer == 30

    def test_serialisation(self):
        m = VecInjectMeta(
            retrieval_layer=17,
            kv_head=0,
            query_head=0,
            injection_layer=18,
        )
        d = m.model_dump()
        m2 = VecInjectMeta(**d)
        assert m2 == m
