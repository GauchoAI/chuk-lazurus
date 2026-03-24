"""Tests for knowledge store v10 — InjectionEntry, KnowledgeStore, routing, serialization."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np

from chuk_lazarus.inference.context.arch_config import ArchitectureConfig
from chuk_lazarus.inference.context.knowledge.store import (
    _entries_to_numpy,
    _numpy_to_entries,
)
from chuk_lazarus.inference.context.knowledge_store import (
    InjectionEntry,
    KnowledgeStore,
    SparseKeywordIndex,
    _extract_keywords_from_text,
    extract_window_keywords,
    streaming_prefill,
)

# ── SparseKeywordIndex ───────────────────────────────────────────────


class TestSparseKeywordIndex:
    def test_add_and_query(self):
        idx = SparseKeywordIndex()
        idx.add(0, ["apollo", "launch", "rocket"])
        idx.add(1, ["porridge", "contest", "scotland"])
        idx.add(2, ["apollo", "lunar", "landing"])

        assert idx.query("apollo mission") == [0, 2]
        assert idx.query("porridge eating") == [1]
        assert idx.query("unknown topic") == []

    def test_query_ranking(self):
        idx = SparseKeywordIndex()
        idx.add(0, ["apollo", "lunar"])
        idx.add(1, ["apollo", "lunar", "orbit"])

        # "apollo lunar" hits both with 2 matches each
        results = idx.query("apollo lunar orbit")
        assert results[0] == 1  # 3 hits vs 2

    def test_save_load_roundtrip(self, tmp_path):
        idx = SparseKeywordIndex()
        idx.add(0, ["hello", "world"])
        idx.add(1, ["foo", "bar"])

        path = tmp_path / "index.json"
        idx.save(path)

        loaded = SparseKeywordIndex.load(path)
        assert loaded.entries == idx.entries
        assert loaded.num_keywords == idx.num_keywords

    def test_empty_index(self):
        idx = SparseKeywordIndex()
        assert idx.query("anything") == []
        assert idx.num_keywords == 0

    def test_deduplication(self):
        idx = SparseKeywordIndex()
        idx.add(0, ["hello", "hello", "hello"])
        assert idx.entries["hello"] == [0]


# ── Keyword extraction ────────────────────────────────────────────────


class TestExtractKeywords:
    def test_filters_function_words(self):
        kws = _extract_keywords_from_text("the quick brown fox is very happy")
        assert "the" not in kws
        assert "is" not in kws
        assert "very" not in kws
        assert "quick" in kws
        assert "brown" in kws

    def test_max_keywords(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        kws = _extract_keywords_from_text(text, max_keywords=3)
        assert len(kws) == 3

    def test_deduplication(self):
        kws = _extract_keywords_from_text("hello hello hello world world")
        assert kws.count("hello") == 1

    def test_min_length(self):
        kws = _extract_keywords_from_text("I am a go to x")
        assert "x" not in kws  # single char excluded (min 2)

    def test_empty_text(self):
        assert _extract_keywords_from_text("") == []


class TestExtractWindowKeywords:
    def test_from_text(self):
        kws = _extract_keywords_from_text("John Coyle won the porridge contest")
        assert len(kws) > 0
        assert "john" in kws or "coyle" in kws

    def test_with_tokenizer(self):
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "Apollo eleven transcript"
        kws = extract_window_keywords([1, 2, 3], mock_tok)
        assert "apollo" in kws


# ── InjectionEntry ───────────────────────────────────────────────────


class TestInjectionEntry:
    def test_creation(self):
        entry = InjectionEntry(
            token_id=42,
            coefficient=4608.0,
            window_id=170,
            position_in_window=103,
            fact_id=0,
        )
        assert entry.token_id == 42
        assert entry.coefficient == 4608.0
        assert entry.window_id == 170
        assert entry.position_in_window == 103

    def test_to_tuple(self):
        entry = InjectionEntry(42, 4608.0, 170, 103, 0)
        t = entry.to_tuple()
        assert t == (42, 4608.0, 170, 103, 0)

    def test_numpy_roundtrip(self):
        entries = [
            InjectionEntry(42, 4608.0, 170, 103, 0),
            InjectionEntry(99, 5120.0, 170, 104, 0),
            InjectionEntry(200, 3000.5, 171, 50, 1),
        ]
        arr = _entries_to_numpy(entries)
        assert arr.shape == (3,)
        assert arr.dtype.names == (
            "token_id",
            "coefficient",
            "window_id",
            "position_in_window",
            "fact_id",
        )

        loaded = _numpy_to_entries(arr)
        assert len(loaded) == 3
        assert loaded[0].token_id == 42
        assert abs(loaded[0].coefficient - 4608.0) < 0.01
        assert loaded[1].token_id == 99
        assert loaded[2].window_id == 171

    def test_empty_numpy_roundtrip(self):
        from chuk_lazarus.inference.context.knowledge.store import _ENTRY_DTYPE

        arr = np.array([], dtype=_ENTRY_DTYPE)
        loaded = _numpy_to_entries(arr)
        assert loaded == []


# ── KnowledgeStore ────────────────────────────────────────────────────


class TestKnowledgeStore:
    def _make_store(self, n_windows: int = 5, entries_per_window: int = 3) -> KnowledgeStore:
        entries = []
        for wid in range(n_windows):
            for pos in range(entries_per_window):
                entries.append(
                    InjectionEntry(
                        token_id=wid * 100 + pos,
                        coefficient=1000.0 + wid * 10 + pos,
                        window_id=wid,
                        position_in_window=wid * 100 + pos,
                        fact_id=wid * entries_per_window + pos,
                    )
                )

        names = ["apollo", "porridge", "lunar", "coyle", "voltara"]
        topics = ["launch", "contest", "orbit", "transcript", "city"]
        keywords = {}
        for wid in range(n_windows):
            keywords[wid] = [names[wid % len(names)], topics[wid % len(topics)]]

        window_tokens = {wid: {wid * 100 + i for i in range(20)} for wid in range(n_windows)}
        idf = dict.fromkeys(range(500), 1.0)

        config = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
        )

        return KnowledgeStore(
            entries=entries,
            window_tokens=window_tokens,
            window_token_lists={},
            idf=idf,
            keywords=keywords,
            config=config,
            num_windows=n_windows,
            num_tokens=n_windows * 512,
        )

    def test_route_keyword(self):
        store = self._make_store()
        assert store.route("coyle discussion", method="keyword") == 3
        assert store.route("apollo analysis", method="keyword") == 0
        assert store.route("xyznonexistent", method="keyword") is None

    def test_get_entries(self):
        store = self._make_store()
        entries = store.get_entries(2)
        assert len(entries) == 3
        assert all(e.window_id == 2 for e in entries)
        # Should be sorted by position
        positions = [e.position_in_window for e in entries]
        assert positions == sorted(positions)

    def test_save_load_roundtrip(self, tmp_path):
        store = self._make_store(n_windows=3, entries_per_window=2)
        store.save(tmp_path / "test_store")

        loaded = KnowledgeStore.load(tmp_path / "test_store")
        assert loaded.num_windows == 3
        assert len(loaded.entries) == 6
        assert loaded.config.retrieval_layer == 29
        assert loaded.num_tokens == 3 * 512

        # Verify entry values
        assert loaded.entries[0].token_id == 0
        assert abs(loaded.entries[0].coefficient - 1000.0) < 0.01
        assert loaded.entries[0].window_id == 0

    def test_save_creates_all_files(self, tmp_path):
        store = self._make_store(n_windows=2, entries_per_window=2)
        out = tmp_path / "store"
        store.save(out)

        assert (out / "manifest.json").exists()
        assert (out / "entries.npz").exists()
        assert (out / "window_tokens.npz").exists()
        assert (out / "idf.json").exists()
        assert (out / "keywords.json").exists()

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["num_entries"] == 4
        assert manifest["num_windows"] == 2
        assert manifest["version"] == 12

    def test_log_stats(self, capsys):
        store = self._make_store(n_windows=2, entries_per_window=2)
        store.log_stats(file=sys.stdout)
        captured = capsys.readouterr()
        assert "KnowledgeStore v12" in captured.out
        assert "4 entries" in captured.out

    def test_v9_store_detected(self, tmp_path):
        """Loading a v9 store should raise a clear error."""
        store_dir = tmp_path / "v9_store"
        store_dir.mkdir()
        # Create v9 files
        np.savez(str(store_dir / "passages.npz"), passages=np.zeros((3, 64)))
        (store_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "num_passages": 3,
                    "arch_config": {
                        "retrieval_layer": 29,
                        "query_head": 4,
                        "injection_layer": 30,
                    },
                }
            )
        )
        import pytest

        with pytest.raises(ValueError, match="v9 format"):
            KnowledgeStore.load(store_dir)

    def test_boundary_residual_roundtrip(self, tmp_path):
        store = self._make_store(n_windows=1, entries_per_window=1)
        store.boundary_residual = mx.array([[[1.0, 2.0, 3.0]]])
        mx.eval(store.boundary_residual)
        store.save(tmp_path / "store")

        loaded = KnowledgeStore.load(tmp_path / "store")
        assert loaded.boundary_residual is not None
        assert loaded.boundary_residual.shape == (1, 1, 3)
        assert float(loaded.boundary_residual[0, 0, 0]) == 1.0


# ── ArchitectureConfig crystal_layer ──────────────────────────────────


class TestCrystalLayer:
    def test_defaults_to_injection_layer(self):
        ac = ArchitectureConfig(retrieval_layer=29, query_head=4, injection_layer=30)
        assert ac.crystal_layer == 30

    def test_explicit_crystal_layer(self):
        ac = ArchitectureConfig(
            retrieval_layer=29, query_head=4, injection_layer=30, crystal_layer=26
        )
        assert ac.crystal_layer == 26

    def test_serialisation_default(self):
        ac = ArchitectureConfig(retrieval_layer=29, query_head=4, injection_layer=30)
        d = ac.to_dict()
        assert "crystal_layer" not in d  # omitted when == injection_layer

    def test_serialisation_explicit(self):
        ac = ArchitectureConfig(
            retrieval_layer=29, query_head=4, injection_layer=30, crystal_layer=26
        )
        d = ac.to_dict()
        assert d["crystal_layer"] == 26

    def test_roundtrip(self):
        ac = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            crystal_layer=26,
            window_size=1024,
        )
        d = ac.to_dict()
        ac2 = ArchitectureConfig.from_dict(d)
        assert ac2.crystal_layer == 26
        assert ac2.window_size == 1024

    def test_window_size_default(self):
        ac = ArchitectureConfig(retrieval_layer=29, query_head=4, injection_layer=30)
        assert ac.window_size == 512

    def test_window_size_serialisation_default_omitted(self):
        ac = ArchitectureConfig(retrieval_layer=29, query_head=4, injection_layer=30)
        d = ac.to_dict()
        assert "window_size" not in d

    def test_window_size_serialisation_non_default(self):
        ac = ArchitectureConfig(
            retrieval_layer=29, query_head=4, injection_layer=30, window_size=1024
        )
        d = ac.to_dict()
        assert d["window_size"] == 1024

    def test_with_geometry_preserves_crystal(self):
        ac = ArchitectureConfig(
            retrieval_layer=29, query_head=4, injection_layer=30, crystal_layer=26
        )
        ac2 = ac.with_geometry(kv_head=1, head_dim=320, hidden_dim=2560)
        assert ac2.crystal_layer == 26


# ── streaming_prefill (mock kv_gen) ──────────────────────────────────


class TestStreamingPrefill:
    def _make_mock_kv_gen(self, hidden_dim: int = 32, vocab_size: int = 256):
        """Create a mock kv_gen with embed_matrix for entry extraction."""
        kv_gen = MagicMock()

        # Create a real embed_matrix for coefficient computation
        embed_matrix = mx.random.normal((vocab_size, hidden_dim))
        mx.eval(embed_matrix)

        backbone = MagicMock()
        backbone.embed_matrix = embed_matrix
        kv_gen.backbone = backbone

        # prefill_to_layer returns residuals at all positions
        def prefill_to_layer(input_ids, target_layer, initial_residual=None):
            S = input_ids.shape[1]
            offset = 1 if initial_residual is not None else 0
            h = mx.random.normal((1, S + offset, hidden_dim))
            mx.eval(h)
            return h

        kv_gen.prefill_to_layer.side_effect = prefill_to_layer
        return kv_gen

    def test_basic_prefill(self):
        hidden_dim = 32
        kv_gen = self._make_mock_kv_gen(hidden_dim=hidden_dim)

        config = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            window_size=10,
            entries_per_window=4,
        )

        tokens = list(range(25))  # 3 windows
        store = streaming_prefill(kv_gen, tokens, config)

        assert store.num_windows == 3
        # Dynamic count: all tokens unique (df=1), so all non-position-0 tokens selected
        # Window 0: 9 tokens (pos 1-9), Window 1: 9 tokens, Window 2: 4 tokens (pos 1-4)
        assert len(store.entries) >= 3 * 4  # at least min_k per window
        assert store.num_tokens == 25
        # All entries should have valid window_ids
        window_ids = {e.window_id for e in store.entries}
        assert window_ids == {0, 1, 2}

    def test_entries_have_coefficients(self):
        kv_gen = self._make_mock_kv_gen(hidden_dim=32)

        config = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            window_size=10,
            entries_per_window=3,
        )

        store = streaming_prefill(kv_gen, list(range(15)), config)
        for entry in store.entries:
            # Coefficients should be non-zero (2x natural)
            assert entry.coefficient != 0.0
            assert entry.token_id >= 0

    def test_tfidf_data_built(self):
        kv_gen = self._make_mock_kv_gen(hidden_dim=32)

        config = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            window_size=10,
        )

        store = streaming_prefill(kv_gen, list(range(20)), config)
        # Window tokens should be populated
        assert len(store.window_tokens) == 2
        assert store.window_tokens[0] == set(range(10))
        assert store.window_tokens[1] == set(range(10, 20))
        # IDF should be computed
        assert len(store.idf) > 0

    def test_progress_callback(self):
        kv_gen = self._make_mock_kv_gen(hidden_dim=16)

        config = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            window_size=5,
        )

        calls = []
        streaming_prefill(
            kv_gen,
            list(range(15)),
            config,
            progress_fn=lambda w, t: calls.append((w, t)),
        )
        assert len(calls) == 3
        assert calls[-1] == (2, 3)

    def test_empty_document(self):
        kv_gen = self._make_mock_kv_gen()
        config = ArchitectureConfig(retrieval_layer=29, query_head=4, injection_layer=30)
        store = streaming_prefill(kv_gen, [], config)
        assert store.num_windows == 0
        assert len(store.entries) == 0

    def test_chains_boundary_residual(self):
        """Second window should receive initial_residual from first."""
        kv_gen = self._make_mock_kv_gen(hidden_dim=16)

        config = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            window_size=5,
        )

        streaming_prefill(kv_gen, list(range(10)), config)

        calls = kv_gen.prefill_to_layer.call_args_list
        # First call: no initial_residual
        assert calls[0][1].get("initial_residual") is None
        # Second call: should have initial_residual
        assert calls[1][1].get("initial_residual") is not None


# ── KnowledgeStore edge cases ────────────────────────────────────────


class TestKnowledgeStoreEdgeCases:
    def test_route_keyword_returns_best_match(self):
        """Route should return the window with the most keyword hits."""
        entries = [
            InjectionEntry(0, 1.0, 0, 0, 0),
            InjectionEntry(1, 1.0, 1, 0, 1),
            InjectionEntry(2, 1.0, 2, 0, 2),
        ]
        keywords = {
            0: ["apollo", "mission"],
            1: ["apollo", "mission", "lunar"],
            2: ["porridge"],
        }
        config = ArchitectureConfig(retrieval_layer=29, query_head=4, injection_layer=30)
        store = KnowledgeStore(
            entries=entries,
            window_tokens={},
            window_token_lists={},
            idf={},
            keywords=keywords,
            config=config,
            num_windows=3,
        )
        # "apollo mission lunar" should match window 1 (3 hits) over 0 (2 hits)
        assert store.route("apollo mission lunar", method="keyword") == 1

    def test_route_no_match(self):
        entries = [InjectionEntry(0, 1.0, 0, 0, 0)]
        keywords = {0: ["hello"]}
        config = ArchitectureConfig(retrieval_layer=29, query_head=4, injection_layer=30)
        store = KnowledgeStore(
            entries=entries,
            window_tokens={},
            window_token_lists={},
            idf={},
            keywords=keywords,
            config=config,
            num_windows=1,
        )
        assert store.route("xyznonexistent", method="keyword") is None

    def test_save_load_preserves_values(self, tmp_path):
        """Verify loaded entry values match originals."""
        entries = [
            InjectionEntry(42, 4608.5, 0, 0, 0),
            InjectionEntry(99, 5120.25, 1, 1, 1),
        ]
        config = ArchitectureConfig(
            retrieval_layer=17,
            query_head=0,
            injection_layer=18,
            crystal_layer=18,
            window_size=256,
        )
        store = KnowledgeStore(
            entries=entries,
            window_tokens={0: {42, 100}, 1: {99, 200}},
            window_token_lists={0: [42, 100], 1: [99, 200]},
            idf={42: 1.5, 99: 2.0, 100: 0.5, 200: 0.5},
            keywords={0: ["test"], 1: ["demo"]},
            config=config,
            num_windows=2,
            num_tokens=512,
        )
        store.save(tmp_path / "store")

        loaded = KnowledgeStore.load(tmp_path / "store")
        assert loaded.config.retrieval_layer == 17
        assert loaded.config.crystal_layer == 18
        assert loaded.config.window_size == 256
        assert loaded.num_tokens == 512
        assert loaded.num_windows == 2
        assert len(loaded.entries) == 2
        # Values preserved
        assert loaded.entries[0].token_id == 42
        assert abs(loaded.entries[0].coefficient - 4608.5) < 1.0
        assert loaded.entries[1].token_id == 99
        # Window tokens preserved
        assert loaded.window_tokens[0] == {42, 100}
        assert loaded.window_tokens[1] == {99, 200}
        # IDF preserved
        assert abs(loaded.idf[42] - 1.5) < 0.01

    def test_single_window_document(self):
        """Document smaller than window_size should produce 1 window.
        With 1 window, all tokens have IDF=0 (log(1/1)), so no entries
        are selected (nothing is distinctive in a single window)."""
        kv_gen = MagicMock()
        embed_matrix = mx.random.normal((256, 8))
        mx.eval(embed_matrix)
        kv_gen.backbone = MagicMock()
        kv_gen.backbone.embed_matrix = embed_matrix
        kv_gen.prefill_to_layer.return_value = mx.random.normal((1, 3, 8))

        config = ArchitectureConfig(
            retrieval_layer=29,
            query_head=4,
            injection_layer=30,
            window_size=512,
            entries_per_window=2,
        )
        store = streaming_prefill(kv_gen, list(range(3)), config)
        assert store.num_windows == 1
        assert len(store.entries) == 0  # IDF=0 for all tokens in single window

    def test_extract_keywords_case_insensitive(self):
        kws = _extract_keywords_from_text("Apollo LUNAR Mission")
        assert "apollo" in kws
        assert "lunar" in kws
        assert "mission" in kws

    def test_extract_keywords_handles_punctuation(self):
        kws = _extract_keywords_from_text("Hello, world! Testing's great.")
        assert "hello" in kws
        assert "world" in kws
        assert "testing's" in kws or "testing" in kws

    def test_sparse_index_multiple_passages_same_keyword(self):
        idx = SparseKeywordIndex()
        idx.add(0, ["rocket"])
        idx.add(1, ["rocket"])
        idx.add(2, ["rocket"])

        results = idx.query("rocket launch")
        assert set(results) == {0, 1, 2}


# ── TFIDFRouter ──────────────────────────────────────────────────────


class TestTFIDFRouter:
    def test_basic_routing(self):
        from chuk_lazarus.inference.context.knowledge.route import TFIDFRouter

        window_tokens = {
            0: {10, 20, 30},  # generic tokens
            1: {40, 50, 60},  # different tokens
            2: {10, 40, 70},  # overlaps with both
        }
        idf = TFIDFRouter.compute_idf(window_tokens)
        router = TFIDFRouter(window_tokens, idf)

        # Query with token 50 (unique to window 1) should route there
        assert router.route([50]) == 1

        # Query with tokens common to multiple windows
        # Token 10 is in windows 0 and 2 — equal IDF
        result = router.route([10, 20])
        assert result == 0  # window 0 has both 10 and 20

    def test_idf_computation(self):
        from chuk_lazarus.inference.context.knowledge.route import TFIDFRouter

        window_tokens = {
            0: {1, 2, 3},
            1: {1, 4, 5},
            2: {1, 6, 7},
        }
        idf = TFIDFRouter.compute_idf(window_tokens)
        # Token 1 appears in all 3 windows: IDF = log(3/3) = 0
        assert abs(idf[1]) < 0.01
        # Token 2 appears in 1 window: IDF = log(3/1) = 1.098
        assert abs(idf[2] - 1.0986) < 0.01

    def test_empty_windows(self):
        from chuk_lazarus.inference.context.knowledge.route import TFIDFRouter

        router = TFIDFRouter({}, {})
        assert router.route([1, 2, 3]) is None

    def test_route_with_score(self):
        from chuk_lazarus.inference.context.knowledge.route import TFIDFRouter

        window_tokens = {0: {10}, 1: {20}}
        idf = {10: 1.0, 20: 2.0}
        router = TFIDFRouter(window_tokens, idf)

        wid, score = router.route_with_score([20])
        assert wid == 1
        assert abs(score - 2.0) < 0.01


# ── inject_1d ────────────────────────────────────────────────────────


class TestInject1D:
    def test_basic_injection(self):
        from chuk_lazarus.inference.context.knowledge.inject import inject_1d

        embed_matrix = mx.array(
            [
                [0.0, 0.0, 0.0],  # token 0
                [1.0, 0.0, 0.0],  # token 1 — unit x direction
                [0.0, 1.0, 0.0],  # token 2 — unit y direction
            ]
        )
        residual = mx.array([0.0, 0.0, 0.0])

        # Inject token 1 with coefficient 5.0
        result = inject_1d(residual, 1, 5.0, embed_matrix)
        mx.eval(result)
        # Should add 5.0 * [1,0,0] / ||[1,0,0]||^2 = [5, 0, 0]
        assert abs(float(result[0]) - 5.0) < 0.01
        assert abs(float(result[1])) < 0.01
        assert abs(float(result[2])) < 0.01

    def test_injection_is_additive(self):
        from chuk_lazarus.inference.context.knowledge.inject import inject_1d

        embed_matrix = mx.array([[0.0, 0.0], [3.0, 0.0]])
        residual = mx.array([10.0, 20.0])

        result = inject_1d(residual, 1, 6.0, embed_matrix)
        mx.eval(result)
        # direction = [3,0] / 9 = [1/3, 0]
        # result = [10, 20] + 6.0 * [1/3, 0] = [12, 20]
        assert abs(float(result[0]) - 12.0) < 0.01
        assert abs(float(result[1]) - 20.0) < 0.01
