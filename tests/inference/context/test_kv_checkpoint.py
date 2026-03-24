"""Tests for KVCheckpoint — save/load/is_resumable."""

from chuk_lazarus.inference.context.kv_checkpoint import (
    CheckpointMeta,
    ContextCheckpointStatus,
    KVCheckpoint,
)


def _make_meta(**overrides) -> CheckpointMeta:
    defaults = {
        "model_id": "test-model",
        "seq_len": 128,
        "total_tokens": 512,
        "num_layers": 4,
        "num_kv_heads": 2,
        "head_dim": 64,
        "chunk_size": 128,
        "source_hash": "abc123",
        "status": ContextCheckpointStatus.PARTIAL,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:01:00Z",
    }
    defaults.update(overrides)
    return CheckpointMeta(**defaults)


class TestCheckpointMeta:
    def test_chunks_done(self):
        meta = _make_meta(seq_len=256, chunk_size=128)
        assert meta.chunks_done == 2

    def test_chunks_total(self):
        meta = _make_meta(total_tokens=512, chunk_size=128)
        assert meta.chunks_total == 4

    def test_is_complete_false(self):
        assert not _make_meta(status=ContextCheckpointStatus.PARTIAL).is_complete

    def test_is_complete_true(self):
        assert _make_meta(status=ContextCheckpointStatus.COMPLETE).is_complete


class TestKVCheckpointLoadMeta:
    def test_load_meta_missing_returns_none(self, tmp_path):
        assert KVCheckpoint.load_meta(tmp_path / "nonexistent") is None

    def test_load_meta_roundtrip(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        meta = _make_meta()
        (ckpt_dir / "meta.json").write_text(meta.model_dump_json())
        loaded = KVCheckpoint.load_meta(ckpt_dir)
        assert loaded is not None
        assert loaded.model_id == meta.model_id
        assert loaded.seq_len == meta.seq_len
        assert loaded.source_hash == meta.source_hash


class TestKVCheckpointIsResumable:
    def test_not_resumable_no_checkpoint(self, tmp_path):
        assert not KVCheckpoint.is_resumable(tmp_path / "empty", "hash123")

    def test_not_resumable_wrong_hash(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        meta = _make_meta(source_hash="correct_hash", status=ContextCheckpointStatus.PARTIAL)
        (ckpt_dir / "meta.json").write_text(meta.model_dump_json())
        assert not KVCheckpoint.is_resumable(ckpt_dir, "wrong_hash")

    def test_not_resumable_complete(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        meta = _make_meta(source_hash="hash123", status=ContextCheckpointStatus.COMPLETE)
        (ckpt_dir / "meta.json").write_text(meta.model_dump_json())
        assert not KVCheckpoint.is_resumable(ckpt_dir, "hash123")

    def test_resumable(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        meta = _make_meta(source_hash="hash123", status=ContextCheckpointStatus.PARTIAL)
        (ckpt_dir / "meta.json").write_text(meta.model_dump_json())
        assert KVCheckpoint.is_resumable(ckpt_dir, "hash123")


class TestKVCheckpointSourceHash:
    def test_deterministic(self):
        data = b"hello world"
        assert KVCheckpoint.source_hash(data) == KVCheckpoint.source_hash(data)

    def test_different_inputs(self):
        assert KVCheckpoint.source_hash(b"foo") != KVCheckpoint.source_hash(b"bar")
