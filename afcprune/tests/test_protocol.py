"""Test-set guard (plan P4)."""

import json

import pytest

from afcprune import protocol


def fake(protocol_name):
    d = dict(train="TR", val="VA", test="TE", pg="PG")
    if protocol_name == "paper":
        d["val"] = d["test"]
    return protocol.get_splits(protocol_name, _loaded=d)


def test_clean_blocks_test():
    s = fake("clean")
    assert (s.train, s.val, s.pg) == ("TR", "VA", "PG")
    with pytest.raises(protocol.TestAccessError):
        s.test


def test_allow_test_logs(tmp_path):
    s = fake("clean")
    with protocol.allow_test("unit test", log_dir=tmp_path):
        assert s.test == "TE"
    recs = [json.loads(l) for l in (tmp_path / "test_access.jsonl").read_text().splitlines()]
    assert len(recs) == 1 and recs[0]["reason"] == "unit test" and recs[0]["split"] == "test"
    with pytest.raises(protocol.TestAccessError):
        s.test                                     # guard closes again


def test_paper_protocol_val_is_guarded(tmp_path):
    s = fake("paper")
    with pytest.raises(protocol.TestAccessError):
        s.val
    with protocol.allow_test("reproduction row", log_dir=tmp_path):
        assert s.val == "TE"


def test_reason_required():
    with pytest.raises(ValueError):
        with protocol.allow_test(""):
            pass
