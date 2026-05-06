import mongomock


def test_record_email_persists_thread_metadata(make_test_memory, tmp_path):
    mem = make_test_memory

    mem.record_email(
        export_dir=tmp_path,
        date_str="2026-05-01",
        from_name="Marta Silva",
        from_addr="marta@example.com",
        to_name="Ben",
        to_addr="ben@example.com",
        subject="Order status",
        body="Can you send an update?",
        timestamp="2026-05-01T09:00:00",
        direction="inbound",
        embed_id="email-parent",
        day=1,
        thread_id="email-parent",
    )
    mem.record_email(
        export_dir=tmp_path,
        date_str="2026-05-01",
        from_name="Ben",
        from_addr="ben@example.com",
        to_name="Marta Silva",
        to_addr="marta@example.com",
        subject="Re: Order status",
        body="I will check.",
        timestamp="2026-05-01T09:12:00",
        direction="outbound",
        embed_id="email-child",
        day=1,
        reply_to_email_id="email-parent",
    )

    parent = mem._db["emails"].find_one({"embed_id": "email-parent"})
    child = mem._db["emails"].find_one({"embed_id": "email-child"})
    assert parent["thread_id"] == "email-parent"
    assert child["thread_id"] == "email-parent"
    assert child["thread_order"] == 1
    assert child["reply_to_email_id"] == "email-parent"
    assert child["eml_path"].endswith("ben_email-child.eml")


def test_validate_dataset_flags_missing_parent_and_actor_alias(tmp_path):
    from dataset_validation import validate_dataset

    db = mongomock.MongoClient().db
    db.sim_config.insert_one(
        {
            "_id": "inbound_email_sources",
            "sources": [{"name": "Marta Silva", "first_name": "Marta"}],
        }
    )
    db.emails.insert_one(
        {
            "embed_id": "child",
            "reply_to_email_id": "missing-parent",
            "timestamp": "2026-05-01T09:12:00",
            "date": "2026-05-01",
            "day": 1,
        }
    )
    db.events.insert_one({"type": "customer_email_routed", "actors": ["Marta"]})

    report = validate_dataset(db, export_dir=tmp_path)

    assert not report["ok"]
    assert any(item["kind"] == "missing_email_parent" for item in report["errors"])
    assert any(item["kind"] == "actor_alias_used" for item in report["warnings"])
