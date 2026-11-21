ADMIN = {"X-Admin-Key": "test-key"}


def upload(client, pdf):
    with open(pdf, "rb") as f:
        return client.post(
            "/api/documents",
            headers=ADMIN,
            files={"file": ("sop.pdf", f, "application/pdf")},
            data={"title": "Sample SOP", "version": "1.0"},
        )


def test_admin_endpoints_need_key(client):
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"X-Admin-Key": "wrong"}).status_code == 401


def test_rejects_non_pdf(client):
    r = client.post("/api/documents", headers=ADMIN, files={"file": ("a.txt", b"hi")}, data={"title": "x"})
    assert r.status_code == 400


def test_upload_chat_feedback_delete(client, sample_pdf):
    r = upload(client, sample_pdf)
    assert r.status_code == 202
    doc_id = r.json()["doc_id"]
    # TestClient runs background tasks before returning
    assert client.get(f"/api/documents/{doc_id}", headers=ADMIN).json()["status"] == "indexed"

    chat = client.post("/api/chat", json={"question": "When can ramp work resume after a lightning alert?"}).json()
    assert chat["grounded"] and chat["citations"][0]["page"] == 4

    assert client.post("/api/feedback", json={"query_id": chat["query_id"], "rating": 1}).status_code == 200
    assert client.post("/api/feedback", json={"query_id": "nope", "rating": 1}).status_code == 404
    assert client.get("/api/metrics", headers=ADMIN).json()["thumbs_up"] == 1

    assert client.delete(f"/api/documents/{doc_id}", headers=ADMIN).status_code == 200
    assert client.get("/health").json()["chunks"] == 0


def test_corrupt_pdf_marks_failed(client, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    doc_id = upload(client, bad).json()["doc_id"]
    doc = client.get(f"/api/documents/{doc_id}", headers=ADMIN).json()
    assert doc["status"] == "failed" and doc["error"]
