"""
VQA_BACKEND=qwen wiring, exercised through the real API (Gemini path is
already covered indirectly by the default-backend tests). The actual
Qwen2.5-VL-3B + LoRA model is heavy (multi-GB download, needs a real
accelerator to be fast) and out of scope for CI, so `run_qwen_vqa` is
monkeypatched at the boundary the same way `_ask_gemini_about_change` is
mocked in test_analyze_change_integration.py — everything above that
boundary (classifier, controller, tool, response shaping) runs for real.
"""


def test_single_image_vqa_uses_qwen_backend_when_configured(client, monkeypatch, sample_t1_files):
    from app.config import Settings

    settings_with_qwen = Settings(vqa_backend="qwen")
    monkeypatch.setattr("app.tools.vqa_tool.get_settings", lambda: settings_with_qwen)
    monkeypatch.setattr(
        "app.tools.vqa_tool.run_qwen_vqa",
        lambda image, question: {
            "answer": "A cropland area with a river running through it.",
            "model": "Qwen2.5-VL-3B-Instruct + SatQuery LoRA",
            "device": "cpu",
            "inference_time_seconds": 0.01,
        },
    )

    files = [("optical_t1_files", (name, content, "image/tiff")) for name, content in sample_t1_files.items()]

    resp = client.post(
        "/api/v1/analyze",
        data={"query": "What land cover is visible in this image?"},
        files=files,
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["task"] == "single_image_vqa"
    assert body["answer"] == "A cropland area with a river running through it."
    assert body["execution_summary"]["parameters"]["backend"] == "qwen"
    assert body["execution_summary"]["parameters"]["device"] == "cpu"
    assert body["execution_summary"]["warnings"] == []


def test_single_image_vqa_tool_reports_domain_adapted_only_for_qwen():
    from app.config import Settings
    from app.tools.vqa_tool import SingleImageVqaTool

    tool = SingleImageVqaTool()

    import app.tools.vqa_tool as vqa_tool_module

    original_get_settings = vqa_tool_module.get_settings
    try:
        vqa_tool_module.get_settings = lambda: Settings(vqa_backend="gemini")
        assert tool.domain_adapted is False

        vqa_tool_module.get_settings = lambda: Settings(vqa_backend="qwen")
        assert tool.domain_adapted is True
    finally:
        vqa_tool_module.get_settings = original_get_settings
