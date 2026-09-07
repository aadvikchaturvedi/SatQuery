# SatQuery AI Backend

Agentic remote-sensing vision-language backend for ISRO/SAC problem statement
26167. Wraps the existing, unmodified ML code in `vqa_and_change_using_gemini/`
(change-detection checkpoint + Gemini VQA) behind a FastAPI service with an
agentic task controller, upload validation, auditable execution reports, and
auth/rate limiting.

## Setup

```bash
cd SatQuery
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

cp backend/.env.example backend/.env
# edit backend/.env: set GEMINI_API_KEY at minimum
```

## Run

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: `http://127.0.0.1:8000/docs`

In development (`ENVIRONMENT=development`, the default), requests without
an `X-API-Key` header are allowed. Set `BACKEND_API_KEY` (generate one with
`python scripts/create_api_key.py`) to require auth, or before deploying —
it's required outright once `ENVIRONMENT=production`.

## Test

```bash
cd backend
pytest -v
```

The integration test runs the real trained checkpoint against the sample
Sentinel-2 tiles in `vqa_and_change_using_gemini/imgs_1` /`imgs_2`; no
network access or `GEMINI_API_KEY` is required for the test suite to pass.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "X-API-Key: $BACKEND_API_KEY" \
  -F "query=What changed between these two dates, and where?" \
  -F "optical_t1_files=@../vqa_and_change_using_gemini/imgs_1/...B02.tif" \
  ... (all 13 bands for t1) \
  -F "optical_t2_files=@../vqa_and_change_using_gemini/imgs_2/...B02.tif" \
  ... (all 13 bands for t2)
```

Field names (`optical_t1_files` / `optical_t2_files` / `sar_t1_files` /
`sar_t2_files`) determine which task the agentic controller selects — see
`app/agent/classifier.py`. A single `optical_t1_files`/`sar_t1_files` with
no T2 runs single-image VQA or captioning; both optical and SAR T1 with no
T2 runs cross-modal fusion; any T1+T2 pair runs change-VQA (using fusion
mode automatically if both optical and SAR pairs are given).

## Known gaps (see the engineering report for full detail)

- Single-image VQA/captioning/fusion currently call Gemini generically —
  the remote-sensing-fine-tuned model (`VQA/inference.py`, Qwen2.5-VL +
  LoRA) isn't wired in yet because its trained adapter weights don't exist
  in this repo. Swap point: `VQA_BACKEND=qwen` in `app/tools/vqa_tool.py`.
- Text-guided region grounding (the PS's other optional single-image task)
  isn't implemented; captioning was chosen instead.
