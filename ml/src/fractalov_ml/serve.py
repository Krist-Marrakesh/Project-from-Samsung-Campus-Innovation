"""FastAPI inference server.

Run via the CLI::

    uv run fractalov-ml serve --checkpoint runs/best/best.pt --port 9000

Endpoints:

* ``GET  /healthz``                — liveness, returns checkpoint metadata
* ``POST /infer/suggest-from-image`` — multipart PNG upload, returns Suggestion
* ``POST /infer/variations``       — body: existing recipe + count, returns
  list of perturbed recipes (rule-based, no model)

The server is a deliberate single-process design: it loads the checkpoint
once at startup and reuses it across requests. There is no per-request lazy
init and no model swapping — restart the process to load a new checkpoint.
For the Stage 7 wiring this is sufficient; multi-checkpoint serving belongs
in a later stage if it ever becomes useful.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .inference import InferenceService, Suggestion, variations_around


log = logging.getLogger("fractalov_ml.serve")


class SuggestionResponse(BaseModel):
    family: str
    family_confidence: float
    family_distribution: dict[str, float]
    c_re: Optional[float] = None
    c_im: Optional[float] = None
    recipe: dict


class VariationsRequest(BaseModel):
    recipe: dict
    count: int = Field(default=4, ge=1, le=16)
    seed: int = Field(default=42)


class VariationsResponse(BaseModel):
    recipes: list[dict]


class HealthResponse(BaseModel):
    status: str
    checkpoint: str
    device: str
    base_channels: int


def _suggestion_to_response(s: Suggestion) -> SuggestionResponse:
    payload = asdict(s)
    return SuggestionResponse(**payload)


def build_app(service: InferenceService) -> FastAPI:
    """Construct the FastAPI application bound to a pre-loaded service.

    Tests use this directly with a stub ``InferenceService``; the CLI calls
    it with the real one. Keeping the binding explicit avoids the global-state
    pattern most FastAPI examples use.
    """
    app = FastAPI(title="fractalov-ml inference", version="0.1.0")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            checkpoint=str(service.checkpoint),
            device=str(service.device),
            base_channels=service.base_channels,
        )

    @app.post("/infer/suggest-from-image", response_model=SuggestionResponse)
    async def suggest_from_image(
        file: UploadFile = File(...),
        target_width_px: Optional[int] = None,
        target_height_px: Optional[int] = None,
    ) -> SuggestionResponse:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=415,
                detail=f"unsupported content-type: {file.content_type}",
            )
        try:
            data = await file.read()
        finally:
            await file.close()
        try:
            suggestion = service.predict_recipe(
                data,
                target_width_px=target_width_px,
                target_height_px=target_height_px,
            )
        except Exception as ex:  # noqa: BLE001 — surface anything as 400
            log.exception("inference failure")
            raise HTTPException(status_code=400, detail=f"inference failed: {ex}")
        return _suggestion_to_response(suggestion)

    @app.post("/infer/variations", response_model=VariationsResponse)
    def variations(req: VariationsRequest) -> VariationsResponse:
        try:
            recipes = variations_around(req.recipe, req.count, req.seed)
        except KeyError as ex:
            # Caller sent a recipe missing a required field.
            raise HTTPException(status_code=400, detail=f"malformed recipe: missing {ex}")
        return VariationsResponse(recipes=recipes)

    return app


def run(
    checkpoint: Path,
    host: str = "0.0.0.0",
    port: int = 9000,
    device_override: Optional[str] = None,
) -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    service = InferenceService(checkpoint=checkpoint, device_override=device_override)
    app = build_app(service)
    uvicorn.run(app, host=host, port=port, log_level="info")
