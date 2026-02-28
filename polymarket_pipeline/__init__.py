"""Baseline Polymarket ingestion and analysis pipeline."""

__all__ = ["PipelineRunner"]


def __getattr__(name: str):
    if name == "PipelineRunner":
        from .pipeline import PipelineRunner

        return PipelineRunner
    raise AttributeError(name)
