"""wav2vec2 reference backend — stub for check-backend-wiring.py.
The wav2vec2 backend shares the CTC decode path with parakeet/canary-ctc.
Per-stage diff testing uses the canary-ctc harness."""
DEFAULT_STAGES = ["encoder_output"]
def dump(model_dir, audio, stages, **kwargs):
    raise NotImplementedError("wav2vec2 ref dump: use crispasr-diff wav2vec2 directly")
