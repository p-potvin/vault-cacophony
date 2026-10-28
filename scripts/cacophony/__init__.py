"""Cacophony Multi-Speaker Conversation & Acoustic Simulation Framework.

Combines:
- HF Inference LLM (DeepInfra / serverless) as dialogue brain
- Local Qwen3-TTS / Mini-Omni for foreground persona synthesis
- FastMSS HMM turn-taking & background rambling speech pool
- Pyroomacoustics 3D room simulation & software mixer (zero virtual cables)
"""

__version__ = "0.1.0"
