from pydantic import BaseModel

class FusionWeights(BaseModel):
    val_audio: float = 0.4
    aro_audio: float = 0.7

class EmotionResponse(BaseModel):
    valence: float
    arousal: float
    mood: str
    audio: dict
    lyrics: dict
