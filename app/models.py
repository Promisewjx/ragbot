from app import settings
from cross_encoder import CrossEncoderReranker

reranker = None
if settings.RERANKER_ENABLE:
    reranker = CrossEncoderReranker(
        model_name=settings.RERANKER_MODEL,
        device=settings.RERANKER_DEVICE,
        batch_size=settings.RERANKER_BATCH,
        max_length=settings.RERANKER_MAXLEN,
    )
