from functools import cached_property
from typing import Literal, Protocol

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
Kind = Literal["passage", "query"]


class Embedder(Protocol):
    model_id: str
    revision: str

    def encode(self, texts: list[str], *, kind: Kind) -> np.ndarray: ...


class E5Embedder:
    model_id = MODEL_ID
    revision = MODEL_REVISION

    @cached_property
    def model(self):
        import torch
        from sentence_transformers import SentenceTransformer

        # Avoid running small batches across all available CPU cores by default.
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        return SentenceTransformer(
            self.model_id,
            revision=self.revision,
            device="cpu",
            model_kwargs={"use_safetensors": True},
        )

    def encode(self, texts: list[str], *, kind: Kind) -> np.ndarray:
        return self.model.encode(
            texts,
            prompt=f"{kind}: ",
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
