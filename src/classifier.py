from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, List, Optional, Tuple

from PIL import Image

from src.config import ProductLabel


@dataclass
class ClassificationResult:
    product: ProductLabel
    score: float


@lru_cache(maxsize=2)
def _load_clip_model(model_name: str, device: str) -> Tuple[Any, Any]:
    """Load (and cache) a single CLIP model/processor pair per model name.

    Multiple ClipProductClassifier instances (e.g. "beverage" and "shampoo")
    reuse the same underlying model weights instead of each loading their own
    copy, which roughly halves memory usage on constrained hosts (e.g.
    Streamlit Community Cloud's 1 GB RAM tier).
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


class ClipProductClassifier:
    def __init__(
        self,
        model_name: str,
        candidate_labels: Iterable[ProductLabel],
        negative_prompts: Optional[Iterable[str]] = None,
        reject_margin: float = 0.0,
    ):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "Torch/Transformers dependencies are unavailable in this runtime."
            ) from exc

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = _load_clip_model(model_name, self.device)
        self.labels: List[ProductLabel] = list(candidate_labels)
        self.prompts = [label.prompt for label in self.labels]
        self.negative_prompts: List[str] = list(negative_prompts or [])
        self.reject_margin = float(reject_margin)

    def classify(self, crop_image: Image.Image) -> ClassificationResult:
        with self._torch.no_grad():
            inputs = self.processor(
                text=self.prompts,
                images=crop_image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)[0]
            best_idx = int(self._torch.argmax(probs).item())

        return ClassificationResult(
            product=self.labels[best_idx],
            score=float(probs[best_idx].item()),
        )

    def classify_or_none(self, crop_image: Image.Image) -> Optional[ClassificationResult]:
        """Classify a crop and optionally reject obvious non-product regions."""
        if not self.negative_prompts:
            return self.classify(crop_image)

        combined_prompts = self.prompts + self.negative_prompts
        with self._torch.no_grad():
            inputs = self.processor(
                text=combined_prompts,
                images=crop_image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        product_count = len(self.labels)
        product_probs = probs[:product_count]
        negative_probs = probs[product_count:]

        best_product_idx = int(self._torch.argmax(product_probs).item())
        best_product_prob = float(product_probs[best_product_idx].item())
        best_negative_prob = float(self._torch.max(negative_probs).item()) if len(negative_probs) else 0.0

        if best_negative_prob >= (best_product_prob + self.reject_margin):
            return None

        return ClassificationResult(
            product=self.labels[best_product_idx],
            score=best_product_prob,
        )
