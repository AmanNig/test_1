from utils.pdf_reader import TarotPDFEmbedder
from initialize.config import VECTOR_DB_DIR, MODEL_NAME

_embedder = TarotPDFEmbedder(model_name="all-MiniLM-L6-v2", collection_name="tarot_cards")

def get_card_meaning(card_name: str, k: int = 3) -> str:
    results = _embedder.retrieve(card_name, top_k=k)
    return "\n\n".join(results)
