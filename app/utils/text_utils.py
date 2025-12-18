import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("text_utils")

def normalize_text(text: str) -> str:
    """Normaliza texto removendo quebras de linha excessivas e espaços."""
    try:
        text = re.sub(r'\n{2,}', '\n\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'(\d+)\s+\.', r'\1.', text)
        text = text.replace(' .', '.').replace(' ,', ',')
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    except Exception:
        logger.exception("Error normalizing text")
        return text