import torch
import uuid
from typing import List, Dict
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =====================
# CONFIG
# =====================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS = 256
BATCH_SIZE = 4   # safe for A100, increase to 8 if needed
TEMPERATURE = 0.2

# =====================
# LOAD MODELS (ONCE)
# =====================
print("Loading embedding model...")
embedder = SentenceTransformer(
    EMBEDDING_MODEL_ID,
    device=DEVICE
)

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# =====================
# UTILS
# =====================
def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def generate_knowledge(prompts: List[str]) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True
        )

    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )
    return decoded


# =====================
# MAIN FUNCTION
# =====================
def read_pdf(pdf_path: str) -> List[Dict]:
    reader = PdfReader(pdf_path)

    all_chunks = []
    metadata = []

    print("Reading and chunking PDF...")
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or len(text) < 20:
            continue

        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append(page_idx + 1)

    print(f"Total chunks: {len(all_chunks)}")

    print("Computing embeddings...")
    embeddings = embedder.encode(
        all_chunks,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    results = []

    print("Running LLM inference...")
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE)):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]

        prompts = [
            f"""
You are a knowledge extractor for a rental agreement.
Extract factual, objective information only.

TEXT:
{chunk}
""".strip()
            for chunk in batch_chunks
        ]

        generations = generate_knowledge(prompts)

        for chunk, knowledge, emb, page in zip(
            batch_chunks,
            generations,
            embeddings[i:i + BATCH_SIZE],
            metadata[i:i + BATCH_SIZE]
        ):
            results.append({
                "chunk_id": str(uuid.uuid4()),
                "embedding": emb,
                "knowledge": knowledge,
                "chunk_text": chunk,
                "page": page
            })

    return results
