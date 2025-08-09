import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = "https://qdrant-vector-db-k3fy.onrender.com"
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY ')
GROQ_API_KEY = os.getenv('GROQ_API_KEY ')

MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD", "your_password_here")
MONGODB_URI = f"mongodb+srv://shashwatlinked:{MONGODB_PASSWORD}@cluster0.jggmqx6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "notebook_lm"
EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50
MAX_CHUNK_SIZE = 1200
OVERLAP_SENTENCES = 2
TOP_K_RESULTS = 12

PDF_OCR_THRESHOLD = 100
TEXT_QUALITY_THRESHOLD = 70

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_CHAT_MODEL = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS = 2000
GROQ_FALLBACK_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

UPLOAD_FOLDER = "uploads"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

MAX_IMAGES_PER_PAGE = 10
OCR_BATCH_SIZE = 5
PROCESSING_TIMEOUT = 300
RATE_LIMIT_DELAY = 0.1

ENABLE_PROCESSING_STATS = True
LOG_PROCESSING_DETAILS = True
SAVE_PROCESSING_METADATA = True

ENABLE_GROQ_OCR = True
ENABLE_FALLBACK_PROCESSING = True
ENABLE_CHUNK_OPTIMIZATION = True
ENABLE_DUPLICATE_REMOVAL = True

MIN_OCR_CONFIDENCE = 0.7
MAX_SPECIAL_CHAR_RATIO = 0.3
MIN_WORDS_PER_CHUNK = 10

MAX_RETRIES = 3
RETRY_DELAY = 1
ENABLE_GRACEFUL_DEGRADATION = True

DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
PRODUCTION_MODE = os.getenv("RAILWAY_ENVIRONMENT") == "production"
TEST_MODE = os.getenv("TEST_MODE", "False").lower() == "true"
MOCK_GROQ_API = os.getenv("MOCK_GROQ_API", "False").lower() == "true"

ESTIMATED_COST_PER_IMAGE = 0.0001
MAX_COST_PER_DOCUMENT = 1.0
WARN_HIGH_COST_THRESHOLD = 0.5

ENABLE_STRUCTURED_OCR = True
ENABLE_CONTEXT_AWARE_OCR = True
ENABLE_MULTI_LANGUAGE_SUPPORT = True
PRESERVE_FORMATTING = True

CHUNK_PROCESSING_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 32
VECTOR_SEARCH_CACHE_SIZE = 1000

MAX_PDF_PAGES = 500
MAX_FILE_SIZE_MB = 16
ALLOWED_FILE_TYPES = ['.pdf']
SCAN_FOR_MALWARE = False

LLAMA4_SCOUT_CONFIG = {
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "temperature": 0.1,
    "max_tokens": 2000,
    "context_window": 128000,
    "cost_per_input_token": 0.00000011,
    "cost_per_output_token": 0.00000034,
}

LLAMA4_MAVERICK_CONFIG = {
    "model": "meta-llama/llama-4-maverick-17b-128e-instruct", 
    "temperature": 0.1,
    "max_tokens": 2000,
    "context_window": 128000,
    "cost_per_input_token": 0.0000005,
    "cost_per_output_token": 0.00000077,
}

if DEBUG_MODE:
    LOG_PROCESSING_DETAILS = True
    ENABLE_PROCESSING_STATS = True
    MAX_RETRIES = 1
    PROCESSING_TIMEOUT = 60

if TEST_MODE:
    CHUNK_SIZE = 400
    MAX_CHUNK_SIZE = 600
    TOP_K_RESULTS = 5
    MAX_PDF_PAGES = 10

def validate_config():
    required_env_vars = ['GROQ_API_KEY']
    missing_vars = [var for var in required_env_vars if not globals().get(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be positive")
    
    if MIN_CHUNK_SIZE >= CHUNK_SIZE:
        raise ValueError("MIN_CHUNK_SIZE must be less than CHUNK_SIZE")
    
    if MAX_CHUNK_SIZE <= CHUNK_SIZE:
        raise ValueError("MAX_CHUNK_SIZE must be greater than CHUNK_SIZE")
    
    print("Configuration validation passed")

if __name__ == "__main__":
    validate_config()
else:
    try:
        validate_config()
    except Exception as e:
        print(f"Configuration warning: {e}")

PROCESSING_CONFIG = {
    "chunk_size": CHUNK_SIZE,
    "max_chunk_size": MAX_CHUNK_SIZE,
    "min_chunk_size": MIN_CHUNK_SIZE,
    "overlap_sentences": OVERLAP_SENTENCES,
    "ocr_threshold": PDF_OCR_THRESHOLD,
    "quality_threshold": TEXT_QUALITY_THRESHOLD,
    "groq_model": GROQ_VISION_MODEL,
    "enable_ocr": ENABLE_GROQ_OCR,
    "enable_optimization": ENABLE_CHUNK_OPTIMIZATION
}

API_CONFIG = {
    "groq_api_key": GROQ_API_KEY,
    "qdrant_url": QDRANT_URL,
    "qdrant_api_key": QDRANT_API_KEY,
    "mongodb_uri": MONGODB_URI,
    "max_retries": MAX_RETRIES,
    "retry_delay": RETRY_DELAY,
    "rate_limit_delay": RATE_LIMIT_DELAY
}