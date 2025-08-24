# UniqScan ML APIs Configuration

# API Server Settings
UNIFIED_API_PORT = 5000
SIMILARITY_API_PORT = 5001
AI_DETECTION_API_PORT = 5002

# Model Configuration
AI_DETECTION_MODEL = "SuperAnnotate/ai-detector"
MAX_CHUNK_SIZE = 512

# File Paths
UPLOAD_FOLDER = "submissions"
REPORTS_FOLDER = "reports"
SIMILARITY_DATABASE = "similarity_database.json"
AI_DETECTION_DATABASE = "ai_detection_database.json"

# Processing Settings
SIMILARITY_THRESHOLD = 3
SIMILARITY_CUTOFF = 5
NGRAM_SIZE = 3
MIN_DISTANCE = 8

# Scoring Weights (for unified report)
PLAGIARISM_WEIGHT = 0.6
AI_WEIGHT = 0.4

# Performance Settings
MAX_WORKERS = 3
PROCESSING_TIMEOUT = 300  # seconds

# Risk Thresholds
LOW_RISK_THRESHOLD = 20
MEDIUM_RISK_THRESHOLD = 40
HIGH_RISK_THRESHOLD = 70

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
