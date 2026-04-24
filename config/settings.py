from dotenv import load_dotenv
import os

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4"  # or "gpt-3.5-turbo"

# Agent Configuration
MAX_ENGAGEMENT_TURNS = 5
MATCH_THRESHOLD = 0.6
INTEREST_THRESHOLD = 0.5

# Data paths
DATA_DIR = "data"
CANDIDATES_FILE = f"{DATA_DIR}/candidates.json"
JD_FILE = f"{DATA_DIR}/job_descriptions.json"
OUTPUT_FILE = f"{DATA_DIR}/output.json"