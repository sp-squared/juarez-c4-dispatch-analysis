import os
from dotenv import load_dotenv

# 1. Look for the hidden .env file in the active folder
load_dotenv()

# 2. Extract the value securely from local memory variables
repo_id = os.environ.get("SOURCE_HF_REPO")

print("--- Environmental Variable Sanity Test ---")
if repo_id == "https://huggingface.co/":
    print("✅ SUCCESS: The environment engine read your .env file cleanly.")
    print(f"🔒 Target Identifier mapped to: '{repo_id}'")
else:
    print("❌ FAILURE: Environment string missing. Verify file is named exactly '.env'.")
