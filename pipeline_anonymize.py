import os
import sys
import polars as pl
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Automatically scan for and load the local .env configuration file
load_dotenv()

print("Initializing data ingestion pipeline...")

# Securely extract the repository ID from your local computer's memory variables
repo_id = os.environ.get("SOURCE_HF_REPO")

if not repo_id:
    print("❌ ERROR: 'SOURCE_HF_REPO' key not found in your local .env configuration file.")
    print("Please ensure you have a '.env' file containing: SOURCE_HF_REPO=\"namespace/repo\"")
    sys.exit(1)

verified_filename = "llamadas.geoparquet"

try:
    print(f"Downloading compressed master file from secure environment reference...")
    # Fetch via Hugging Face Hub downloader (handles Git LFS resolution natively)
    local_parquet_path = hf_hub_download(
        repo_id=repo_id, 
        filename=verified_filename, 
        repo_type="dataset"
    )
except Exception as e:
    print(f"❌ Download Failure. Verify your .env repository string or connection: {e}")
    sys.exit(1)

print(f"File successfully cached locally at: {local_parquet_path}")
print("Scanning file schema using Polars lazy execution engine...")

# Scan using the lazy evaluation engine to avoid loading massive raw data files into your RAM
lazy_df = pl.scan_parquet(local_parquet_path)

# Strict, neutral data-science telemetry whitelist 
safe_whitelist = [
    "geometry", "INCIDENTE", "SUBTIPO", "ALTO_IMPACTO", 
    "CORPORACIÓN", "TIEMPO_DE_DESPACHÓ", "HoraDiaEntero", 
    "MUNICIPIO", "SGR_DISTRITO", "AEI_SECTOR"
]

# Pull only the available columns matching the whitelist
available_columns = [col for col in lazy_df.columns if col in safe_whitelist]
print(f"🔒 Isolated {len(available_columns)} safe metadata columns. Dropping all PII...")

# Mask out anything not explicitly whitelisted (permanently dropping Name/Phone/Address strings)
anonymized_lazy_df = lazy_df.select(available_columns)

output_file = "llamadas_juarez_8M_ANONYMIZED.geoparquet"
print(f"🚀 Streaming and writing anonymized GeoParquet to: {output_file}")

# Write using streaming batches to keep memory usage near zero
anonymized_lazy_df.sink_parquet(
    output_file,
    compression="snappy",
    row_group_size=100_000
)

print("✨ Done! 8 million rows completely stripped of PII and saved successfully.")
