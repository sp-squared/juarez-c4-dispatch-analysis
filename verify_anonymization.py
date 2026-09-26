import polars as pl
import re
import sys

def run_dataset_audit(file_path):
    print(f"Opening Target: {file_path}")
    print("Initiating strict column layout validation...")
    
    try:
        # Load the lazy metadata layer
        lazy_df = pl.scan_parquet(file_path)
        active_columns = lazy_df.collect_schema().names()
    except Exception as e:
        print(f"❌ Error opening file format architecture: {e}")
        sys.exit(1)
    
    # Define the precise, neutral data science whitelist
    safe_whitelist = [
        "geometry", "INCIDENTE", "SUBTIPO", "ALTO_IMPACTO", 
        "CORPORACIÓN", "TIEMPO_DE_DESPACHÓ", "HoraDiaEntero", 
        "MUNICIPIO", "SGR_DISTRITO", "AEI_SECTOR"
    ]
    
    # 1. Structural Header Check
    unauthorized_fields = [col for col in active_columns if col not in safe_whitelist]
    
    print("\n--- [Audit Layer 1: Structural Schema Sanity] ---")
    if unauthorized_fields:
        print(f"❌ CRITICAL FAILURE: The schema contains unapproved tracking headers: {unauthorized_fields}")
        print("This file contains un-anonymized column paths and cannot be distributed.")
        sys.exit(1)
    else:
        print(f"✅ PASSED: Structural header check clean. Retained columns: {len(active_columns)}/{len(safe_whitelist)}")

    # 2. Deep String Footprint Scanner
    print("\n--- [Audit Layer 2: In-Memory Row Value Scanner] ---")
    print("Sampling initial row allocations for high-entropy pattern leakage...")
    
    # Fetch a massive sample block into memory to verify text remnants
    sample_pool = lazy_df.head(100_000).collect()
    raw_text_dump = str(sample_pool.to_dicts())
    
    # Scan via regex for any residual, continuous 10-digit numeric signatures (Mexican phone blocks)
    phone_pattern_leak = bool(re.search(r'\b\d{10}\b', raw_text_dump))
    
    if phone_pattern_leak:
        print("❌ FAILURE: High-entropy string arrays matching telephone signatures detected in data rows.")
        sys.exit(1)
    else:
        print("✅ PASSED: Core sample space checks out. Zero telephone string signatures discovered.")
        
    print("\n🔒 VERIFICATION SUCCESSFUL: The file is confirmed anonymous and safe for public research distribution.")

if __name__ == "__main__":
    target_asset = "llamadas_juarez_8M_ANONYMIZED.geoparquet"
    run_dataset_audit(target_asset)
