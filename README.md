# 📊 Ciudad Juárez Municipal Dispatch & Spatial Logistics Analysis

## Project Overview
This repository contains an open-source geospatial analysis framework designed to process and handle large-scale municipal emergency dispatch tracking telemetry (2019–2026) from Ciudad Juárez, Chihuahua. 

The core objective of this project is to provide data scientists, transit planners, and urban logistics researchers with reproducible pipelines to analyze public safety resource allocation, monitor dispatch latency metrics, and lay the foundation for high-resolution patrol optimization modeling. 

To maintain strict adherence to data minimization principles, **all structural personal identifiers have been permanently removed at the database stream level**. The project utilizes an anonymized data model containing zero caller names, phone numbers, landmarks, or street address text strings.

---

## 📁 Repository Structure
```text
├── .gitignore                   # Prevents local data caches from public tracking
├── README.md                    # Core project documentation and logistics blueprint
├── requirements.txt             # Locked dependency matrix (Polars, GeoPandas, etc.)
├── pipeline_anonymize.py        # High-speed data-cleaning streaming script
└── verify_anonymization.py      # Independent compliance and schema audit script
```

---

## 📍 Spatial Allocation Framework (Future Milestones)
Planned expansions for this analysis suite intend to isolate operational dispatch patterns across the high-density urban core of Ciudad Juárez using localized coordinate projections. Future mapping modules will utilize specific parameters to ensure processing efficiency.

### Targeted Mapping Constraints
The upcoming visualization engine is designed to filter and clip spatial data points within the following targeted Web Mercator viewport limits:
* **Horizontal Limits (X):** `-1.1858e+07` to `-1.1837e+07`
* **Vertical Limits (Y):** `3.7020e+06` to `3.7190e+06`

### Grid Density Layering Objectives
Future plotting pipelines will aim to aggregate **approx. 7.8 million rows** into a high-resolution `hexbin` deployment layer using a structural grid size of **450 steps**. By implementing a reduced opacity alpha threshold of `0.4` and blending an underlying **Esri World Topographic Basemap (Forced Zoom 14)**, future research iterations will allow contributors to visually correlate call volume densities against municipal thoroughfares, local district boundaries (`SGR_DISTRITO`), and specific agency patrol sectors (`AEI_SECTOR`).

---

## 🛠️ Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd juarez-c4-dispatch-analysis
   ```

2. **Configure dependencies:**
   Ensure you have a Python environment configured, then execute the automated installation matrix:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔒 Anonymization & Verification Protocol

### 1. Data Processing Pipeline
The initial raw repository is cleaned via `pipeline_anonymize.py` using **Polars Lazy Frames**. This stream-level architecture passes only an approved whitelist of operational variables into the final `.geoparquet` file, completely discarding unapproved text objects before rows can hit system memory.

### 2. Independent Compliance Audit
Before sharing data assets or developing downstream visualization blocks, the file structure can be audited locally using the verification module:
```bash
python verify_anonymization.py
```

The validation layer executes a dual-layer sanity check:
* **Structural Check:** Scans table metadata headers to guarantee fields like `NOMBRE_DEL_RELATOR` or `TELÉFONO` do not exist.
* **Content Check:** Runs a high-entropy regular expression signature loop to verify that zero 10-digit numerical phone sequences or residual identity text blocks reside inside row values.

---

## 📊 Streamed Usage Example (Python)
Because the dataset is hosted in an optimized cloud format, you can leverage Polars lazy execution to stream and parse data coordinates over remote HTTP boundaries without manual local downloads:

```python
import polars as pl

# Reference your hosted anonymized asset path on Hugging Face
remote_url = "https://huggingface.co"

# Connect to cloud metadata stream
lazy_df = pl.scan_parquet(remote_url)

# Isolate high-impact spatial clusters during peak operational hours
optimized_slice = lazy_df.filter(
    (pl.col("ALTO_IMPACTO") == 1) & 
    (pl.col("HoraDiaEntero").is_between(20, 23))
).collect()

print(f"Isolated {len(optimized_slice)} high-impact baseline call clusters.")
```

---

## ⚖️ License & Open Data Use
This project is distributed under open data research parameters. The processed operational metadata attributes are intended solely for academic research, public infrastructure modeling, and urban resource scheduling optimizations.
