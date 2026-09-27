"""
pipeline_anonymize_hardened.py
--------------------------------------------------------------------
Hardened anonymization pipeline for the Ciudad Juárez C4 dispatch
dataset, rewritten against the actual ArcGIS field schema (not a
guessed one).

READ THIS FIRST: anonymization is not the whole problem
========================================================
This script hardens the DATA (what gets published). It does not
and cannot fix ACCESS CONTROL (who can query the live system). Those
are separate problems, and for an operational C4/CAD dataset the
second one is the more urgent one:

  * Authenticate every call, CAD, AVL, and incident layer. An
    "unlisted" REST endpoint is not authenticated. If this dataset
    is being pulled from a live ArcGIS Server/Portal instance,
    confirm the source layers require auth — this script can't do
    that for you, and no anonymization pipeline downstream fixes an
    open upstream service.
  * Caller PII (TELÉFONO, NOMBRE_DEL_RELATOR, USUARIO) belongs in a
    restricted table with its own access policy, not as a droppable
    "attribute" in an otherwise-public feature layer. Dropping the
    column here is necessary but is a data-hygiene step, not an
    access-control policy.
  * Public research/community layers must not share a server,
    workspace, or connection pool with operational C4 layers
    (live CAD, AVL, active incidents). This script only ever reads
    a static, already-exported file — it has no way to enforce that
    separation, so that separation has to exist upstream of this
    ever running.
  * Audit Create/Update/Delete capability on anything emergency-
    related. A writable dispatch layer, even one nobody reads,
    means somebody can invent a call or move a unit. Not this
    script's job either, but worth stating plainly: if you can run
    this pipeline against a feature service, check whether an
    unauthenticated actor could also POST to it.
  * Responder routing, sector assignment, and response-time data are
    operational security, not GIS attributes, in a context like
    Juárez. This script treats every *_DISTRITO / *_SECTOR field
    (there are 13 agency-specific pairs in the real schema — SPM,
    PV, PF, SGR, CRUM, CROJA, HCB, AEI, CRISIS, C4, SEDENA, UEPC,
    RESCATE, CES) as sensitive-by-default and drops all but one
    coarse district/sector pair from public output (see
    PUBLISHABLE_DISTRICT / PUBLISHABLE_SECTOR below) — but if this
    data is genuinely at operational-security risk, the right
    control is not shipping any of it to a public/scannable
    endpoint at all, regardless of what this script strips.
  * Assume continuous scanners exist. If a "safe" export answers
    anonymously over HTTP, someone will eventually crawl it and
    correlate it against news reports, social media, or other public
    C4/CAD leaks. This script reduces what's in the file; it can't
    make an openly-reachable endpoint un-findable.

None of the code below substitutes for those five points. Treat this
as data minimization for something that has already been cleared,
by someone with the authority to make that call, for external release.

What the actual schema contains (verified against the ArcGIS field
list, not assumed)
====================================================================
DIRECT IDENTIFIERS — hard-dropped, never whitelisted, never
recoverable from output no matter what other options are passed:
    TELÉFONO, NOMBRE_DEL_RELATOR, USUARIO, FOLIO, FOLIO_2DA_LLAMADA,
    REGISTRO, TARJETA_DE_SEGURIDAD, CÓDIGO (confirmed: internal
    case/dispatch reference number, same re-identification role as
    FOLIO)

LOCATION QUASI-IDENTIFIERS (free text / fine address detail) —
hard-dropped:
    CALLE, CALLE_ESQUINA, COLONIA, CODIGO_POSTAL, NÚMERO_EXTERIOR,
    NÚMERO_INTERIOR, REFERENCIA, LOCALIDAD, CRUCE_FRONTERIZO,
    KILÓMETRO, ORIGEN, DESTINO, HOSPITAL_DESTINO,
    INSTITUCIÓN_RECEPCIÓN

EXACT COORDINATES — generalized to a grid, never published raw:
    COORDENADA_X, COORDENADA_Y (esriFieldTypeInteger — a projected
    coordinate system, not lat/lon degrees; grid size assumes meters
    and should be checked against the actual SRS before use)

FINE-GRAINED CALL-LIFECYCLE TIMESTAMPS — dropped in favor of the
pre-existing coarse fields (HoraDiaEntero, DiaSemana, PeriodoDia,
Dia/Mes/Ano):
    FECHA_Y_HORA_INICIO_LLAMADA(_1), FECHA_Y_HORA_ENVIO_A_DESPACHO,
    FECHA_Y_HORA_FIN_LLAMADA, FECHA_Y_HORA_DESPACHÓ,
    FECHA_Y_HORA_ACUDIÓ, FECHA_Y_HORA_CIERRE, and every
    _INICIO_/_FIN_ pair for medical instruction, medical advisory,
    medical regulation, and transport — a dozen-plus fields that,
    paired with coordinates, are a strong correlation-attack vector
    on their own even with names/phones removed.

OPERATIONAL / SENSITIVE-BY-DEFAULT — only one coarse district/sector
pair is retained (configurable), the other 12 agencies' worth are
dropped:
    SPM_, PV_, PF_, SGR_, CRUM_, CROJA_, HCB_, AEI_, CRISIS_, C4_,
    SEDENA_, UEPC_, RESCATE_, CES_ (each has _DISTRITO and _SECTOR)
    CORPORACIÓN, CODIGO_NARANJA, ESTATUS, PRIORIDAD

RETAINED AFTER HARDENING (subject to further k-anonymity
generalization/suppression below):
    INCIDENTE, SUBTIPO, Precategoria_Incidente,
    Subcategoria_Incidente, ALTO_IMPACTO, MUNICIPIO, DiaSemana,
    PeriodoDia, HoraDiaEntero, Dia, Mes, Ano, one *_DISTRITO /
    *_SECTOR pair, gridded x/y

This is a stronger baseline, not a certification. Whether ANY of
this should be published — even hardened — for an active
operational dataset in a context with a documented pattern of
targeting responders is a policy decision for people with the
authority and local context to make it, not something this script
can decide by running.
"""

import os
import sys
from typing import Optional

import polars as pl
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# Tunable privacy parameters
# ----------------------------------------------------------------------

K_ANONYMITY_THRESHOLD = 5

# COORDENADA_X/Y are esriFieldTypeInteger in a projected CRS per the
# schema. CONFIRM THE ACTUAL SRS/UNITS before trusting this grid size —
# if the source is in a geographic CRS (degrees) rather than a
# projected one (meters), this constant needs to change to a degree
# fraction, not stay 150.
SPATIAL_GRID_METERS = 150
SPATIAL_CELL_MIN_COUNT = 5

# Exactly one agency's district/sector pair is retained for
# geographic-analysis usefulness. Every other agency's routing/sector
# data is dropped by default because it is operational, not a plain
# map attribute (see module docstring). Change this only with an
# explicit decision that a *different* single agency pair is the
# minimum needed — never widen it to "keep several."
PUBLISHABLE_DISTRICT_COL = "SGR_DISTRITO"
PUBLISHABLE_SECTOR_COL = "AEI_SECTOR"

# ----------------------------------------------------------------------
# Column classification, built directly from the ArcGIS field list
# ----------------------------------------------------------------------

DIRECT_IDENTIFIERS = [
    "TELÉFONO", "NOMBRE_DEL_RELATOR", "USUARIO", "FOLIO",
    "FOLIO_2DA_LLAMADA", "REGISTRO", "TARJETA_DE_SEGURIDAD",
    # Confirmed: an internal case/dispatch reference number, same
    # role as FOLIO — lets a specific record be looked up or re-linked
    # even with every other identifier stripped. Originally left in
    # RETAINED_BASE_COLS on a guess; moved here once confirmed.
    "CÓDIGO",
]

# GÉNERO_RELATOR added: a personal attribute of the caller/reporter.
# Not a direct identifier on its own, but it's PII about a specific
# person with no analytical role in incident-pattern research, so it
# is dropped alongside the other identifiers rather than retained.
PERSONAL_ATTRIBUTE_COLS = [
    "GÉNERO_RELATOR",
]

LOCATION_QUASI_IDENTIFIERS = [
    "CALLE", "CALLE_ESQUINA", "COLONIA", "CODIGO_POSTAL",
    "NÚMERO_EXTERIOR", "NÚMERO_INTERIOR", "REFERENCIA", "LOCALIDAD",
    "CRUCE_FRONTERIZO", "KILÓMETRO", "ORIGEN", "DESTINO",
    "HOSPITAL_DESTINO", "INSTITUCIÓN_RECEPCIÓN",
]

# Raw point geometry. Confirmed against a live schema dump:
#   Shape    -> float64   (not a geometry type; likely a leftover
#                          scalar — shape length/area — from the
#                          Esri export, not itself a coordinate pair,
#                          but dropped anyway since it serves no
#                          purpose without the geometry it describes
#                          and it isn't a vetted, safe field)
#   geometry -> unknown   (loads with a "geoarrow.wkb is not
#                          registered" warning — this IS full-
#                          precision point geometry in a WKB
#                          extension type Polars doesn't recognize
#                          natively. COORDENADA_X/Y get gridded by
#                          generalize_geometry(), but this raw
#                          geometry column bypasses that entirely if
#                          it isn't also dropped here.)
RAW_GEOMETRY_COLS = [
    "Shape", "geometry",
]

FINE_TIMESTAMP_COLS = [
    "FECHA_Y_HORA_INICIO_LLAMADA", "FECHA_Y_HORA_INICIO_LLAMADA_1",
    "FECHA_Y_HORA_ENVIO_A_DESPACHO", "TIEMPO_DE_ENVÍO_A_DESPACHO",
    "FECHA_Y_HORA_FIN_LLAMADA", "TIEMPO_LLAMADA",
    "FECHA_Y_HORA_INICIO_INSTRUCCIÓN", "FECHA_Y_HORA_FIN_INSTRUCCIÓN_MÉ",
    "TIEMPO_INSTRUCCIÓN_MÉDICA", "NÚMERO_INSTRUCCIONES_MÉDICAS",
    "PROTOCOLO_INSTRUCCIÓN_MÉDICA",
    "FECHA_Y_HORA_DESPACHÓ", "TIEMPO_DE_DESPACHÓ",
    "FECHA_Y_HORA_INICIO_ASESORÍA_MÉ", "FECHA_Y_HORA_FIN_ASESORÍA_MÉDIC",
    "TIEMPO_ASESORÍA_MÉDICA", "NÚMERO_ASESORÍAS_MÉDICAS",
    "FECHA_Y_HORA_INICIO_REGULACIÓN_", "FECHA_Y_HORA_FIN_REGULACIÓN_MÉD",
    "TIEMPO_REGULACIÓN_MÉDICA", "NÚMERO_REGULACÓNES_MÉDICAS",
    "FECHA_Y_HORA_ACUDIÓ",
    "FECHA_Y_HORA_INICIO_TRASLADO", "FECHA_Y_HORA_FIN_TRASLADO",
    "TIEMPO_TRASLADO", "NÚMERO_TRASLADOS",
    "FECHA_Y_HORA_CIERRE",
]

# All 14 agency prefixes that carry a _DISTRITO/_SECTOR pair in the
# real schema. PUBLISHABLE_DISTRICT_COL/PUBLISHABLE_SECTOR_COL above
# are the one exception kept out of this drop list.
AGENCY_PREFIXES = [
    "SPM", "PV", "PF", "SGR", "CRUM", "CROJA", "HCB", "AEI",
    "CRISIS", "C4", "SEDENA", "UEPC", "RESCATE", "CES",
]

OTHER_OPERATIONAL_COLS = [
    "CORPORACIÓN", "CODIGO_NARANJA", "ESTATUS", "PRIORIDAD",
    "SIMULACRO", "AGREGO_INFORMACIÓN", "CLASIFICACIÓN_LLAMADA_IMPROCEDE",
    "TOTAL_2DA_LLAMADA",
    # Confirmed via live schema dump — same family as the *_DISTRITO/
    # *_SECTOR fields: dispatch subcenter and "corrected"/canonical
    # district-quadrant-sector fields. Treated as sensitive-by-default
    # for the same reason those are (see module docstring).
    "SUBCENTRO", "DistritoCorrecto", "CuadranteCorrecto",
    "SectorCorrecto", "Ubicacion",
]

COORD_COLS = ["COORDENADA_X", "COORDENADA_Y"]

# Columns that are safe, coarse, and useful to keep as-is.
#
# TIPO: kept here as a working assumption, NOT a confirmed one.
# Structurally it sits beside SUBTIPO/INCIDENTE and looks like a
# parent incident-type category (large_string), the same kind of
# thing as INCIDENTE, not an identifier. Confirm what it actually
# represents before trusting this classification. (CÓDIGO, which sat
# next to it, was checked and turned out to be an internal
# case/dispatch reference number — see DIRECT_IDENTIFIERS above —
# so the same kind of surprise is possible here too.)
RETAINED_BASE_COLS = [
    "INCIDENTE", "SUBTIPO", "TIPO", "Precategoria_Incidente",
    "Subcategoria_Incidente", "ALTO_IMPACTO", "MUNICIPIO",
    "DiaSemana", "DiaSemanaNum", "PeriodoDia", "HoraDiaEntero",
    "Dia", "Mes", "Ano",
]

INPUT_FILENAME = "llamadas.geoparquet"
OUTPUT_FILENAME = "llamadas_juarez_HARDENED_ANONYMIZED.geoparquet"


def _agency_sensitive_cols(schema_cols: list) -> list:
    """Every *_DISTRITO / *_SECTOR field for the 14 agency prefixes,
    minus whichever single pair is explicitly kept public."""
    keep = {PUBLISHABLE_DISTRICT_COL, PUBLISHABLE_SECTOR_COL}
    drop = []
    for prefix in AGENCY_PREFIXES:
        for suffix in ("_DISTRITO", "_SECTOR"):
            col = f"{prefix}{suffix}"
            if col in schema_cols and col not in keep:
                drop.append(col)
    return drop


def fetch_source(repo_id: str) -> str:
    print("Downloading compressed master file from secure environment reference...")
    return hf_hub_download(repo_id=repo_id, filename=INPUT_FILENAME, repo_type="dataset")


def drop_identifiers_and_operational_fields(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Single hard-drop pass covering: direct identifiers, address-level
    quasi-identifiers, fine timestamps, and sensitive-by-default
    operational routing fields. Unlike the original script's
    whitelist-by-selection approach, this explicitly names what is
    being removed and why, so a schema change that adds a new PII
    column doesn't silently sail through an unchanged whitelist.
    """
    schema_cols = lazy_df.collect_schema().names()

    agency_drop = _agency_sensitive_cols(schema_cols)

    all_drop_candidates = (
        DIRECT_IDENTIFIERS
        + PERSONAL_ATTRIBUTE_COLS
        + LOCATION_QUASI_IDENTIFIERS
        + RAW_GEOMETRY_COLS
        + FINE_TIMESTAMP_COLS
        + agency_drop
        + OTHER_OPERATIONAL_COLS
    )
    present_to_drop = [c for c in all_drop_candidates if c in schema_cols]

    print(f"Hard-dropping {len(present_to_drop)} identifying/operational columns:")
    print(f"  direct identifiers:      {[c for c in DIRECT_IDENTIFIERS if c in schema_cols]}")
    print(f"  personal attributes:     {[c for c in PERSONAL_ATTRIBUTE_COLS if c in schema_cols]}")
    print(f"  location quasi-ids:      {[c for c in LOCATION_QUASI_IDENTIFIERS if c in schema_cols]}")
    print(f"  raw geometry:            {[c for c in RAW_GEOMETRY_COLS if c in schema_cols]}")
    print(f"  fine timestamps:         {len([c for c in FINE_TIMESTAMP_COLS if c in schema_cols])} fields")
    print(f"  other-agency dist/sector:{agency_drop}")
    print(f"  other operational:       {[c for c in OTHER_OPERATIONAL_COLS if c in schema_cols]}")

    lazy_df = lazy_df.drop(present_to_drop)

    # Explicitly report what's left, rather than trusting a whitelist
    # to have anticipated every column — anything not accounted for
    # above and not in RETAINED_BASE_COLS/COORD_COLS/publishable pair
    # is unexpected and should be reviewed by a person before release.
    remaining = list(lazy_df.collect_schema().names())
    expected_remaining = set(
        RETAINED_BASE_COLS + COORD_COLS
        + [PUBLISHABLE_DISTRICT_COL, PUBLISHABLE_SECTOR_COL]
    )
    unexpected = [c for c in remaining if c not in expected_remaining and c != "OBJECTID"]
    if unexpected:
        print(
            f"⚠️  UNRECOGNIZED COLUMNS SURVIVED THE DROP LIST: {unexpected}\n"
            "   These are not in this script's classification and have NOT been "
            "vetted as safe. Add them to the appropriate drop list above and "
            "re-run, or explicitly confirm they belong in RETAINED_BASE_COLS."
        )

    return lazy_df


def generalize_geometry(lazy_df: pl.LazyFrame, grid_size: int) -> pl.LazyFrame:
    """
    Snap COORDENADA_X/COORDENADA_Y to a coarse grid and drop the exact
    originals. Confirmed via live schema dump: both are float64 (the
    original module docstring assumed esriFieldTypeInteger from the
    static field list — the live parquet actually stores them as
    doubles, which doesn't change the grid math but did need
    correcting here). NOTE: confirm the source SRS/units before
    trusting `grid_size` as meters — if these floats are in a
    geographic CRS (degrees) this needs to be reworked as a
    degree-fraction grid, not a flat meter offset.
    """
    schema_cols = lazy_df.collect_schema().names()
    if "COORDENADA_X" not in schema_cols or "COORDENADA_Y" not in schema_cols:
        print("⚠️  COORDENADA_X/COORDENADA_Y not found — spatial generalization skipped.")
        return lazy_df

    print(f"Snapping COORDENADA_X/Y to a grid of size {grid_size} (verify units first)...")
    return lazy_df.with_columns([
        (pl.col("COORDENADA_X") / grid_size).floor().mul(grid_size).cast(pl.Int64).alias("x_grid"),
        (pl.col("COORDENADA_Y") / grid_size).floor().mul(grid_size).cast(pl.Int64).alias("y_grid"),
    ]).drop(["COORDENADA_X", "COORDENADA_Y"])


def enforce_k_anonymity(lazy_df: pl.LazyFrame, k: int) -> pl.LazyFrame:
    """
    For each (district, sector, SUBTIPO, HoraDiaEntero) group, require
    at least k rows. Below-threshold rows get SUBTIPO generalized up
    to INCIDENTE; if still below k, the row is suppressed. Uses the
    single publishable district/sector pair, since the others were
    already dropped.

    Eagerly collects for the group-by pass — a single aggregation over
    ~8M rows is tractable in memory even though it isn't pure streaming.
    """
    group_cols = [PUBLISHABLE_DISTRICT_COL, PUBLISHABLE_SECTOR_COL, "SUBTIPO", "HoraDiaEntero"]
    required = set(group_cols) | {"INCIDENTE"}
    schema_cols = set(lazy_df.collect_schema().names())
    missing = required - schema_cols
    if missing:
        print(f"⚠️  Columns missing for k-anonymity check: {missing} — skipping this step.")
        return lazy_df

    print(f"Evaluating k-anonymity (k={k}) over {group_cols}...")
    df = lazy_df.collect()

    counts = df.group_by(group_cols).agg(pl.len().alias("_group_size"))
    df = df.join(counts, on=group_cols, how="left")

    n_below = (df["_group_size"] < k).sum()
    print(f"  {n_below:,} rows fall below k={k} at full granularity; generalizing SUBTIPO for these.")

    df = df.with_columns(
        pl.when(pl.col("_group_size") < k)
        .then(pl.col("INCIDENTE"))
        .otherwise(pl.col("SUBTIPO"))
        .alias("SUBTIPO")
    )

    counts_2 = df.group_by(group_cols).agg(pl.len().alias("_group_size_2"))
    df = df.drop("_group_size").join(counts_2, on=group_cols, how="left")

    n_suppressed = (df["_group_size_2"] < k).sum()
    if n_suppressed:
        print(f"  {n_suppressed:,} rows still below k={k} after generalization — suppressing these rows.")
    df = df.filter(pl.col("_group_size_2") >= k).drop("_group_size_2")

    return df.lazy()


def suppress_sparse_spatial_cells(lazy_df: pl.LazyFrame, min_count: int) -> pl.LazyFrame:
    schema_cols = lazy_df.collect_schema().names()
    if "x_grid" not in schema_cols or "y_grid" not in schema_cols:
        print("⚠️  No gridded coordinates found — spatial cell suppression skipped.")
        return lazy_df

    print(f"Suppressing spatial grid cells with fewer than {min_count} records...")
    df = lazy_df.collect()
    cell_counts = df.group_by(["x_grid", "y_grid"]).agg(pl.len().alias("_cell_size"))
    df = df.join(cell_counts, on=["x_grid", "y_grid"], how="left")
    n_suppressed = (df["_cell_size"] < min_count).sum()
    print(f"  Suppressing {n_suppressed:,} rows in sparse (< {min_count}-record) grid cells.")
    df = df.filter(pl.col("_cell_size") >= min_count).drop("_cell_size")
    return df.lazy()


def run_pipeline(repo_id: Optional[str] = None) -> str:
    load_dotenv()
    repo_id = repo_id or os.environ.get("SOURCE_HF_REPO")
    if not repo_id:
        print("❌ ERROR: 'SOURCE_HF_REPO' not found in environment or .env file.")
        sys.exit(1)

    try:
        local_path = fetch_source(repo_id)
    except Exception as e:
        print(f"❌ Download failure: {e}")
        sys.exit(1)

    print(f"File cached locally at: {local_path}")
    lazy_df = pl.scan_parquet(local_path)

    lazy_df = drop_identifiers_and_operational_fields(lazy_df)
    lazy_df = generalize_geometry(lazy_df, SPATIAL_GRID_METERS)
    lazy_df = enforce_k_anonymity(lazy_df, K_ANONYMITY_THRESHOLD)
    lazy_df = suppress_sparse_spatial_cells(lazy_df, SPATIAL_CELL_MIN_COUNT)

    print(f"🚀 Writing hardened anonymized output to: {OUTPUT_FILENAME}")
    lazy_df.sink_parquet(OUTPUT_FILENAME, compression="snappy", row_group_size=100_000)
    print("✨ Done. Remember: this hardens the FILE. It does not authenticate")
    print("   the upstream service, separate public from operational layers,")
    print("   or audit write access. See the module docstring.")
    return OUTPUT_FILENAME


if __name__ == "__main__":
    run_pipeline()