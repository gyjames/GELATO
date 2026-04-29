# `annot_genome.py` — Genome annotation pipeline

This script chains **Prokka → KofamScan (HMM) → DIAMOND (protein cluster DB)**. Each gene is mapped to **KEGG Orthology (KO) when available, otherwise a DIAMOND cluster ID, otherwise `<oov>`**, and written as a **single-line, tab-separated token file** suitable for GELATO training or inference.

---

## Pipeline overview

| Step | Tool | Role |
|------|------|------|
| 1 | **Prokka** | Gene prediction and protein sequences (`.faa`) |
| 2 | **KofamScan** (`exec_annotation`) | HMM-based KO annotation |
| 3 | **DIAMOND** `blastp` | Search against a custom cluster library to fill gaps |
| 4 | Integration | Per-gene order: `KO` or `ClusterID` or `<oov>` → one line in a text file named `{sample_name}` (stem of the input genome; no extra extension) |

If intermediate files already exist (e.g. `prokka_*`, `*_kofam.tsv`, `*_vs_cdb.tsv`), the corresponding steps are **skipped** so you can resume safely.

---

## Dependencies

- **Python 3** with `pandas`
- Executables on `PATH` or passed explicitly:
  - `prokka`
  - **KofamScan wrapper**: default `exec_annotation` (override with `--kofam_exec`)
  - `diamond`
- **Data**:
  - `--ko_list` — KO list for KofamScan
  - `--profiles` — Kofam profile directory (can be copied to RAM with `--mem_profiles`)
  - `--cdb` — DIAMOND database (`.dmnd`)

---

## Inputs and outputs

### Inputs

- `-i` / `--input_dir` — Directory of genomes to annotate  
- `-e` / `--ext` — Genome file extension (default: `fna`, i.e. `*.fna`)

### Outputs (under `--output_dir`)

| Path | Contents |
|------|----------|
| `results/` | Final per-sample files named by genome **stem** (e.g. `GCA_000123.1`): **one line**, **tab-separated** tokens as above. |
| `intermediate/` | Prokka output dirs, KofamScan TSV, DIAMOND hits, etc. |

KofamScan scratch space is rooted at `--tmp_dir` (default `/dev/shm/tmp`); per-sample subdirs are created and removed when possible after each run.

---

## Command-line reference

| Argument | Description |
|----------|-------------|
| `-i`, `--input_dir` | Input directory (**required**) |
| `-o`, `--output_dir` | Output root (**required**) |
| `-e`, `--ext` | Genome extension (default: `fna`) |
| `-t`, `--threads` | Threads **per genome** for Prokka / KofamScan / DIAMOND (default: `8`) |
| `--ko_list` | Path to KO list file |
| `--profiles` | Kofam profile directory |
| `--kofam_exec` | KofamScan executable name or path (default: `exec_annotation`) |
| `--tmp_dir` | Base temp dir for KofamScan (default: `/dev/shm/tmp`) |
| `--cdb` | DIAMOND database `.dmnd` (**required**) |
| `--mem_cdb` | Copy DIAMOND DB to `/dev/shm` before use (helps on slow shared storage) |
| `--mem_profiles` | Copy profile directory to `/dev/shm` |
| `--keep_mem_db` | Do not remove DB/profiles from memory on exit (default: clean up via `atexit`) |

Example:

```bash
python annot_genome.py \
  -i /path/to/genomes \
  -o /path/to/out \
  -e fna \
  -t 8 \
  --ko_list /path/to/ko_list \
  --profiles /path/to/profiles \
  --cdb /path/to/cluster.dmnd \
  --mem_cdb --mem_profiles
```

---

## Large-scale runs: **many parallel jobs, not one job using all CPUs**

Inside the script, genomes under `-i` are processed **sequentially** in a `for` loop; `-t` applies to **one genome at a time** for Prokka / HMM / DIAMOND.

For **large scale genomes**, prefer:

1. **Split work across independent parallel jobs**  
   Partition genomes into **multiple subdirectories** (or one job per genome / small batch in a job array), and run **several** `annot_genome.py` instances at once, each with a **moderate** `-t` (e.g. 4–16 per job, tuned to RAM and I/O).  
   Total **wall time** usually improves more from **N concurrent jobs × modest threads** than from a **single** run that tries to use every core on the box.

2. **Use `--mem_cdb` / `--mem_profiles` where helpful**  
   On multi-node clusters, **per-node** copies into local RAM or fast local disk reduce many processes hammering the **same** remote library paths.


---
