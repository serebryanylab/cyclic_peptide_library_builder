# Cyclic Peptide Library Builder

A browser-based tool for partitioning large pools of cyclic peptide sequences into non-overlapping libraries that are free of mass-spectrometry ambiguity and biochemical redundancy. Built with [stlite](https://github.com/whitphx/stlite), it runs entirely client-side via Pyodide.

*Serebryany Lab · Stony Brook University*

## Problem Statement

When screening combinatorial cyclic peptide libraries by mass spectrometry, two peptides cause issues in two ways:

- **Mass collision** — if their cyclic or linear molecular weights fall within the instrument's resolution window, they are practically indistinguishable by MS.
- **Sequence redundancy** — if their sequences are too similar by BLOSUM62 substitution score, they are biochemically redundant.

A good library avoids both. This tool builds one (or several) such libraries automatically.

## How it works

The tool formulates library partitioning as a **Maximum Independent Set** problem on a pairwise *conflict graph*:

- Each peptide is a node.
- Two peptides share a conflict edge if they are too close in mass **or** too similar in sequence (BLOSUM62 distance below the threshold).

A fast greedy solver runs over multiple random seeds to maximize the number of peptides retained per library, returning a set of internally conflict-free, non-overlapping libraries ready for synthesis and MS-based screening. Increasing the number of seeds improves the chance of finding a good split at the cost of runtime.

## Inputs

Provide one peptide sequence per line, using standard one-letter amino acid codes (`ACDEFGHIKLMNPQRSTVWY`). Two entry modes are supported:

- **Upload CSV / TXT** — upload a file with one sequence per row. For CSVs you choose which column holds the sequences.
- **Paste sequences** — paste sequences directly, one per line.

Sequences containing non-standard amino acids or empty entries are flagged before the run begins.

## Parameters

All parameters live in the sidebar under *Conflict Graph Parameters*:

- **Number of libraries** (1–8) — how many non-overlapping libraries to construct.
- **Sequence distance threshold** (0.0–1.0) — BLOSUM62 distance below which two peptides conflict. Set to `0` to skip sequence-based filtering entirely (this also disables the dendrogram, which needs the distance matrix).
- **Mass threshold (Da)** (0.1–20.0) — cyclic/linear MW difference below which two peptides are treated as indistinguishable by MS.
- **Seeds to evaluate** (1–200) — more seeds give a better chance of an optimal split but run slower.

Changing any parameter clears previous results so a stale split is never shown.

## Running a selection

1. Load or paste your sequences.
2. Set the parameters in the sidebar.
3. Click **Run Library Selection**.

A live status panel reports progress while `PeptideSelection` initializes, computes the conflict graph, and constructs the libraries.

## Outputs and visualizations

After a run completes, results persist across interface reruns and include:

- **Summary metrics** — input count, total selected, not selected, overall yield, and per-library size with its share of the input.
- **Cyclic MW distribution** — overlaid histogram by library (with unselected peptides shown for context).
- **Sequence length distribution** — grouped bar chart of residue lengths per library.
- **Sequence diversity dendrogram** — a UPGMA dendrogram on BLOSUM62 distances, with leaves colored by library. Capped at 400 leaves; larger pools are shown via a stratified sample. You can optionally include unselected peptides. Requires a sequence threshold greater than 0.
- **Per-library tables** — sortable sequence tables with cyclic and linear MW, plus per-library MW statistics. An *Unselected* tab lists any peptides not placed in a library.
- **Conflict graph** — a network view of the pairwise conflicts (automatically hidden for inputs larger than 400 peptides).
- **Run log** — captured stdout/stderr from the selection run for inspection.

The visualization toggles in the sidebar control which panels are displayed.

## Downloads

- Per-library CSVs (sequence, cyclic MW, linear MW).
- Combined CSV across all libraries, with a `library` column.
- Unselected-peptides CSV.
- Conflict-graph PNG.
- A multi-page **PDF report** containing the run parameters, summary and MW statistics, MW and length distributions, the dendrogram (when available), and per-library sequence tables.
