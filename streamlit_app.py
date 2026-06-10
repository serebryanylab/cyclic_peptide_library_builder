"""
Cyclic Peptide Library Builder — stlite (browser) edition
No server required: runs entirely inside the browser via Pyodide / stlite.
"""
from __future__ import annotations

import base64
import io
import textwrap
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from peptide_selection import PeptideSelection
    TOOL_AVAILABLE = True
    IMPORT_ERROR: str | None = None
except Exception as _e:
    TOOL_AVAILABLE = False
    IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

# ── Logo ──────────────────────────────────────────────────────────────────────
_logo_b64: str | None = None
try:
    with open("peptide_logo.svg", "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
except Exception:
    pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyclic Peptide Library Builder",
    page_icon="🧬",
    layout="wide",
)

if _logo_b64:
    st.markdown(
        f'<link rel="icon" type="image/svg+xml" '
        f'href="data:image/svg+xml;base64,{_logo_b64}">',
        unsafe_allow_html=True,
    )

# Palette shared across all plots (up to 8 libraries)
LIB_PALETTE   = ["#2ecc71", "#00bcd4", "#f300cf", "#f39c12",
                  "#1abc9c", "#e91e63", "#e74c3c", "#8e44ad"]
UNSELECTED_COLOR = "#7f8c8d"


# ── PDF report builder ────────────────────────────────────────────────────────

def _build_pdf_report(
    libraries: dict,
    total_input: int,
    total_sel: int,
    all_seqs: list,
    cyc_mw_all: np.ndarray,
    lin_mw_all: np.ndarray,
    idx_to_lib: dict,
    params: dict,
    ps=None,
) -> bytes:
    from datetime import date as _date
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()

    light = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "black",       "axes.labelcolor": "black",
        "xtick.color": "black",      "ytick.color": "black",
        "axes.edgecolor": "#cccccc", "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
    }

    def _style_table(tbl, header_color: str) -> None:
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if row == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")

    with plt.rc_context(light):
        with PdfPages(buf) as pdf:
            d = pdf.infodict()
            d["Title"]  = "Cyclic Peptide Library Builder Report"
            d["Author"] = "Serebryany Lab · Stony Brook University"

            # ── Page 1: Title + Parameters + Summary ─────────────────────────
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, "Cyclic Peptide Library Builder",
                     ha="center", fontsize=22, fontweight="bold", color="#1a1a2e")
            fig.text(0.5, 0.91, "Serebryany Lab · Stony Brook University",
                     ha="center", fontsize=13, color="#444")
            fig.text(0.5, 0.88, f"Generated: {_date.today().strftime('%B %d, %Y')}",
                     ha="center", fontsize=10, color="#777")

            # Parameters
            fig.text(0.08, 0.84, "Run Parameters", fontsize=13, fontweight="bold")
            ax_p = fig.add_axes([0.08, 0.66, 0.84, 0.16])
            ax_p.axis("off")
            _style_table(ax_p.table(
                cellText=[
                    ["Number of libraries",
                     str(params["n_lib"])],
                    ["Sequence distance threshold (BLOSUM62)",
                     f"{params['seq_threshold']:.2f}"],
                    ["Mass threshold (Da)",
                     f"{params['mw_threshold']:.1f}"],
                    ["Seeds evaluated",
                     str(params["n_seeds"])],
                ],
                colLabels=["Parameter", "Value"],
                cellLoc="left", loc="center",
                colWidths=[0.70, 0.30],
            ), header_color="#2563eb")

            # Summary
            fig.text(0.08, 0.63, "Summary", fontsize=13, fontweight="bold")
            summary_rows = [
                ["Input peptides",  f"{total_input:,}"],
                ["Total selected",  f"{total_sel:,}"],
                ["Not selected",    f"{total_input - total_sel:,}"],
                ["Overall yield",   f"{100 * total_sel / total_input:.1f}%"],
            ]
            for li, ldf in libraries.items():
                summary_rows.append([
                    f"Library {li}",
                    f"{len(ldf):,} peptides  ({100 * len(ldf) / total_input:.1f}%)",
                ])
            ax_s = fig.add_axes([0.08, 0.63 - 0.045 * len(summary_rows) - 0.02,
                                  0.84, 0.045 * len(summary_rows)])
            ax_s.axis("off")
            _style_table(ax_s.table(
                cellText=summary_rows,
                colLabels=["Metric", "Value"],
                cellLoc="left", loc="center",
                colWidths=[0.70, 0.30],
            ), header_color="#2563eb")

            # Per-library MW stats
            mw_rows = []
            for li, ldf in libraries.items():
                c = ldf["cyclic_mw"]
                mw_rows.append([
                    f"Library {li}",
                    f"{c.min():.3f}", f"{c.median():.3f}", f"{c.max():.3f}",
                    str(c.nunique()),
                ])
            y_mw = 0.63 - 0.045 * len(summary_rows) - 0.10
            fig.text(0.08, y_mw + 0.04, "Cyclic MW Statistics",
                     fontsize=13, fontweight="bold")
            ax_mw = fig.add_axes([0.08, y_mw - 0.045 * len(mw_rows),
                                   0.84, 0.045 * len(mw_rows)])
            ax_mw.axis("off")
            _style_table(ax_mw.table(
                cellText=mw_rows,
                colLabels=["Library", "Min (Da)", "Median (Da)", "Max (Da)", "Unique MWs"],
                cellLoc="center", loc="center",
                colWidths=[0.20, 0.20, 0.20, 0.20, 0.20],
            ), header_color="#2563eb")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 2: MW Distribution ───────────────────────────────────────
            fig, ax = plt.subplots(figsize=(8.5, 5))
            bins = np.linspace(cyc_mw_all.min(), cyc_mw_all.max(), 61)
            unsel_mask = np.array(
                [idx_to_lib.get(i, -1) == -1 for i in range(len(all_seqs))]
            )
            if unsel_mask.any():
                ax.hist(cyc_mw_all[unsel_mask], bins=bins, alpha=0.35,
                        color=UNSELECTED_COLOR,
                        label=f"Not selected (n={int(unsel_mask.sum()):,})",
                        linewidth=0)
            for li, ldf in libraries.items():
                ax.hist(ldf["cyclic_mw"].to_numpy(), bins=bins, alpha=0.65,
                        color=LIB_PALETTE[li % len(LIB_PALETTE)],
                        label=f"Library {li} (n={len(ldf):,})", linewidth=0)
            ax.set_xlabel("Cyclic MW (Da)")
            ax.set_ylabel("Count")
            ax.set_title("Cyclic MW Distribution by Library")
            ax.legend(framealpha=0.8)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 3: Sequence Length Distribution ──────────────────────────
            lengths_all = np.array([len(s) for s in all_seqs])
            len_min, len_max = int(lengths_all.min()), int(lengths_all.max())
            all_lengths = list(range(len_min, len_max + 1))
            n_libs = len(libraries)
            bar_w = 0.8 / max(n_libs, 1)

            fig, ax = plt.subplots(figsize=(8.5, 5))
            for li, ldf in libraries.items():
                lens = [len(s) for s in ldf["sequence"].tolist()]
                counts = [lens.count(l) for l in all_lengths]
                offsets = (np.array(all_lengths, dtype=float)
                           + li * bar_w - (n_libs - 1) * bar_w / 2)
                ax.bar(offsets, counts, width=bar_w * 0.9,
                       color=LIB_PALETTE[li % len(LIB_PALETTE)],
                       alpha=0.85, label=f"Library {li}")
            ax.set_xlabel("Sequence length (residues)")
            ax.set_ylabel("Count")
            ax.set_title("Sequence Length Distribution by Library")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.legend(framealpha=0.8)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 4: Dendrogram (selected peptides, if available) ──────────
            if ps is not None and getattr(ps, "seq_dist_matrix", None) is not None:
                try:
                    from scipy.cluster.hierarchy import (
                        linkage as _lk, dendrogram as _sd)
                    from scipy.spatial.distance import squareform as _sq

                    DCAP  = 400
                    rng_d = np.random.default_rng(42)
                    sel_idx = np.concatenate(
                        [ldf["index"].to_numpy() for ldf in libraries.values()]
                    )
                    if len(sel_idx) > DCAP:
                        sel_idx = rng_d.choice(sel_idx, DCAP, replace=False)

                    sub = ps.seq_dist_matrix[np.ix_(sel_idx, sel_idx)].copy()
                    np.fill_diagonal(sub, 0.0)
                    Z = _lk(_sq(sub), method="average")
                    seq_labels = [ps._peptide_seq[i] for i in sel_idx]

                    leaf_colors = {}
                    for li, ldf in libraries.items():
                        for seq in ldf["sequence"].tolist():
                            leaf_colors[seq] = LIB_PALETTE[li % len(LIB_PALETTE)]

                    n_leaves = len(sel_idx)
                    fig_h = max(6, min(0.14 * n_leaves, 16))
                    fig, ax = plt.subplots(figsize=(8.5, fig_h))
                    _sd(Z, labels=seq_labels, ax=ax, orientation="right",
                        leaf_font_size=5 if n_leaves > 80 else 7,
                        count_sort="descendent")
                    for lbl in ax.get_yticklabels():
                        lbl.set_color(
                            leaf_colors.get(lbl.get_text(), UNSELECTED_COLOR)
                        )
                    note = (
                        f"{n_leaves}"
                        + (f" sampled from {len(libraries)} libraries"
                           if len(sel_idx) < sum(len(d) for d in libraries.values())
                           else "")
                    )
                    ax.set_title(
                        f"BLOSUM62 Sequence Diversity — UPGMA ({note} peptides)"
                    )
                    ax.set_xlabel("BLOSUM62 distance")
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                except Exception:
                    pass

            # ── Pages 5+: Per-library sequence tables ─────────────────────────
            PAGE_ROWS, MAX_ROWS = 45, 500
            for li, ldf in libraries.items():
                display   = ldf[["sequence", "cyclic_mw", "linear_mw"]].head(MAX_ROWS)
                n_rows    = len(display)
                truncated = len(ldf) > MAX_ROWS
                n_pages   = max(1, (n_rows + PAGE_ROWS - 1) // PAGE_ROWS)
                lib_color = LIB_PALETTE[li % len(LIB_PALETTE)]

                for page_i in range(n_pages):
                    chunk = display.iloc[page_i * PAGE_ROWS: (page_i + 1) * PAGE_ROWS]
                    title = f"Library {li} — Sequences"
                    if n_pages > 1:
                        title += f"  (page {page_i + 1}/{n_pages})"
                    if truncated and page_i == n_pages - 1:
                        title += f"  [first {MAX_ROWS:,} of {len(ldf):,} shown]"

                    fig_h = min(1.6 + len(chunk) * 0.22, 11)
                    fig   = plt.figure(figsize=(8.5, fig_h))
                    fig.text(0.06, 0.97, title, fontsize=12, fontweight="bold",
                             va="top", ha="left")
                    ax = fig.add_axes([0.02, 0.0, 0.96, 0.91])
                    ax.axis("off")
                    rows = [
                        [row["sequence"],
                         f"{row['cyclic_mw']:.4f}",
                         f"{row['linear_mw']:.4f}"]
                        for _, row in chunk.iterrows()
                    ]
                    tbl = ax.table(
                        cellText=rows,
                        colLabels=["Sequence", "Cyclic MW (Da)", "Linear MW (Da)"],
                        cellLoc="left", loc="upper center",
                        colWidths=[0.46, 0.27, 0.27],
                    )
                    _style_table(tbl, header_color=lib_color)
                    tbl.set_fontsize(8)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

    buf.seek(0)
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if _logo_b64:
        st.markdown(
            f'<img src="data:image/svg+xml;base64,{_logo_b64}" '
            f'style="width:110px; display:block; margin:0 auto 10px auto;">',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p style="text-align:center; font-size:0.8rem; color:rgba(250,250,250,0.6);">'
        '<i>Cyclic Peptide Library Builder</i><br>Serebryany Lab · Stony Brook University</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Conflict Graph Parameters")
    st.caption(
        "Two peptides conflict if their BLOSUM62 sequence distance is below the "
        "threshold **or** their cyclic/linear masses are within the mass threshold."
    )

    n_lib = st.number_input(
        "Number of libraries", min_value=1, max_value=8, value=2, step=1,
        help="Non-overlapping libraries to construct.",
    )
    seq_threshold = st.slider(
        "Sequence distance threshold", min_value=0.0, max_value=1.0,
        value=0.2, step=0.01,
        help="BLOSUM62 distance below which two peptides conflict. "
             "0 = skip sequence filtering.",
    )
    mw_threshold = st.number_input(
        "Mass threshold (Da)", min_value=0.1, max_value=20.0, value=3.0, step=0.5,
        help="Cyclic/linear MW difference below which two peptides are "
             "indistinguishable by MS.",
    )
    n_seeds = st.number_input(
        "Seeds to evaluate", min_value=1, max_value=200, value=20, step=1,
        help="More seeds = better chance of finding an optimal split, but slower.",
    )

    st.divider()
    st.subheader("Visualisation")
    show_mw_hist    = st.toggle("MW distribution",               value=True)
    show_len_hist   = st.toggle("Sequence length distribution",  value=True)
    show_dendrogram = st.toggle(
        "Sequence diversity dendrogram", value=True,
        help="UPGMA dendrogram on BLOSUM62 distances. "
             "Requires seq_threshold > 0. Capped at 400 leaves.",
    )
    show_graph      = st.toggle(
        "Conflict graph", value=True,
        help="Disabled automatically for >400 peptides.",
    )

    st.divider()


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Cyclic Peptide Library Builder")
st.markdown(
    """
<p style="text-align: justify;">
Cyclic peptides are a powerful class of macrocyclic compounds that combine the conformational
rigidity of constrained scaffolds with the chemical diversity of natural amino acid side chains,
making them attractive leads for biochemical screening. When screening large combinatorial
libraries by mass spectrometry, two peptides become practically indistinguishable if their
cyclic or linear molecular weights fall within the instrument's resolution window — and
biochemically redundant if their sequences are too similar by BLOSUM62 substitution score.
This tool solves both problems simultaneously by formulating library partitioning as a
<b>Maximum Independent Set</b> problem on a pairwise conflict graph: two peptides share a conflict
edge if they are too close in mass <i>or</i> too similar in sequence. A fast greedy solver is run
over multiple random seeds to maximise the number of peptides retained per library. The result
is a set of non-overlapping libraries, each internally free of mass-spec ambiguity, ready for
direct synthesis and MS-based screening.
</p>
""",
    unsafe_allow_html=True,
)

# ── Input tabs ────────────────────────────────────────────────────────────────
upload_tab, paste_tab = st.tabs(["Upload CSV / TXT", "Paste sequences"])
peptide_list: list[str] = []

with upload_tab:
    uploaded = st.file_uploader(
        "One peptide sequence per row (CSV or plain text).",
        type=["csv", "txt"],
    )
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".csv"):
            df_upload = pd.read_csv(io.StringIO(raw))
            col_pick = st.selectbox(
                "Select the sequence column", df_upload.columns.tolist()
            )
            peptide_list = (
                df_upload[col_pick].dropna().astype(str).str.strip().tolist()
            )
        else:
            peptide_list = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        st.success(f"Loaded **{len(peptide_list):,}** sequences.")

with paste_tab:
    pasted = st.text_area(
        "One sequence per line",
        height=180,
        placeholder=textwrap.dedent("""\
            ACDEFG
            ACDEFH
            KLMNPQ
            KLMNPR
            WYVFIL
        """),
    )
    if pasted.strip():
        peptide_list = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
        st.info(f"Detected **{len(peptide_list):,}** sequences.")

# ── Clear session if parameters changed ──────────────────────────────────────
_current_params = dict(
    n_lib=int(n_lib),
    seq_threshold=float(seq_threshold),
    mw_threshold=float(mw_threshold),
    n_seeds=int(n_seeds),
)
if st.session_state.get("last_params") != _current_params:
    st.session_state.pop("results", None)
    st.session_state.pop("pdf_bytes", None)
    st.session_state["last_params"] = _current_params

# ── Run ───────────────────────────────────────────────────────────────────────
st.divider()
col_btn, col_warn = st.columns([0.22, 0.78])
with col_btn:
    run_btn = st.button(
        "▶  Run Library Selection", type="primary", use_container_width=True
    )

if not TOOL_AVAILABLE:
    col_warn.error(f"Could not import PeptideSelection — **{IMPORT_ERROR}**")

if run_btn:
    if not peptide_list:
        st.error("Please upload or paste at least one peptide sequence first.")
        st.stop()

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    bad = [s for s in peptide_list if not s or not set(s.upper()).issubset(valid_aa)]
    if bad:
        st.error(
            f"{len(bad)} sequence(s) contain non-standard amino acids or are empty: "
            f"{bad[:5]}{'…' if len(bad) > 5 else ''}"
        )
        st.stop()

    peptide_list = [s.upper() for s in peptide_list]

    log_capture = io.StringIO()
    import sys
    with st.status("Running library selection …", expanded=True) as status:
        st.write("Initialising PeptideSelection …")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = log_capture
        sys.stderr = log_capture
        try:
            if TOOL_AVAILABLE:
                ps = PeptideSelection(
                    peptide_seqs=peptide_list,
                    n_lib=int(n_lib),
                    seq_threshold=float(seq_threshold),
                    mw_threshold=float(mw_threshold),
                )
                st.write(
                    f"Computing libraries ({int(n_seeds)} seeds, "
                    f"{int(n_lib)} {'library' if n_lib == 1 else 'libraries'}) …"
                )
                ps.construct_libraries(n_seeds=int(n_seeds))
                libraries   = ps.libraries
                total_input = ps.total_peptides
                total_sel   = ps.total_selected
                all_seqs    = ps._peptide_seq
                cyc_mw_all  = ps._cyc_mw
                lin_mw_all  = ps._lin_mw
            else:
                rng  = np.random.default_rng(42)
                aa   = list("ACDEFGHIKLMNPQRSTVWY")
                mock = [
                    "".join(rng.choice(aa, rng.integers(5, 10)))
                    for _ in range(60)
                ]
                half = len(mock) // int(n_lib)
                libraries = {
                    i: pd.DataFrame({
                        "sequence":  mock[i * half: (i + 1) * half],
                        "cyclic_mw": rng.uniform(600, 900, half),
                        "linear_mw": rng.uniform(618, 918, half),
                        "index":     np.arange(i * half, (i + 1) * half),
                    })
                    for i in range(int(n_lib))
                }
                total_input = len(mock)
                total_sel   = sum(len(d) for d in libraries.values())
                all_seqs    = mock
                cyc_mw_all  = np.array([rng.uniform(600, 900) for _ in mock])
                lin_mw_all  = cyc_mw_all + 18.01
                ps          = None
        except Exception as exc:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            st.exception(exc)
            st.stop()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        status.update(label="Done!", state="complete", expanded=False)

    idx_to_lib: Dict[int, int] = {}
    for lib_i, df_i in libraries.items():
        for gi in df_i["index"].tolist():
            idx_to_lib[gi] = lib_i

    st.session_state["results"] = dict(
        libraries=libraries,
        total_input=total_input,
        total_sel=total_sel,
        all_seqs=all_seqs,
        cyc_mw_all=cyc_mw_all,
        lin_mw_all=lin_mw_all,
        idx_to_lib=idx_to_lib,
        ps=ps,
        log_text=log_capture.getvalue(),
        params=dict(
            n_lib=int(n_lib),
            seq_threshold=float(seq_threshold),
            mw_threshold=float(mw_threshold),
            n_seeds=int(n_seeds),
        ),
    )
    st.session_state.pop("pdf_bytes", None)

# ── Render results (persists across reruns via session_state) ─────────────────
if "results" in st.session_state:
    _r          = st.session_state["results"]
    libraries   = _r["libraries"]
    total_input = _r["total_input"]
    total_sel   = _r["total_sel"]
    all_seqs    = _r["all_seqs"]
    cyc_mw_all  = _r["cyc_mw_all"]
    lin_mw_all  = _r["lin_mw_all"]
    idx_to_lib  = _r["idx_to_lib"]
    ps          = _r["ps"]
    log_text    = _r["log_text"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("Summary")
    n_unselected = total_input - total_sel
    m_cols = st.columns(4 + int(n_lib))
    m_cols[0].metric("Input peptides", f"{total_input:,}")
    m_cols[1].metric("Total selected", f"{total_sel:,}")
    m_cols[2].metric("Not selected",   f"{n_unselected:,}")
    m_cols[3].metric("Overall yield",  f"{100 * total_sel / total_input:.1f}%")
    for lib_i, lib_df in libraries.items():
        y = 100 * len(lib_df) / total_input
        m_cols[4 + lib_i].metric(
            f"Library {lib_i}",
            f"{len(lib_df):,}",
            delta=f"{y:.1f}% of input",
        )

    # ── Visualisations ────────────────────────────────────────────────────────
    vis_sections = [v for v in [show_mw_hist, show_len_hist] if v]
    if vis_sections:
        st.divider()
        st.subheader("Visualisations")

    _plotly_layout = dict(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(bgcolor="#1e1e2e", bordercolor="#444", borderwidth=1),
        margin=dict(l=60, r=20, t=50, b=50),
    )

    if show_mw_hist:
        fig_hist   = go.Figure()
        bins_edges = np.linspace(cyc_mw_all.min(), cyc_mw_all.max(), 61)
        unsel_mask = np.array(
            [idx_to_lib.get(i, -1) == -1 for i in range(len(all_seqs))]
        )
        if unsel_mask.any():
            fig_hist.add_trace(go.Histogram(
                x=cyc_mw_all[unsel_mask].tolist(),
                xbins=dict(start=bins_edges[0], end=bins_edges[-1],
                           size=bins_edges[1] - bins_edges[0]),
                name=f"Not selected (n={int(unsel_mask.sum()):,})",
                marker_color=UNSELECTED_COLOR,
                opacity=0.4,
            ))
        for lib_i, lib_df in libraries.items():
            fig_hist.add_trace(go.Histogram(
                x=lib_df["cyclic_mw"].tolist(),
                xbins=dict(start=bins_edges[0], end=bins_edges[-1],
                           size=bins_edges[1] - bins_edges[0]),
                name=f"Library {lib_i} (n={len(lib_df):,})",
                marker_color=LIB_PALETTE[lib_i % len(LIB_PALETTE)],
                opacity=0.7,
            ))
        fig_hist.update_layout(
            **_plotly_layout,
            barmode="overlay",
            title="Cyclic MW Distribution by Library",
            xaxis_title="Cyclic MW (Da)",
            yaxis_title="Count",
            height=380,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    if show_len_hist:
        lengths_all = np.array([len(s) for s in all_seqs])
        len_min, len_max = int(lengths_all.min()), int(lengths_all.max())
        all_lengths = list(range(len_min, len_max + 1))

        fig_len = go.Figure()
        for lib_i, lib_df in libraries.items():
            lens   = [len(s) for s in lib_df["sequence"].tolist()]
            counts = [lens.count(l) for l in all_lengths]
            fig_len.add_trace(go.Bar(
                x=all_lengths,
                y=counts,
                name=f"Library {lib_i}",
                marker_color=LIB_PALETTE[lib_i % len(LIB_PALETTE)],
                opacity=0.85,
            ))
        fig_len.update_layout(
            **_plotly_layout,
            barmode="group",
            title="Sequence Length Distribution by Library",
            xaxis_title="Sequence length (residues)",
            yaxis_title="Count",
            xaxis=dict(tickmode="linear", dtick=1),
            height=380,
        )
        st.plotly_chart(fig_len, use_container_width=True)

    if show_dendrogram:
        st.divider()
        st.subheader("Sequence Diversity Dendrogram")

        if ps is None or ps.seq_dist_matrix is None:
            st.info(
                "Dendrogram requires **seq_threshold > 0** so that the BLOSUM62 "
                "distance matrix is computed. Re-run with a non-zero threshold."
            )
        else:
            try:
                from scipy.cluster.hierarchy import (
                    linkage as _linkage, dendrogram as _scipy_dendro)
                from scipy.spatial.distance import squareform as _squareform

                dendro_include_unsel = st.toggle(
                    "Include unselected peptides", value=False, key="dendro_unsel"
                )

                DENDRO_CAP = 400
                n_total    = ps.total_peptides
                rng_d      = np.random.default_rng(42)

                groups: Dict[str, np.ndarray] = {}
                for _li, _ldf in libraries.items():
                    groups[f"lib_{_li}"] = _ldf["index"].to_numpy()
                _unsel_idx = np.array(
                    [i for i in range(n_total) if idx_to_lib.get(i, -1) == -1]
                )
                if dendro_include_unsel and len(_unsel_idx):
                    groups["unsel"] = _unsel_idx

                pool_idx = (
                    np.concatenate(list(groups.values()))
                    if groups else np.array([], dtype=int)
                )
                n_pool = len(pool_idx)

                if n_pool > DENDRO_CAP:
                    sampled: list[int] = []
                    for _gidx in groups.values():
                        k = max(1, round(DENDRO_CAP * len(_gidx) / n_pool))
                        k = min(k, len(_gidx))
                        sampled.extend(rng_d.choice(_gidx, k, replace=False).tolist())
                    sampled_idx = np.array(sampled[:DENDRO_CAP])
                    note = (
                        f"stratified sample of {len(sampled_idx):,} "
                        f"from {n_pool:,} selected"
                    )
                else:
                    sampled_idx = pool_idx
                    note        = f"{n_pool:,} sequences"

                sub = ps.seq_dist_matrix[np.ix_(sampled_idx, sampled_idx)].copy()
                np.fill_diagonal(sub, 0.0)
                Z = _linkage(_squareform(sub), method="average")

                seq_labels = [ps._peptide_seq[i] for i in sampled_idx]
                dendro     = _scipy_dendro(
                    Z, labels=seq_labels, no_plot=True, count_sort="descendent"
                )

                fig_d = go.Figure()
                for xs, ys in zip(dendro["dcoord"], dendro["icoord"]):
                    fig_d.add_trace(go.Scatter(
                        x=xs, y=ys,
                        mode="lines",
                        line=dict(color="#555e6e", width=0.9),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                leaf_labels = dendro["ivl"]
                n_leaves    = len(leaf_labels)
                leaf_y      = np.arange(5, 5 + 10 * n_leaves, 10)
                seq_to_y    = {seq: leaf_y[j] for j, seq in enumerate(leaf_labels)}

                for _li, _ldf in libraries.items():
                    lib_seqs = set(_ldf["sequence"].tolist())
                    ys_lib   = [seq_to_y[s] for s in leaf_labels if s in lib_seqs]
                    seqs_lib = [s           for s in leaf_labels if s in lib_seqs]
                    if not ys_lib:
                        continue
                    fig_d.add_trace(go.Scatter(
                        x=[0] * len(ys_lib), y=ys_lib,
                        mode="markers",
                        name=f"Library {_li}",
                        marker=dict(color=LIB_PALETTE[_li % len(LIB_PALETTE)],
                                    size=7, symbol="circle"),
                        text=seqs_lib,
                        hovertemplate=(
                            "<b>%{text}</b><br>Library " + str(_li) + "<extra></extra>"
                        ),
                    ))

                if dendro_include_unsel:
                    unsel_seqs = {
                        ps._peptide_seq[i]
                        for i in range(n_total)
                        if idx_to_lib.get(i, -1) == -1
                    }
                    ys_u   = [seq_to_y[s] for s in leaf_labels if s in unsel_seqs]
                    seqs_u = [s           for s in leaf_labels if s in unsel_seqs]
                    if ys_u:
                        fig_d.add_trace(go.Scatter(
                            x=[0] * len(ys_u), y=ys_u,
                            mode="markers",
                            name="Not selected",
                            marker=dict(color=UNSELECTED_COLOR, size=5,
                                        symbol="circle-open"),
                            text=seqs_u,
                            hovertemplate=(
                                "<b>%{text}</b><br>Not selected<extra></extra>"
                            ),
                        ))

                show_tick_labels = n_leaves <= 80
                height_d         = max(520, 11 * n_leaves)
                fig_d.update_layout(
                    **_plotly_layout,
                    title=f"BLOSUM62 Sequence Diversity — UPGMA ({note})",
                    xaxis=dict(title="BLOSUM62 distance",
                               zeroline=False, showgrid=False),
                    yaxis=dict(
                        showticklabels=show_tick_labels,
                        tickvals=leaf_y.tolist() if show_tick_labels else [],
                        ticktext=leaf_labels      if show_tick_labels else [],
                        tickfont=dict(size=8, family="monospace"),
                        showgrid=False,
                        zeroline=False,
                    ),
                    height=height_d,
                )
                st.plotly_chart(fig_d, use_container_width=True)

            except ImportError:
                st.warning(
                    "scipy is not available in this browser session yet. "
                    "It is installed on first use — try re-running after a moment."
                )

    # ── Per-library tables ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Libraries")

    lib_tab_labels = [f"Library {i}" for i in libraries] + ["Unselected"]
    lib_tabs       = st.tabs(lib_tab_labels)

    for tab, (lib_i, lib_df) in zip(lib_tabs[:-1], libraries.items()):
        with tab:
            display         = lib_df[["sequence", "cyclic_mw", "linear_mw"]].copy()
            display.columns = ["Sequence", "Cyclic MW (Da)", "Linear MW (Da)"]

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Min cyclic MW",    f"{lib_df['cyclic_mw'].min():.4f}")
            sc2.metric("Median cyclic MW", f"{lib_df['cyclic_mw'].median():.4f}")
            sc3.metric("Max cyclic MW",    f"{lib_df['cyclic_mw'].max():.4f}")
            sc4.metric("Unique MWs",       str(lib_df["cyclic_mw"].nunique()))

            st.dataframe(
                display.style.format(
                    {"Cyclic MW (Da)": "{:.4f}", "Linear MW (Da)": "{:.4f}"}
                ),
                use_container_width=True,
                height=320,
            )
            st.download_button(
                f"⬇  Download Library {lib_i} CSV",
                data=display.to_csv(index=False).encode(),
                file_name=f"peptide_library_{lib_i}.csv",
                mime="text/csv",
                key=f"dl_lib_{lib_i}",
            )

    with lib_tabs[-1]:
        unsel_indices = [
            i for i in range(len(all_seqs)) if idx_to_lib.get(i, -1) == -1
        ]
        if unsel_indices:
            unsel_df = pd.DataFrame({
                "Sequence":       [all_seqs[i] for i in unsel_indices],
                "Cyclic MW (Da)": cyc_mw_all[unsel_indices],
                "Linear MW (Da)": lin_mw_all[unsel_indices],
            })
            st.info(f"{len(unsel_df):,} peptides were not placed in any library.")
            st.dataframe(
                unsel_df.style.format(
                    {"Cyclic MW (Da)": "{:.4f}", "Linear MW (Da)": "{:.4f}"}
                ),
                use_container_width=True,
                height=320,
            )
            st.download_button(
                "⬇  Download unselected CSV",
                data=unsel_df.to_csv(index=False).encode(),
                file_name="peptide_unselected.csv",
                mime="text/csv",
                key="dl_unsel",
            )
        else:
            st.success("All peptides were assigned to a library.")

    # ── Combined download + PDF ───────────────────────────────────────────────
    st.divider()
    combined_frames = []
    for lib_i, lib_df in libraries.items():
        f = lib_df[["sequence", "cyclic_mw", "linear_mw"]].copy()
        f.insert(0, "library", lib_i)
        combined_frames.append(f)
    combined_csv = pd.concat(combined_frames, ignore_index=True).to_csv(
        index=False
    ).encode()

    dl_col1, dl_col2 = st.columns([0.35, 0.65])
    dl_col1.download_button(
        "⬇  Download all libraries (combined CSV)",
        data=combined_csv,
        file_name="peptide_libraries_all.csv",
        mime="text/csv",
        key="dl_all",
    )

    if "pdf_bytes" not in st.session_state:
        if dl_col2.button("📄  Generate PDF report", key="btn_pdf"):
            with st.spinner("Building PDF report …"):
                st.session_state["pdf_bytes"] = _build_pdf_report(
                    libraries=libraries,
                    total_input=total_input,
                    total_sel=total_sel,
                    all_seqs=all_seqs,
                    cyc_mw_all=cyc_mw_all,
                    lin_mw_all=lin_mw_all,
                    idx_to_lib=idx_to_lib,
                    params=_r["params"],
                    ps=ps,
                )
            st.rerun()
    else:
        pdf_col1, pdf_col2 = dl_col2.columns([0.65, 0.35])
        pdf_col1.download_button(
            "⬇  Download PDF report",
            data=st.session_state["pdf_bytes"],
            file_name="peptide_library_report.pdf",
            mime="application/pdf",
            key="dl_pdf",
        )
        if pdf_col2.button("↺  Regenerate", key="btn_pdf_regen"):
            st.session_state.pop("pdf_bytes")
            st.rerun()

    # ── Conflict graph ────────────────────────────────────────────────────────
    if show_graph:
        st.divider()
        st.subheader("Conflict Graph")
        if total_input > 400:
            st.info(
                f"Conflict graph is hidden for inputs larger than 400 peptides "
                f"({total_input:,} provided). Reduce the input or toggle the "
                f"visualisation off."
            )
        elif ps is not None and TOOL_AVAILABLE:
            with st.spinner("Rendering conflict graph …"):
                fig_g = ps.plot_library_conflict_graph(figsize=(14, 9))
                st.pyplot(fig_g)
                buf = io.BytesIO()
                fig_g.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig_g)
                st.download_button(
                    "⬇  Download conflict graph PNG",
                    data=buf.getvalue(),
                    file_name="conflict_graph.png",
                    mime="image/png",
                    key="dl_graph",
                )
        else:
            st.info(
                "Conflict graph is available when running with the real "
                "PeptideSelection class."
            )

    # ── Run log ───────────────────────────────────────────────────────────────
    with st.expander("📋  Run log"):
        st.code(log_text or "(no output captured)", language="text")
