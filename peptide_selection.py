from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Hardcoded BLOSUM62 (replaces Bio.Align.substitution_matrices) ─────────────
# Standard BLOSUM62, alphabet order: A R N D C Q E G H I L K M F P S T W Y V
_BLOSUM62_ALPHA = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_MATRIX = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
], dtype=np.float64)
_BLOSUM62_IDX = {aa: i for i, aa in enumerate(_BLOSUM62_ALPHA)}


class PeptideSelection:
    """Selects non-conflicting peptide libraries based on sequence similarity
    and mass spectrometry constraints.

    Peptides are considered conflicting if they are too similar in sequence
    (BLOSUM62 distance) or too close in molecular weight (cyclic or linear),
    which would make them indistinguishable by mass spec.

    Stlite/browser edition — no biopython, no multiprocessing.
    """

    AA_MASSES: Dict[str, float] = {
        "G": 57.021463735, "A": 71.037113805, "S": 87.032028435, "P": 97.052763875,
        "V": 99.068413945, "T": 101.047678505, "C": 103.009184505, "I": 113.084064015,
        "L": 113.084064015, "N": 114.042927470, "D": 115.026943065, "Q": 128.058577540,
        "K": 128.094963050, "E": 129.042593135, "M": 131.040484645, "H": 137.058911875,
        "F": 147.068413945, "R": 156.101111050, "Y": 163.063328575, "W": 186.079312980,
    }

    def __init__(
        self,
        peptide_seqs: Iterable[str],
        n_lib: int = 1,
        seq_threshold: float = 0.2,
        mw_threshold: float = 3.0,
    ) -> None:
        print("Initialising PeptideSelection ...")

        self._peptide_seq: List[str] = sorted(set(peptide_seqs))
        self.n_lib = n_lib
        self.seq_threshold = seq_threshold
        self.mw_threshold = mw_threshold

        self._lib_peptide_seq: Set[str] = set()
        self.libraries: Dict[int, pd.DataFrame] = {}

        self._cyc_mw: np.ndarray = self._calculate_peptide_weights(self._peptide_seq)
        self._lin_mw: np.ndarray = self._cyc_mw + 18.010565

        if self.seq_threshold > 0:
            self.seq_dist_matrix: Optional[np.ndarray] = self._calc_blosum62_dist_matrix(
                self._peptide_seq
            )
        else:
            print("seq_threshold=0: skipping BLOSUM62 distance matrix computation.")
            self.seq_dist_matrix = None

        self.pept_df = pd.DataFrame({
            "sequence":  self._peptide_seq,
            "cyclic_mw": self._cyc_mw,
            "linear_mw": self._lin_mw,
            "index":     np.arange(len(self._peptide_seq)),
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def construct_libraries(self, n_seeds: int = 20) -> None:
        seeds = list(range(n_seeds))
        n_lib_str = f"{self.n_lib} librar{'y' if self.n_lib == 1 else 'ies'}"
        print(f"Searching {n_seeds} seed(s) for the best {n_lib_str} ...\n")

        best: Dict = {"total": -1, "libraries": {}, "seed": seeds[0]}

        with tqdm(total=n_seeds, desc="Seed search") as pbar:
            for seed in seeds:
                libraries, total = self._trial_construct_libraries(seed)
                if total > best["total"]:
                    best["total"] = total
                    best["libraries"] = libraries
                    best["seed"] = seed
                pbar.set_postfix(best=best["total"], best_seed=best["seed"])
                pbar.update(1)

        self.libraries = best["libraries"]
        self._lib_peptide_seq = set(
            seq
            for df in self.libraries.values()
            for seq in df["sequence"].tolist()
        )

        sep = "─" * 62
        overall_yield = 100 * best["total"] / self.total_peptides
        n_unique_input_mw = len(np.unique(self._cyc_mw))
        frac_unique_input = 100 * n_unique_input_mw / self.total_peptides
        print(f"\n{sep}")
        print(f"  Best seed : {best['seed']}")
        print(f"  Input     : {self.total_peptides:,} peptides  "
              f"({n_unique_input_mw:,} unique cyclic MWs, {frac_unique_input:.1f}%)")
        print(f"  Selected  : {best['total']:,} / {self.total_peptides:,}  "
              f"({overall_yield:.1f}% overall yield)")
        print(sep)

        for lib_idx, df in self.libraries.items():
            n_sel       = len(df)
            yield_      = 100 * n_sel / self.total_peptides
            n_unique_mw = len(df["cyclic_mw"].unique())
            frac_unique = 100 * n_unique_mw / n_sel
            mw          = self._pairwise_mw_stats(df)
            print(f"\n  Library {lib_idx}  —  {n_sel:,} peptides  ({yield_:.1f}% of input)  "
                  f"|  {n_unique_mw:,} unique MWs ({frac_unique:.1f}%)")
            print(f"    Cyclic MW   min {mw['cyc_min']:.3f}  "
                  f"median {mw['cyc_med']:.3f}  "
                  f"max {mw['cyc_max']:.3f} Da")

        print(f"\n{sep}")

    @property
    def total_peptides(self) -> int:
        return len(self._peptide_seq)

    @property
    def total_selected(self) -> int:
        return sum(len(df) for df in self.libraries.values())

    # ------------------------------------------------------------------
    # Internal trial runner
    # ------------------------------------------------------------------

    def _trial_construct_libraries(
        self, seed: int, verbose: bool = False,
    ) -> Tuple[Dict[int, pd.DataFrame], int]:
        libraries: Dict[int, pd.DataFrame] = {}
        used_seqs: Set[str] = set()

        for lib_idx in range(self.n_lib):
            remaining_seqs = set(self._peptide_seq) - used_seqs
            lib_df  = self.pept_df[self.pept_df["sequence"].isin(remaining_seqs)]
            prev_df = self.pept_df[self.pept_df["sequence"].isin(used_seqs)]

            indices      = lib_df["index"].to_numpy()
            n_candidates = len(indices)

            if self.seq_dist_matrix is not None:
                sub_dist_matrix = self.seq_dist_matrix[np.ix_(indices, indices)]
            else:
                sub_dist_matrix = np.zeros((n_candidates, n_candidates))

            if len(prev_df) == 0:
                prev_indices: Optional[np.ndarray] = None
                prev_sel_dist: Optional[np.ndarray] = None
            else:
                prev_indices = prev_df["index"].to_numpy()
                prev_sel_dist = (
                    self.seq_dist_matrix[np.ix_(indices, prev_indices)]
                    if self.seq_dist_matrix is not None
                    else None
                )

            selected_local_indices, _ = self._get_constrained_independent_set(
                seq_dist_matrix=sub_dist_matrix,
                mw_cyclic=self._cyc_mw[indices],
                mw_linear=self._lin_mw[indices],
                prev_sel_idx=prev_indices,
                prev_sel_dist=prev_sel_dist,
                random_seed=seed,
            )

            selected_df = lib_df.iloc[selected_local_indices].reset_index(drop=True)
            used_seqs.update(selected_df["sequence"].tolist())
            libraries[lib_idx] = selected_df

        total = sum(len(df) for df in libraries.values())
        return libraries, total

    # ------------------------------------------------------------------
    # Core solver
    # ------------------------------------------------------------------

    def _get_constrained_independent_set(
        self,
        seq_dist_matrix: np.ndarray,
        mw_cyclic: np.ndarray,
        mw_linear: np.ndarray,
        node_labels: Optional[List[str]] = None,
        exact: bool = False,
        prev_sel_idx: Optional[np.ndarray] = None,
        prev_sel_dist: Optional[np.ndarray] = None,
        random_seed: int = 42,
        verbose: bool = False,
    ) -> Tuple[List[int], List[str]]:
        mw_cyclic = np.asarray(mw_cyclic)
        mw_linear = np.asarray(mw_linear)
        n = len(mw_cyclic)

        allowed = np.arange(n)
        if (
            self.seq_threshold > 0
            and prev_sel_idx is not None
            and prev_sel_dist is not None
            and len(prev_sel_idx) > 0
        ):
            forbidden_mask = (prev_sel_dist < self.seq_threshold).any(axis=1)
            allowed = np.where(~forbidden_mask)[0]

        if len(allowed) == 0:
            return [], []

        cyc_allowed = mw_cyclic[allowed]
        unique_cyc, rep_in_allowed = np.unique(cyc_allowed, return_index=True)
        unique_lin = unique_cyc + 18.010565

        red_cc = np.abs(unique_cyc[:, None] - unique_cyc[None, :]) < self.mw_threshold
        red_cl = np.abs(unique_cyc[:, None] - unique_lin[None, :]) < self.mw_threshold
        red_lc = np.abs(unique_lin[:, None] - unique_cyc[None, :]) < self.mw_threshold
        red_conflict = red_cc | red_cl | red_lc

        if self.seq_threshold > 0:
            rep_pool_idx = allowed[rep_in_allowed]
            rep_seq = seq_dist_matrix[np.ix_(rep_pool_idx, rep_pool_idx)] < self.seq_threshold
            red_conflict |= rep_seq

        np.fill_diagonal(red_conflict, False)

        mis_group_list = self._numpy_maximal_independent_set(red_conflict, seed=random_seed)

        selected_rep_in_allowed = rep_in_allowed[mis_group_list]
        best_indices: Set[int] = {int(allowed[i]) for i in selected_rep_in_allowed}

        sorted_indices = sorted(best_indices)
        labels_out = [node_labels[i] for i in sorted_indices] if node_labels else []
        return sorted_indices, labels_out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numpy_maximal_independent_set(
        conflict: np.ndarray,
        seed: int,
    ) -> List[int]:
        rng = np.random.default_rng(seed)
        n = conflict.shape[0]
        available = np.ones(n, dtype=bool)
        selected: List[int] = []
        for i in rng.permutation(n):
            if available[i]:
                selected.append(int(i))
                available[conflict[i]] = False
                available[i] = False
        return selected

    def _pairwise_mw_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        def _stats(mws: np.ndarray) -> Tuple[float, float, float]:
            diffs = np.abs(mws[:, None] - mws[None, :])
            upper = diffs[np.triu_indices(len(mws), k=1)]
            if len(upper) == 0:
                return 0.0, 0.0, 0.0
            return float(upper.min()), float(np.median(upper)), float(upper.max())

        cyc = df["cyclic_mw"].to_numpy()
        lin = df["linear_mw"].to_numpy()
        cyc_min, cyc_med, cyc_max = _stats(cyc)
        lin_min, lin_med, lin_max = _stats(lin)
        return dict(cyc_min=cyc_min, cyc_med=cyc_med, cyc_max=cyc_max,
                    lin_min=lin_min, lin_med=lin_med, lin_max=lin_max)

    def _cross_library_seq_stats(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame
    ) -> Dict[str, float]:
        if self.seq_dist_matrix is None:
            return {"min": 0.0, "med": 0.0, "max": 0.0}
        idx_a = df_a["index"].to_numpy()
        idx_b = df_b["index"].to_numpy()
        dists = self.seq_dist_matrix[np.ix_(idx_a, idx_b)].ravel()
        if len(dists) == 0:
            return {"min": 0.0, "med": 0.0, "max": 0.0}
        return {"min": float(dists.min()), "med": float(np.median(dists)),
                "max": float(dists.max())}

    def _calculate_peptide_weights(self, sequences: List[str]) -> np.ndarray:
        return np.array([
            sum(self.AA_MASSES[aa] for aa in seq)
            for seq in sequences
        ])

    def _calc_blosum62_dist_matrix(self, peptides: List[str]) -> np.ndarray:
        """Compute pairwise BLOSUM62-normalised distance matrix (no biopython)."""
        score_mat = _BLOSUM62_MATRIX
        aa_idx    = _BLOSUM62_IDX

        n       = len(peptides)
        max_len = max(len(p) for p in peptides)

        encoded = np.full((n, max_len), -1, dtype=np.int32)
        for i, pep in enumerate(tqdm(peptides, desc="Encoding peptides")):
            encoded[i, :len(pep)] = [aa_idx[aa] for aa in pep]

        valid    = encoded >= 0
        enc_safe = np.where(valid, encoded, 0)

        diag        = score_mat[enc_safe, enc_safe]
        self_scores = (diag * valid).sum(axis=1)

        raw = np.zeros((n, n), dtype=np.float64)
        for p in tqdm(range(max_len), desc="Computing distance matrix"):
            col        = enc_safe[:, p]
            vmask      = valid[:, p]
            valid_pair = vmask[:, None] & vmask[None, :]
            contrib    = score_mat[col[:, None], col[None, :]]
            raw       += np.where(valid_pair, contrib, 0.0)

        norm = np.sqrt(np.outer(self_scores, self_scores))
        sim  = np.clip(raw / norm, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        return 1.0 - sim

    def plot_library_conflict_graph(self, figsize=(16, 11)):
        """Conflict graph coloured by library. Requires matplotlib & networkx."""
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        import matplotlib.pyplot as plt
        import networkx as nx

        LIB_COLORS  = ["#2ecc71", "#00bcd4", "#f300cf", "#f39c12", "#1abc9c",
                       "#e91e63", "#e74c3c", "#8e44ad"]
        UNSELECTED  = "#95a5a6"
        n = len(self._peptide_seq)

        if self.seq_dist_matrix is not None:
            conflict_seq = self.seq_dist_matrix < self.seq_threshold
        else:
            conflict_seq = np.zeros((n, n), dtype=bool)

        conflict_cc = np.abs(self._cyc_mw[:, None] - self._cyc_mw[None, :]) < self.mw_threshold
        conflict_cl = np.abs(self._cyc_mw[:, None] - self._lin_mw[None, :]) < self.mw_threshold
        conflict_lc = np.abs(self._lin_mw[:, None] - self._cyc_mw[None, :]) < self.mw_threshold
        master = conflict_seq | conflict_cc | conflict_cl | conflict_lc
        np.fill_diagonal(master, False)
        G = nx.from_numpy_array(master)

        idx_to_lib: Dict[int, int] = {}
        for lib_idx, df in self.libraries.items():
            for gi in df["index"].tolist():
                idx_to_lib[gi] = lib_idx

        node_colors, node_sizes, node_alpha = [], [], []
        for node in G.nodes():
            if node in idx_to_lib:
                lib = idx_to_lib[node]
                node_colors.append(LIB_COLORS[lib % len(LIB_COLORS)])
                node_sizes.append(300)
                node_alpha.append(1.0)
            else:
                node_colors.append(UNSELECTED)
                node_sizes.append(150)
                node_alpha.append(0.5)

        edge_by_type: Dict[str, list] = {"seq": [], "mw_cc": [], "mw_cl": []}
        for i, j in G.edges():
            if i >= j:
                continue
            if conflict_seq[i, j]:
                edge_by_type["seq"].append((i, j))
            elif conflict_cc[i, j]:
                edge_by_type["mw_cc"].append((i, j))
            else:
                edge_by_type["mw_cl"].append((i, j))

        pos = (nx.kamada_kawai_layout(G) if n < 15
               else nx.spring_layout(G, seed=42, k=0.5))

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=node_alpha)
        edge_styles = {
            "seq":   dict(edge_color="#e67e22", style="solid",  width=1.4, alpha=0.55),
            "mw_cc": dict(edge_color="#9b59b6", style="dashed", width=1.0, alpha=0.45),
            "mw_cl": dict(edge_color="#3498db", style="dotted", width=1.0, alpha=0.45),
        }
        for etype, edges in edge_by_type.items():
            if edges:
                nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, **edge_styles[etype])
        if n <= 60:
            labels = {i: self._peptide_seq[i] for i in range(n)}
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                    font_size=6, font_color="white", font_weight="bold")

        legend_handles = []
        for lib_idx in sorted(self.libraries.keys()):
            color = LIB_COLORS[lib_idx % len(LIB_COLORS)]
            n_pep = len(self.libraries[lib_idx])
            legend_handles.append(
                mpatches.Patch(color=color, label=f"Library {lib_idx}  ({n_pep} peptides)")
            )
        legend_handles.append(mpatches.Patch(color=UNSELECTED, label="Not selected"))
        legend_handles += [
            mlines.Line2D([], [], color="#e67e22", linewidth=1.4,
                          label=f"Conflict: sequence  (dist < {self.seq_threshold})"),
            mlines.Line2D([], [], color="#9b59b6", linewidth=1.0, linestyle="dashed",
                          label=f"Conflict: cyclic-cyclic MW  (< {self.mw_threshold} Da)"),
            mlines.Line2D([], [], color="#3498db", linewidth=1.0, linestyle="dotted",
                          label=f"Conflict: cyclic-linear MW  (< {self.mw_threshold} Da)"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.92)
        ax.set_title(
            f"Peptide Conflict Graph  |  {n} peptides  |  "
            f"{G.number_of_edges()} conflict edges  |  {len(self.libraries)} libraries",
            fontsize=13, pad=14,
        )
        ax.axis("off")
        plt.tight_layout()
        return fig
