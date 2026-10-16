print("SCRIPT STARTED")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import webbrowser
from pathlib import Path
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist


CSV_PATH = "mean_expression_by_cancer.csv"          
GENE_COLUMN = "gene_symbol"                          
OUTPUT_HTML = "repairome_heatmap_interactive.html"


ALTERED_Z_THRESHOLD = 1.5

RANK_BY = "max_abs_z"


ZMIN, ZMAX = -3, 3
LINKAGE_METHOD    = "ward"
DISTANCE_METRIC   = "euclidean"
N_GENE_CLUSTERS   = 6
DEFAULT_VIEW      = "peak_dysregulation"



def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = GENE_COLUMN
    return df



def row_zscore(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=1)
    stds = df.std(axis=1, ddof=0)
    stds_safe = stds.replace(0, np.nan)  
    z = df.sub(means, axis=0).div(stds_safe, axis=0)
    z = z.fillna(0)  
    return z

def compute_clustering(matrix: np.ndarray):
    dist = pdist(matrix, metric=DISTANCE_METRIC)
    link = linkage(dist, method=LINKAGE_METHOD)
    dend = dendrogram(link, no_plot=True)
    return dend["leaves"], link, dend


def dendrogram_traces(dend, orientation, n_leaves, color="rgb(60,60,60)"):
    traces = []
    for xs, ys in zip(dend["icoord"], dend["dcoord"]):
        leaf = [(v - 5.0) / 10.0 for v in xs]
        dist = list(ys)
        px, py = (leaf, dist) if orientation == "top" else ([-d for d in dist], leaf)
        traces.append(go.Scatter(
            x=px, y=py, mode="lines",
            line=dict(color=color, width=1),
            hoverinfo="skip", showlegend=False,
        ))
    return traces

def build_candidate_table(z: pd.DataFrame, threshold: float) -> pd.DataFrame:
    max_abs_z = z.abs().max(axis=1)
    peak_cancer = z.abs().idxmax(axis=1)
    peak_z_signed = [z.loc[g, c] for g, c in zip(z.index, peak_cancer)]

    n_up = (z > threshold).sum(axis=1)
    n_down = (z < -threshold).sum(axis=1)
    n_altered = (z.abs() > threshold).sum(axis=1)
    n_total_cancers = z.shape[1]

    table = pd.DataFrame({
        GENE_COLUMN: z.index,
        
        "peak_dysregulation_zscore": max_abs_z.values,
        "peak_cancer_type": peak_cancer.values,
        "peak_zscore_direction": ["up" if v > 0 else "down" for v in peak_z_signed],
        "num_cancers_overexpressed": n_up.values,
        "num_cancers_underexpressed": n_down.values,
        "num_cancers_altered_total": n_altered.values,
        "pct_of_cancers_altered": (100 * n_altered.values / n_total_cancers).round(1),
    })

    return table 

def main():

    print("Loading data...")
    df = load_data(CSV_PATH)
    print(f"  Loaded {df.shape[0]} genes x {df.shape[1]} cancer types")

    print("Row z-scoring...")
    z = row_zscore(df)

    print("Building candidate ranking table...")
    table = build_candidate_table(z, ALTERED_Z_THRESHOLD)
    table["rank_by_peak_magnitude"] = (
        table["peak_dysregulation_zscore"]
        .rank(ascending=False, method="min").astype(int)
    )
    table["rank_by_breadth"] = (
        table["num_cancers_altered_total"]
        .rank(ascending=False, method="min").astype(int)
    )
    table = table.sort_values(
        "peak_dysregulation_zscore", ascending=False
    ).reset_index(drop=True)
    col_order = [
        GENE_COLUMN, "rank_by_peak_magnitude", "rank_by_breadth", "peak_dysregulation_zscore", "peak_cancer_type",
        "peak_zscore_direction", "num_cancers_altered_total",
        "pct_of_cancers_altered", "num_cancers_overexpressed",
        "num_cancers_underexpressed",
    ]
    table = table[col_order]
    print(f"  Top 5 by peak dysregulation:")
    print(table[[GENE_COLUMN, "peak_dysregulation_zscore", "peak_cancer_type"]].head(5).to_string(index=False))

    # --- Three gene row orderings ---
    order_peak    = table[GENE_COLUMN].tolist()
    order_breadth = (
        table.sort_values("num_cancers_altered_total", ascending=False)
        [GENE_COLUMN].tolist()
    )
    print("Clustering genes (Ward)...")
    row_order, row_link, row_dend = compute_clustering(z.values)
    order_clustered = z.index[row_order].tolist()

    # --- Cancer-type column clustering ---
    print("Clustering cancer types (Ward)...")
    col_order_idx, col_link, col_dend = compute_clustering(z.values.T)
    cancers_ordered = z.columns[col_order_idx].tolist()
    n_cancers       = len(cancers_ordered)

    # --- Pre-compute heatmap matrices for each of the three views ---
    def make_zmatrix(gene_order):
        z_ord   = z.loc[gene_order, cancers_ordered]
        raw_ord = df.loc[gene_order, cancers_ordered]
        rank_mag = dict(zip(table[GENE_COLUMN], table["rank_by_peak_magnitude"]))
        rank_brd = dict(zip(table[GENE_COLUMN], table["rank_by_breadth"]))
        hover = [
            [
                f"Gene: {g}<br>"
                f"Cancer: {c}<br>"
                f"Z-score: {z_ord.iloc[i,j]:.2f}<br>"
                f"Mean log2(TPM+0.001): {raw_ord.iloc[i,j]:.2f}<br>"
                f"Rank (peak): #{rank_mag[g]}  |  Rank (breadth): #{rank_brd[g]}"
                for j, c in enumerate(cancers_ordered)
            ]
            for i, g in enumerate(gene_order)
        ]
        return z_ord.values, gene_order, hover

    print("Pre-computing three row orderings...")
    z_peak,      genes_peak,      hover_peak      = make_zmatrix(order_peak)
    z_breadth,   genes_breadth,   hover_breadth   = make_zmatrix(order_breadth)
    z_clustered, genes_clustered, hover_clustered = make_zmatrix(order_clustered)
    n_genes = len(order_peak)

    # --- Figure layout: column dendrogram (top) + heatmap (bottom) ---
    print("Building interactive figure...")
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.08, 0.92],
        vertical_spacing=0.005,
        specs=[[{"type": "scatter"}], [{"type": "heatmap"}]],
    )

    # Column dendrogram above heatmap
    for tr in dendrogram_traces(col_dend, orientation="top", n_leaves=n_cancers):
        fig.add_trace(tr, row=1, col=1)
    fig.update_xaxes(
        range=[-0.5, n_cancers - 0.5],
        showticklabels=False, showgrid=False, zeroline=False, row=1, col=1,
    )
    fig.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False, row=1, col=1,
    )

    # Heatmap — opens in DEFAULT_VIEW
    view_map = {
        "peak_dysregulation": (z_peak,      genes_peak,      hover_peak),
        "breadth":            (z_breadth,   genes_breadth,   hover_breadth),
        "co_clustered":       (z_clustered, genes_clustered, hover_clustered),
    }
    z_init, genes_init, hover_init = view_map[DEFAULT_VIEW]

    fig.add_trace(go.Heatmap(
        z=z_init,
        x=list(range(n_cancers)),
        y=list(range(n_genes)),
        colorscale="RdYlBu_r",
        zmin=ZMIN, zmax=ZMAX,
        text=hover_init,
        hoverinfo="text",
        colorbar=dict(title="Expression<br>Z-Score", y=0.46, len=0.92),
    ), row=2, col=1)

    fig.update_xaxes(
        tickmode="array", tickvals=list(range(n_cancers)),
        ticktext=cancers_ordered, tickangle=45, tickfont=dict(size=9),
        row=2, col=1,
    )
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(n_genes)),
        ticktext=genes_init,
        autorange="reversed", tickfont=dict(size=7),
        row=2, col=1,
    )

    # Row-ordering toggle buttons
    def make_button(label, z_mat, gene_names, hover_mat):
        return dict(
            label=label,
            method="update",
            args=[
                {"z": [z_mat], "text": [hover_mat]},
                {"yaxis2.ticktext": gene_names,
                 "title.text": f"DNA Repairome Heatmap — {label}"},
            ],
        )

    fig.update_layout(
        title=f"DNA Repairome Heatmap — {DEFAULT_VIEW.replace('_', ' ').title()}",
        width=1500,
        height=max(980, n_genes * 9 + 200),
        margin=dict(l=170, r=160, t=120, b=160),
        plot_bgcolor="white",
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.0, y=1.07,
            xanchor="left", yanchor="top",
            pad=dict(r=10, t=10),
            buttons=[
                make_button("Peak Dysregulation",
                            z_peak,      genes_peak,      hover_peak),
                make_button("Breadth of Alteration",
                            z_breadth,   genes_breadth,   hover_breadth),
                make_button("Co-Clustered (Ward)",
                            z_clustered, genes_clustered, hover_clustered),
            ],
            bgcolor="white",
            bordercolor="rgb(180,180,180)",
            font=dict(size=12),
        )],
        annotations=[dict(
            text="Row order:",
            x=0.0, y=1.095,
            xref="paper", yref="paper",
            showarrow=False, xanchor="left",
            font=dict(size=12),
        )],
    )

    # --- Export HTML ---
    plot_config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "repairome_heatmap",
            "scale": 4,
        },
        "displaylogo": False,
    }
    fig.write_html(OUTPUT_HTML, config=plot_config)
    output_path = Path(OUTPUT_HTML).resolve()
    print(f"Saved interactive heatmap to: {output_path}")

    # --- Open browser ---
    file_uri = output_path.as_uri()
    opened   = webbrowser.open(file_uri)
    if not opened:
        print("\nCould not auto-open browser.")
        print(f"Open manually or paste into browser:\n  {file_uri}")
    else:
        print("Opening in your default web browser...")


if __name__ == "__main__":
    main()
