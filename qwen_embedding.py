import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    from sentence_transformers import SentenceTransformer
    import torch
    return SentenceTransformer, torch


@app.cell
def _():
    import pymde
    import altair as alt
    return alt, pymde


@app.cell
def _():
    import pymupdf.layout  # Activates the layout feature in PyMuPDF
    import pymupdf4llm
    return (pymupdf4llm,)


@app.cell
def _():
    import pathlib as pl
    import glob

    pdf_dir = pl.Path('/media/data/paper')
    pdf_files = glob.glob(str(pdf_dir / "**/*.pdf"))
    return (pdf_files,)


@app.cell
def _():
    from collections import Counter

    def top_ngrams(text, n=2, top_k=10):
        """
        Return the top-K most frequent n-grams from a given text.
        """
        # basic tokenization — split on whitespace
        tokens = text.split()
    
        # build n-grams
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams = [' '.join(ng) for ng in ngrams]

        # count & return most common
        return Counter(ngrams).most_common(top_k)

    return (top_ngrams,)


@app.cell
def _(pdf_files, pymupdf4llm):
    pdf_txts = []
    for p in pdf_files:
        try:
            res = pymupdf4llm.to_markdown(p)
        except Exception as exc:
            print(exc)
            res = None
        pdf_txts.append(res)
    return (pdf_txts,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pdf_files):
    len(pdf_files)
    return


@app.cell
def _():
    from tqdm import tqdm
    return (tqdm,)


@app.cell
def _(pdf_files, pdf_txts, top_ngrams, tqdm):
    pdf_dicts = []
    for _p, txt in tqdm(zip(pdf_files, pdf_txts)):
        if txt is not None and txt:
            top5tri = ",".join([ss for (ss, cnt) in top_ngrams(txt, n=3, top_k=5)])
            pdf_dicts.append(dict(txt=txt,path=_p, top5tri=top5tri))
    return (pdf_dicts,)


@app.cell
def _(df, model):
    pdf_embeddings = model.encode([txt[:3200] for txt in df['txt'].tolist()])
    return (pdf_embeddings,)


@app.cell
def _(pd, pdf_dicts):
    pdf = pd.DataFrame(pdf_dicts)
    return (pdf,)


@app.cell
def _(pdf_embeddings):
    pdf_embeddings[:].shape
    return


@app.cell
def _(pdf, pdf_embeddings):
    pdf['embeddings'] = list(pdf_embeddings)
    return


@app.cell
def _(pdf):
    pdf
    return


@app.cell
def _(np, pdf):
    X = np.stack(pdf['embeddings'].values)
    return (X,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(pdf):
    pdf
    return


@app.cell
def _(X, pdf):
    from sklearn.cluster import KMeans

    n_clusters = 32
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pdf['cluster'] = kmeans.fit_predict(X)
    return


@app.cell
def _(X, pdf):
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)

    pdf['umap_1'] = X_umap[:, 0]
    pdf['umap_2'] = X_umap[:, 1]

    return


@app.cell
def _(X, pdf):
    from sklearn.manifold import TSNE

    # Run t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)  # adjust perplexity as needed
    X_tsne = tsne.fit_transform(X)

    # Add t-SNE coordinates to DataFrame
    pdf['tsne_1'] = X_tsne[:, 0]
    pdf['tsne_2'] = X_tsne[:, 1]

    return


@app.cell
def _(pdf):
    pdf
    return


@app.cell
def _(pdf):
    pdf.to_parquet("demo.parquet")
    return


@app.cell
def _(alt, mo, pdf):
    chart = mo.ui.altair_chart(
        alt.Chart(pdf.drop(columns=['embeddings']))
        .mark_circle(size=32)
        .encode(
            # x=alt.X("umap_1:Q").scale(domain=(4, 10)),
            # y=alt.Y("umap_2:Q").scale(domain=(0, 10)),
            x='tsne_1',
            y='tsne_2', 
            color=alt.Color("cluster:N"),
        )
        .properties(width=500, height=500),
        chart_selection="interval",
    )
    chart
    return (chart,)


@app.cell
def _(table):
    table
    return


@app.cell
def _(chart, mo, table):
    # mo.stop() prevents this cell from running if the chart has
    # no selection
    mo.stop(not len(chart.value))
    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table
    selected_top5tri = (
        "\n\n".join(chart.value["top5tri"])
        if not len(table.value)
        else  "\n\n".join(table.value["top5tri"])
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:
        {mo.as_html(selected_top5tri)}
        """
    )
    return


@app.cell
def _(chart, mo):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell
def _(mo, pdf, pymde, torch):
    def compute_embedding(embedding_dim, constraint):
        mo.output.append(
            mo.md("Your embedding is being computed ... hang tight!").callout(kind="warn")
        )

        mde = pymde.preserve_neighbors(
            pdf['embeddings'],
            embedding_dim=embedding_dim,
            constraint=constraint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
        )
    
        X = mde.embed(verbose=True)
        mo.output.clear()
    
        return X
    return


@app.cell
def _(pd, pdf_dicts):

    df = pd.DataFrame(pdf_dicts)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    return


@app.cell
def _(document_embeddings):
    document_embeddings.shape
    return


@app.cell
def _(SentenceTransformer, torch):
    # Load the model
    # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    # together with setting `padding_side` to "left":
    model = SentenceTransformer(
        '/media/2nvme/llm/Qwen3-Embedding-0.6B',
        model_kwargs={
            "torch_dtype": torch.float16,             # ← force FP16
            "attn_implementation": "flash_attention_2",
            "device_map": "auto"
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    return (model,)


@app.cell
def _(model):

    # The queries and documents to embed
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    # Encode the queries and documents. Note that queries benefit from using a prompt
    # Here we use the prompt called "query" stored under `model.prompts`, but you can
    # also pass your own prompt via the `prompt` argument
    query_embeddings = model.encode(queries, prompt_name="query")
    document_embeddings = model.encode(documents)

    # Compute the (cosine) similarity between the query and document embeddings
    similarity = model.similarity(query_embeddings, document_embeddings)
    print(similarity)
    # tensor([[0.7646, 0.1414],
    #         [0.1355, 0.6000]])
    return (document_embeddings,)


@app.cell
def _():
    def _():
        import altair as alt
        import pandas as pd
        from sklearn.datasets import make_blobs

        # --- Create some sample data ---
        X, y = make_blobs(n_samples=200, centers=4, random_state=42, cluster_std=1.2)
        df = pd.DataFrame(X, columns=["x", "y"])
        df["cluster"] = y.astype(str)

        # --- Base scatter plot ---
        points = alt.Chart(df).mark_circle(size=60).encode(
            x="x",
            y="y",
            color="cluster:N",
            tooltip=["x", "y", "cluster"]
        )

        # --- Compute cluster centroids for labeling ---
        centroids = df.groupby("cluster")[["x", "y"]].mean().reset_index()

        # --- Text labels for clusters ---
        text = alt.Chart(centroids).mark_text(
            align="center",
            baseline="middle",
            fontSize=34,
            fontWeight="bold",
            dy=-10,  # shift label upward
        ).encode(
            x="x",
            y="y",
            text="cluster"
        )

        # --- Combine both layers ---
        chart = points + text
        return chart.properties(
            title="Clustered Scatter Plot with Labels",
            width=500,
            height=400
        )


    _()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sklearn
    import sklearn.datasets
    import sklearn.manifold

    raw_digits, raw_labels = sklearn.datasets.load_digits(return_X_y=True)
    return raw_digits, sklearn


@app.cell
def _(raw_digits):
    raw_digits.shape
    return


@app.cell
def _(raw_digits, sklearn):
    X_embedded = sklearn.decomposition.PCA(
        n_components=3, whiten=True
    ).fit_transform(raw_digits)
    return (X_embedded,)


@app.cell
def _(X_embedded):
    X_embedded.shape
    return


if __name__ == "__main__":
    app.run()
