import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from io import BytesIO
    from PIL import Image
    import base64
    import marimo as mo
    return Image, base64, mo, pd


@app.cell
def _():
    import io
    return (io,)


@app.cell
def _(pd):
    PET_PARQUET = "/media/2nvme/data/oxford-iiit-pet/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(PET_PARQUET)
    return (df,)


@app.cell
def _(mo):
    # Initialize state for spam ids and current page
    spam_ids, set_spam_ids = mo.state([])           # Central list of spam IDs
    current_page, set_current_page = mo.state(0)
    items_per_page = 5
    return (
        current_page,
        items_per_page,
        set_current_page,
        set_spam_ids,
        spam_ids,
    )


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(current_page, df, items_per_page):
    # cell: 1
    # Get current page and slice data
    page = current_page()
    start_idx = page * items_per_page
    end_idx = start_idx + items_per_page
    page_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    return page, page_df


@app.cell
def _():
    return


@app.cell
def _(
    Image,
    base64,
    current_page,
    df,
    io,
    items_per_page,
    mo,
    page,
    page_df,
    set_current_page,
    set_spam_ids,
    spam_ids,
):
    def bytes_to_img_tag(img):
        img_bytes = img['bytes']

        img = Image.open(io.BytesIO(img_bytes))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
    
        return f'<img src="data:image/png;base64,{img_str}" style="max-height: 200px; max-width: 200px; object-fit: contain;">'

    # Add image HTML column
    page_df['image_html'] = page_df['image'].apply(bytes_to_img_tag)

    btns = []
    for _idx, _row in page_df.iterrows():
        _rid = _row["image_id"]
        _btn = mo.ui.button(
            label=_rid,
            on_click=lambda _, r=_rid: set_spam_ids(spam_ids() + [r])
        )
        btns.append(_btn)

    # UI: Display 5 rows with SPAM buttons
    rows_ui = []
    for idx, row in page_df.iterrows():
        rid = row["image_id"]
        i = idx
        row_ui = mo.vstack([
            btns[i],
            mo.Html(f"<strong>ID: {rid}</strong>"),
            mo.Html(row['image_html']),
        ], justify="start", gap="1rem").style(width="100%")

        rows_ui.append(row_ui)

    # Pagination controls
    total_pages = (len(df) + items_per_page - 1) // items_per_page

    prev_btn = mo.ui.button(
        label="Previous",
        disabled=(page == 0),
        on_click=lambda _: print(current_page()) or set_current_page(max(0, current_page() -1)),
    )
    next_btn = mo.ui.button(
        label="Next",
        disabled=(page >= total_pages - 1),
        on_click=lambda _: print(current_page()) or set_current_page(min(total_pages - 1, current_page() + 1)),
    )

    pagination = mo.hstack([prev_btn, mo.md(f"**Page {page + 1} of {total_pages}**"), next_btn])

    return pagination, rows_ui


@app.cell
def _(mo, pagination, rows_ui, spam_ids):
    # Final UI
    mo.vstack([
        mo.md("### Image Review - Mark SPAM"),
        mo.hstack(rows_ui),
        mo.md("---"),
        pagination,
        mo.md(f"**Marked SPAM IDs:** {spam_ids()}")
    ])
    return


@app.cell
def _(set_spam_ids):
    set_spam_ids([])
    return


@app.cell
def _(mo, spam_ids):
    # cell: 2 (optional) - Retrieve spam list later
    mo.md(f"### Final SPAM List\nYou can access it anytime using `spam_ids()`:\n\n```python\n{spam_ids()}\n```")
    return


if __name__ == "__main__":
    app.run()
