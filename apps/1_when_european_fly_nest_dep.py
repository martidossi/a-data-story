import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    # Libraries
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # '%matplotlib inline' command is supported automatically in marimo
    import seaborn as sns
    import altair as alt
    import plotly.express as px

    # plt.rcParams["font.family"] = "Arial"
    return alt, mcolors, mo, pd, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### **Dataviz Python makeover**
    # When Europeans fly nest
    """
    )
    return


@app.cell
def _(mo):
    mo.image(
        src="public/europeans_parental_household.png",
        width=500,
        rounded=True
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This chart, despite its simplicity, contains several questionable design choices and leaves ample room for improvement. It is a bubble chart representing the average age at which young people leave home in various European countries. Each country is depicted by a bubble indicating the related average age, with the bubble color corresponding to an age group, as shown in the legend. See [references](https://www.luzmo.com/blog/bad-data-visualization).

    But what determines the placement of these bubbles? And why are all the circles the same size, even though their underlying values differ?

    The goal of this notebook is to explore alternative visualization approaches that represent this data more effectively and intuitively.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Data
    The first step to better understand and improve the visualization involves the data itself: we start by recreating the dataset from scratch based on the values shown in the image.
    """
    )
    return


@app.cell
def _(pd):
    data = [
        ['Finland', 21.3, 'below 23'],
        ['Estonia', 22.7, 'below 23'],
        ['Sweden', 21.4, 'below 23'],
        ['Denmark', 21.7, 'below 23'],
        ['Netherlands', 23, '23-25'],
        ['Germany', 23.8, '23-25'],
        ['Lithuania', 24.7, '23-25'],
        ['France', 23.4, '23-25'],
        ['Belgium', 26.3, '25-27'],
        ['Ireland', 26.9, '25-27'],
        ['Luxembourg', 26.8, '25-27'],
        ['Latvia', 26.8, '25-27'],
        ['Austria', 25.3, '25-27'],
        ['Czechia', 25.9, '25-27'],
        ['Poland', 28.9, '27-29'],
        ['Hungary', 27.1, '27-29'],
        ['Cyprus', 27.5, '27-29'],
        ['Romania', 27.7, '27-29'],
        ['Slovenia', 29.4, '29-31'],
        ['Greece', 30.7, '29-31'],
        ['Malta', 30.1, '29-31'],
        ['Portugal', 29.7, '29-31'],
        ['Spain', 30.3, '29-31'],
        ['Italy', 30.0, '29-31'],
        ['Bulgaria', 30.3, '29-31'],
        ['Slovakia', 30.8, '29-31'],
        ['Croatia', 33.4, 'over 31'],
    ]

    df = pd.DataFrame(data=data, columns=['country', 'avg_age_22', 'age_cluster']).sort_values(by='avg_age_22')

    df
    return (df,)


@app.cell
def _(df):
    print(f'Number of rows: {df.shape[0]}, number of columns: {df.shape[1]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Barplot
    The easiest yet most effective way to redesign this visualization is by using a bar chart. In addition to changing the visual model, we also update the **color scheme**: rather than using the rainbow palette from the original data visualization, we apply a more perceptually friendly one from the Seaborn library. For the sake of readability, we opt for a horizontal bar chart.
    """
    )
    return


@app.cell
def _(df, mcolors, sns):
    # Rather than the color palette used in the chart, we opt for this one:
    palette = sns.color_palette("rocket", n_colors=df.age_cluster.nunique())
    # (for more color palette: https://www.practicalpythonfordatascience.com/ap_seaborn_palette).

    # Get hex codes:
    hex_colors = [mcolors.to_hex(color) for color in palette][::-1]

    # Add a col to the df:
    dict_age_color = dict(zip(df.age_cluster.unique(), hex_colors))
    df['color'] = df.age_cluster.map(dict_age_color)
    return (palette,)


@app.cell
def _(mo):
    # Dropdown to decide how to sort
    sort_by = mo.ui.dropdown(
        options=["by age", "by country name"],
        value="by age",
        label="How do you want to sort the bars?",
    )
    mo.md(f"### Sort values\n\n{sort_by}")
    return (sort_by,)


@app.cell
def _(alt, df, mcolors, mo, palette, sort_by):
    domain = df.age_cluster.unique()
    range_ = [mcolors.to_hex(color) for color in palette][::-1]
    sort_method = "ascending" if sort_by.value == "by country name" else "-x"

    barh_chart = mo.ui.altair_chart(
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("country:N", title="Country", sort=sort_method),
            x=alt.X("avg_age_22:Q", title="Average age"),
            color=alt.Color("age_cluster:N").scale(domain=domain, range=range_)
        )
        .properties(title="When Europeans fly nest (2022)", height=500),
    )

    barh_chart = barh_chart.configure_title(fontSize=20)
    barh_chart
    return domain, range_


@app.cell
def _(df):
    # Utility cols for charts
    df['idx'] = df.groupby('age_cluster').cumcount().add(1)

    dict_age_cluster_id = dict(zip(df.age_cluster.unique(), range(6)))
    df['age_cluster_id'] = df.age_cluster.map(dict_age_cluster_id)
    df.head()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Bubble chart
    This visualization preserves the original bubble chart model. However, the bubble sizes are so close in value that differences in area are barely noticeable, even though the size is now proportional to the encoded value. Visually, grouping the bubbles by cluster creates mental order and allows for a quick comparison of cluster sizes.
    """
    )
    return


@app.cell
def _(alt, df, domain, mo, range_):
    cluster_labels = df['age_cluster'].unique().tolist() 

    # Build JS expression for label replacement
    label_expr = "datum.value == null ? '' : " + " || ".join(
        [f"(datum.value == {i} ? '{label}' : '')" for i, label in enumerate(cluster_labels)]
    )

    points = mo.ui.altair_chart(
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X(
                "idx:Q", 
                title=None,
                axis=None
            ),
            y=alt.Y(
                "age_cluster_id:Q", 
                axis=alt.Axis(
                    title="Age group",
                    labelExpr=label_expr,
                    tickCount=len(cluster_labels)
                )),
            size=alt.Size(
                "avg_age_22:Q", 
                scale=alt.Scale(type='sqrt', range=[100*df.avg_age_22.min(), 100*df.avg_age_22.max()]), legend=None),
            color=alt.Color("age_cluster:N", legend=None).scale(domain=domain, range=range_),
            tooltip=[
                alt.Tooltip('country:N', title='Country'),
                alt.Tooltip('avg_age_22:Q', title='Avg age'),
            ],
        )
        .properties(
            width=600,
            height=400,
            title="When Europeans fly nest (2022)",
        )
    )

    # Label over bubbles
    text = alt.Chart(df).mark_text(
        dy=0, 
        fontSize=11
    ).encode(
        x="idx:Q", # shift text above the bubble
        y="age_cluster_id:Q",
        text="country:N"
    )

    # Combine and autosize
    scatter_plot = points + text
    scatter_plot = (
        scatter_plot
            .configure_view(
                strokeWidth=0
            ).configure_title(
                fontSize=20)
            .properties(
                autosize=alt.AutoSizeParams(
                type="fit",  # or "fit-x", "pad"
                contains="padding"
                )
            )
    )

    scatter_plot
    return


if __name__ == "__main__":
    app.run()
