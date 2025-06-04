import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    # Libraries
    import marimo as mo
    import pandas as pd

    # dataviz
    import seaborn as sns
    import altair as alt
    import plotly.express as px
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # '%matplotlib inline' command is supported automatically in marimo

    # plt.rcParams["font.family"] = "Arial"
    return alt, mcolors, mo, pd, px, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### **A Python Dataviz Makeover Series**
    ---
    # Episode 1 – **When Europeans fly nest**

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="public/europeans_parental_household.png",
        width=500,
        rounded=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Despite its simplicity and overall ease of readability, this chart presents a few questionable design choices that could be revisited to enhance clarity and effectiveness (_e.g., what drives the positioning of the bubbles? Why do all the circles share the same size, even though their underlying values differ? Is the chosen colormap truly effective in this context?_). It is a bubble chart representing the average age at which young people leave home in European countries. Each country is depicted by a bubble indicating the related average age, with the bubble color corresponding to an age group, as shown in the legend. See [references](#references).

    🔍 **The goal** of this notebook is to explore alternative visualization approaches that represent the same data more effectively and intuitively, eventually with more contextual information.

    💻 **How**: everything is developed in Python. You're currently in a [Marimo](https://marimo.io/) notebook, an interactive and lightweight Python environment designed for enhanced data exploration and storytelling. Depending on the specific need, visualizations are built using [Altair](https://altair-viz.github.io/) or [Plotly](https://plotly.com/python/).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Index

    - [1. Data](#1-data)
    - [2. Bar plot](#2-bar-plot)
    - [3. Bubble chart](#3-bubble-chart)
    - [4. Choropleth map](#4-choropleth-map)
    - [5. Adding contextual data](#5-adding-contextual-data)
    - [References](#references)
    ----
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Data
    The first step to better understand and improve the visualization involves the data itself: we start by recreating the dataset from scratch based on the values shown in the image.

    🔢 **How is this metric calculated?** We're looking at the _estimated average age at which young people leave their parental home_ across European countries. According to Eurostat, this represents the age at which 50% of the population no longer live in a household with their parent(s). (See [references](#references) for more details.)
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
    The easiest yet most effective way to redesign this visualization is by using a bar chart. In addition to changing the visual model, we also update the **color scheme**: rather than using the rainbow palette from the original data visualization, we apply a more perceptually friendly one from the Seaborn library, named `magma`. For the sake of readability, we also opt for a horizontal bar chart.
    """
    )
    return


@app.cell
def _(df, mcolors, sns):
    # Rather than the color palette used in the chart, we use this one:
    palette = sns.color_palette("rocket", n_colors=df.age_cluster.nunique())
    # (for more color palette: https://www.practicalpythonfordatascience.com/ap_seaborn_palette).

    # Get hex codes:
    hex_colors = [mcolors.to_hex(color) for color in palette]

    # Add a color column to the df:
    dict_age_color = dict(zip(df.age_cluster.unique(), hex_colors))
    df['color'] = df.age_cluster.map(dict_age_color)
    return dict_age_color, palette


@app.cell
def _(mo):
    # Dropdown to decide how to sort:
    sort_by = mo.ui.dropdown(
        options=["by age", "by country name"],
        value="by age",
        label="How do you want to sort the bars?",
    )
    mo.md(f"### Sort values\n\n{sort_by}")
    return (sort_by,)


@app.cell
def _(alt, df, mcolors, mo, palette, pd, sort_by):
    domain = df.age_cluster.unique()
    # Color list in reverse order for intuitive color mapping:
    range_ = [mcolors.to_hex(color) for color in palette][::-1]
    sort_method = "ascending" if sort_by.value == "by country name" else "-x"

    bar = (
        alt.Chart(df)
            .mark_bar(opacity=0.85)
            .encode(
                y=alt.Y("country:N", title="Country", sort=sort_method),
                x=alt.X("avg_age_22:Q", title="Average age"),
                color=alt.Color("age_cluster:N").scale(domain=domain, range=range_),
                tooltip=[alt.Tooltip('avg_age_22:Q', title='Avg age')]
            )
            .properties(
                title="When Europeans fly nest (2022)", 
                height=500
            )
    )

    # Vertical line at x = 26.4 (avg EU value across the 27 countries)
    # This value is not in the dataset, but we know it from the chart
    avg_val = 26.4
    df_avg_val_22 = pd.DataFrame({'x': [avg_val], 'label': ['avg EU value']})

    vline_label = alt.layer(
        alt.Chart(df_avg_val_22)
            .mark_rule(color='black', strokeDash=[4, 2])
            .encode(x='x:Q'),
        alt.Chart(df_avg_val_22)
            .mark_text(align='left', dx=5, dy=475, fontSize=12)
            .encode(
                x='x:Q',
                y=alt.value(0),
                text='label:N'
            )
    )

    # Combine all layers
    barh_chart = (
        mo.ui.altair_chart(bar + vline_label)
            .properties(title="When Europeans fly nest (2022)", height=500)
            .configure_title(fontSize=20)
    )

    barh_chart
    return domain, range_


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Bubble chart
    This other visualization preserves the original bubble chart idea. Visually, grouping the bubbles by cluster creates mental order and allows for a quick comparison of cluster sizes. Even though each bubble's size is now proportional to its encoded value, the differences are so small that variations in area are barely perceptible.
    """
    )
    return


@app.cell
def _(df):
    # We add a few new columns that are functional to the creation of this chart
    df['idx'] = df.groupby('age_cluster').cumcount().add(1)

    dict_age_cluster_id = dict(zip(df.age_cluster.unique(), range(6)))
    df['age_cluster_id'] = df.age_cluster.map(dict_age_cluster_id)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Choroplet map""")
    return


@app.cell
def _(df, dict_age_color, mo, px):
    fig = px.choropleth(
        df,
        locations='country',
        locationmode='country names',
        color='age_cluster',
        scope='europe',
        color_discrete_map=dict_age_color,
        title='When Europeans fly nest (2022)',
        hover_data={'country': True, 'avg_age_22': True}
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showframe=True,
        showcoastlines=True,
        projection_type='natural earth'
    )

    fig.update_layout(
        title_x=0.5,  # centers the title
        height=400,
        margin={"r":0, "t": 50, "l": 0, "b": 0},
        legend_title_text="Age group"
    )

    fig.update_traces(
        hovertemplate='<b>%{location}</b><br>Avg age: %{customdata[1]}<extra></extra>'
    )

    mo.ui.plotly(fig)
    return


@app.cell(hide_code=True)
def _():
    ## 5. Add context
    return


@app.cell
def _(pd):
    df_context = pd.read_csv('public/yth_demo.csv', sep=';')
    df_context = df_context.rename(columns={'Country': 'country'})
    df_context
    return (df_context,)


@app.cell
def _(df_context):
    df_context_male = df_context[df_context.Sex=='Males']
    df_context_female = df_context[df_context.Sex=='Females']
    df_context_total = df_context[df_context.Sex=='Total']
    return df_context_female, df_context_male


@app.cell
def _(df, df_context_female, df_context_male):
    df_enriched = df.merge(df_context_male[['country', '2022']], on='country')
    df_enriched = df_enriched.rename(columns={'2022': 'male_2022'})

    df_enriched = df_enriched.merge(df_context_female[['country', '2022']], on='country')
    df_enriched = df_enriched.rename(columns={'2022': 'female_2022'})
    return


@app.cell
def _(df_context_female, df_context_male, pd):
    df_plot_dot = (
        pd.concat(
            [
                df_context_male[['country', 'Sex', '2022']],
                df_context_female[['country', 'Sex', '2022']]
            ]
        )
        .rename(columns={'Sex': 'sex', '2022': 'avg_age_22'})
    )
    df_plot_dot = df_plot_dot[df_plot_dot.country!='European Union - 27 countries (from 2020)']
    return (df_plot_dot,)


@app.cell
def _(alt, df_plot_dot, mo):
    # Chart base - Ranged Dot Plot
    chart_dot = (
        alt.Chart(df_plot_dot)
            .encode(
                x=alt.X("avg_age_22:Q", title="Average age", scale=alt.Scale(domain=[18, 38])),
                y=alt.Y("country:N")
            )
    )

    lines_dot = (
        chart_dot.mark_line(color="black")
            .encode(detail="country:N")
    )

    color_female_male = (
        alt.Color("sex:N")
            .scale(domain=['Females', 'Males'], range=["#c83f49", "#003d80"])
    )

    points_dot = (
        chart_dot.mark_point(
            size=100,
            opacity=1,
            filled=True,
        )
            .encode(color=color_female_male)
    )

    # Combine all layers
    ranged_dot_plot = (
        mo.ui.altair_chart(lines_dot + points_dot)
            .properties(title="When Europeans fly nest (2022)", height=500)
            .configure_title(fontSize=20)
    )

    ranged_dot_plot
    return


@app.cell(hide_code=True)
def _():
    ## References
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The dataset ::lucide:arrow-down-right:: [URL](https://ec.europa.eu/eurostat/databrowser/view/yth_demo_030/default/table?lang=en)

    - The viz source ::lucide:arrow-down-right:: [URL](https://www.luzmo.com/blog/bad-data-visualization)

    - Related article with data ::lucide:arrow-down-right:: [URL](https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20230904-1#:~:text=In%202022%2C%20young%20people%20across,average%20varied%20among%20EU%20countries)

    - GitHub repo ::lucide:arrow-down-right:: [URL](https://github.com/martidossi/data_viz_makeover)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Thanks for following along, see you in the next one! 🐱

    💬 Feel free to get in touch if you have any questions —the project is still in its early stages, and I’d really value your feedback! 🫶

    - _Do you have any thoughts on the structure so far?_
    - _How would you visualize this data, do you have alternative chart suggestions?_
    - _Are there other interesting use cases you’d like me to explore or revisit in one of the next notebooks?_
    """
    )
    return


if __name__ == "__main__":
    app.run()
