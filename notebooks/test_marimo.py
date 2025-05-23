import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### **Dataviz Python makeover**
        # When Europeans fly nest
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        *References*
        - mkladv
        - 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <img src="../pics/europeans_parental_household.png" style="height: 500px;"/>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 0. Import libraries
        We begin by setting up the coding environment.
        """
    )
    return


@app.cell
def _():
    # Libraries

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # '%matplotlib inline' command supported automatically in marimo
    import seaborn as sns
    return mcolors, pd, plt, sns


@app.cell
def _(plt, sns):
    # Chart style configuration

    # sns.set_style("ticks")
    sns.reset_orig() # restore all RC params to original settings (respects custom rc)
    plt.rcParams["font.family"] = "Arial"
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'svg'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Import data
        We create the dataset from scratch given the values in the image.
        """
    )
    return


@app.cell
def _():
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
    return (data,)


@app.cell
def _(data, pd):
    df = pd.DataFrame(data=data, columns=['country', 'avg_age_22', 'age_cluster']).sort_values(by='avg_age_22')
    df.head()
    return (df,)


@app.cell
def _(df):
    df.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Barplot
        """
    )
    return


@app.cell
def _(df):
    # We need dedicated colors for different age clusters.
    df.age_cluster.unique()
    return


@app.cell
def _(df, sns):
    # Rather than the color palette used in the chart, we opt for this one:
    palette = sns.color_palette("rocket", n_colors=df.age_cluster.nunique())
    palette
    # (for more color palette: https://www.practicalpythonfordatascience.com/ap_seaborn_palette).
    return (palette,)


@app.cell
def _(mcolors, palette):
    # Get hex codes:
    hex_colors = [mcolors.to_hex(color) for color in palette]
    hex_colors = hex_colors[::-1]
    return (hex_colors,)


@app.cell
def _(df, hex_colors):
    # Add a col to the df:
    dict_age_color = dict(zip(df.age_cluster.unique(), hex_colors))
    df['color'] = df.age_cluster.map(dict_age_color)
    return


@app.cell
def _(df, plt):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Horizontal bar chart
    ax.barh(y=range(len(df)), width=df.avg_age_22, color=df.color, label=df.age_cluster)

    # Update labels on the y-axis
    ax.set_yticks(range(len(df)), labels=df.country, fontsize=10)

    # Axis labelling
    ax.set_xlabel('Average age')

    # Hide axis and ticks on the top and right side
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Enable and customize the x-axis grid
    ax.xaxis.grid(True, linestyle=':')

    # Set title & fig-level subtitle
    # Mid point of left and right x-positions
    fig.suptitle('When Europeans fly nest', fontsize=15, fontweight='bold', x=0.125, y=0.98, horizontalalignment='left', verticalalignment='top')
    ax.set_title('Average age at which young people leave the parent household, 2022.', fontsize=10, loc='left')

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Age groups')

    fig.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
