import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3    # mpl to d3


# matplotlib 홈페이지 / examples에서 원하는 형태를 찾아본다!
def matplotlib_example():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # Compute pie slices
    N = 20
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = 10 * np.random.rand(N)
    width = np.pi / 4 * np.random.rand(N)
    colors = plt.cm.viridis(radii / 10.)

    ax = plt.subplot(projection='polar')
    ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

    plt.show()


def seaborn_usage():
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create the data
    rs = np.random.RandomState(1979)
    x = rs.randn(500)
    g = np.tile(list("ABCDEFGHIJ"), 50)
    df = pd.DataFrame(dict(x=x, g=g))
    m = df.g.map(ord)
    df["x"] += m

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # plt.show()
    mpld3.show()


def mpld3_usage():
    # 브라우저에 그린다!
    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
    N = 100

    scatter = ax.scatter(np.random.normal(size=N),
                         np.random.normal(size=N),
                         c=np.random.random(size=N),
                         s=1000 * np.random.random(size=N),
                         alpha=0.3,
                         cmap=plt.cm.jet)
    ax.grid(color='white', linestyle='solid')

    ax.set_title("Scatter Plot (with tooltips!)", size=20)

    labels = ['point {0}'.format(i + 1) for i in range(N)]
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)

    mpld3.show()


if __name__ == "__main__":
    # seaborn_usage()
    # mpld3_usage()
    matplotlib_example()
