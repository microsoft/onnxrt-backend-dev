from typing import Any, Dict, List, Optional, Tuple, Union


def memory_peak_plot(
    df1: Union[List[Dict[str, Any]], "pandas.DataFrame"],  # noqa: F821
    key: Union[str, Tuple[str, ...]] = "export",
    suptitle: str = "Memory Peak",
    bars: Optional[Union[float, List[float]]] = None,
    figsize: Tuple[int, int] = (10, 6),
    fontsize: Optional[int] = 6,
) -> "matplotlib.axes.Axes":  # noqa: F821
    """
    Draws a plot showing data coming from a memory profiling.
    See function :func:`onnxrt_backend_dev.monitoring.memory_peak.start_spying_on`.

    :param df1: data
    :param key: used to index figures
    :param subtitle: title for the whole graph
    :param bars: horizontal bars to show thresholds or limits
    :param figsize: figure size
    :param fontsize: font size
    :return: axes

    .. plot::

        import matplotlib.pyplot as plt
        from onnxrt_backend_dev.plotting.data import memory_peak_plot_data
        from onnxrt_backend_dev.plotting.memory import memory_peak_plot

        data = memory_peak_plot_data()
        ax = memory_peak_plot(
            data,
            suptitle="nice",
            bars=[55, 110],
            key=("export", "aot", "compute"),
            figsize=(18 * 2, 7 * 2),
        )
        plt.show()
    """
    import matplotlib.pyplot as plt

    if isinstance(df1, (dict, list)):
        import pandas

        df1 = pandas.DataFrame(df1)

    keys = [key] if isinstance(key, str) else list(key)

    df1 = df1.copy()
    df1["peak-begin"] = df1["peak"] - df1["begin"]
    df1["mean-begin"] = df1["mean"] - df1["begin"]
    if "gpu0_peak" in df1.columns:
        df1["gpu0_peak-begin"] = df1["gpu0_peak"] - df1["gpu0_begin"]
        df1["gpu0_mean-begin"] = df1["gpu0_mean"] - df1["gpu0_begin"]

    fig, ax = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(suptitle)

    dfi = df1[keys + ["peak"]].set_index(keys)
    dfi["peak"].plot.bar(ax=ax[0, 0], title="Memory peak (Mb)", rot=30)
    dfi = df1[keys + ["peak-begin"]].set_index(keys)
    dfi["peak-begin"].plot.bar(
        ax=ax[0, 1], title="Memory peak - memory begin (Mb)", rot=30
    )
    dfi = df1[keys + ["mean-begin"]].set_index(keys)
    dfi["mean-begin"].plot.bar(
        ax=ax[0, 2], title="Memory average - memory begin (Mb)", rot=30
    )

    if "gpu0_peak" in df1.columns:
        dfi = df1[keys + ["gpu0_peak"]].set_index(keys)
        dfi["gpu0_peak"].plot.bar(ax=ax[1, 0], title="GPU Memory peak (Mb)", rot=30)
        dfi = df1[keys + ["gpu0_peak-begin"]].set_index(keys)
        dfi["gpu0_peak-begin"].plot.bar(
            ax=ax[1, 1], title="GPU Memory peak - memory begin (Mb)", rot=30
        )
        dfi = df1[keys + ["gpu0_mean-begin"]].set_index(keys)
        dfi["gpu0_mean-begin"].plot.bar(
            ax=ax[1, 2], title="GPU Memory average - memory begin (Mb)", rot=30
        )
    if bars:
        if isinstance(bars, float):
            bars = [bars]
        n = df1.groupby(keys).count().shape[0]
        for i in range(0, ax.shape[0]):
            for j in range(1, ax.shape[1]):
                for bar in bars:
                    ax[i, j].plot([0, n], [bar, bar], "r--")
    if fontsize:
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].tick_params(axis="both", which="major", labelsize=fontsize)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ls = ax[i, j].get_xticklabels()
            ax[i, j].set_xticklabels(ls, ha="right")
    fig.tight_layout()
    return ax
