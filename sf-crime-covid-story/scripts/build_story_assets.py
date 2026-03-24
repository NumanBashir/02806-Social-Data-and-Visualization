from __future__ import annotations

from pathlib import Path
import json
import os

MPL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".mpl-cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pyarrow.csv as csv
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[0]
SOURCE_CSV = PROJECT_ROOT / "Week1" / "Police_Department_Incident_Reports__2018_to_Present_20260203.csv"
CACHE_PARQUET = ROOT / "data" / "crime_2019_2021.parquet"
DISTRICT_GEOJSON = PROJECT_ROOT / "Week5" / "sfpd.geojson"

STATIC_OUT = ROOT / "visualizations" / "static-chart" / "monthly-crime-trend.png"
MAP_OUT = ROOT / "visualizations" / "map" / "district-pandemic-shift.html"
INTERACTIVE_OUT = ROOT / "visualizations" / "interactive" / "category-shift-heatmap.html"

PALETTE = {
    "ink": "#1f2a33",
    "blue": "#295c7a",
    "teal": "#12756b",
    "rust": "#c65d3f",
    "gold": "#d8a047",
    "sand": "#efe3d3",
    "paper": "#fbf7f1",
    "grid": "#d9d0c3",
}


def ensure_cache() -> None:
    if CACHE_PARQUET.exists():
        return

    read_options = csv.ReadOptions(use_threads=True)
    convert_options = csv.ConvertOptions(
        include_columns=[
            "Incident Datetime",
            "Incident Category",
            "Incident Subcategory",
            "Police District",
            "Analysis Neighborhood",
            "Latitude",
            "Longitude",
        ]
    )
    table = csv.read_csv(
        SOURCE_CSV,
        read_options=read_options,
        convert_options=convert_options,
    )
    dt_col = table["Incident Datetime"]
    mask = pc.and_(
        pc.greater_equal(dt_col, "2019/01/01 12:00:00 AM"),
        pc.less(dt_col, "2022/01/02 12:00:00 AM"),
    )
    subset = table.filter(mask)
    CACHE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, CACHE_PARQUET)


def load_data() -> pd.DataFrame:
    ensure_cache()
    df = pd.read_parquet(CACHE_PARQUET)
    df["dt"] = pd.to_datetime(
        df["Incident Datetime"],
        format="%Y/%m/%d %I:%M:%S %p",
        errors="coerce",
    )
    df = df.dropna(subset=["dt"]).copy()
    df = df[(df["dt"] >= "2019-01-01") & (df["dt"] < "2022-01-01")].copy()
    df["year"] = df["dt"].dt.year
    df["month_start"] = df["dt"].dt.to_period("M").dt.to_timestamp()
    df["district_upper"] = df["Police District"].fillna("Unknown").str.upper()
    return df


def build_static_chart(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby("month_start")
        .size()
        .rename("incidents")
        .reset_index()
        .sort_values("month_start")
    )

    april_2019 = int(monthly.loc[monthly["month_start"] == pd.Timestamp("2019-04-01"), "incidents"].iloc[0])
    april_2020 = int(monthly.loc[monthly["month_start"] == pd.Timestamp("2020-04-01"), "incidents"].iloc[0])
    drop_pct = (april_2020 - april_2019) / april_2019 * 100

    fig, ax = plt.subplots(figsize=(13, 7), facecolor=PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])

    ax.plot(
        monthly["month_start"],
        monthly["incidents"],
        color=PALETTE["blue"],
        linewidth=3,
        solid_capstyle="round",
    )
    ax.fill_between(
        monthly["month_start"],
        monthly["incidents"],
        color=PALETTE["blue"],
        alpha=0.08,
    )

    stay_home = pd.Timestamp("2020-03-17")
    ax.axvline(stay_home, color=PALETTE["rust"], linewidth=2.5, linestyle=(0, (6, 6)))
    ax.axvspan(pd.Timestamp("2020-03-17"), pd.Timestamp("2020-06-15"), color=PALETTE["sand"], alpha=0.55)

    nadir_date = pd.Timestamp("2020-04-01")
    ax.scatter([nadir_date], [april_2020], s=95, color=PALETTE["rust"], zorder=5)
    ax.annotate(
        "Stay Home order\nstarts March 17, 2020",
        xy=(stay_home, 10800),
        xytext=(pd.Timestamp("2019-10-01"), 12850),
        fontsize=11,
        color=PALETTE["ink"],
        arrowprops={"arrowstyle": "-", "color": PALETTE["rust"], "lw": 1.5},
    )
    ax.annotate(
        f"April 2020 nadir:\n{april_2020:,} incidents\n({drop_pct:.0f}% vs. April 2019)",
        xy=(nadir_date, april_2020),
        xytext=(pd.Timestamp("2020-08-01"), 8600),
        fontsize=11,
        color=PALETTE["ink"],
        arrowprops={"arrowstyle": "->", "color": PALETTE["rust"], "lw": 1.6},
    )

    ax.set_title(
        "Reported incidents fell sharply as San Francisco shut down",
        loc="left",
        fontsize=22,
        color=PALETTE["ink"],
        pad=18,
        fontweight="bold",
    )
    ax.text(
        0,
        1.02,
        "Monthly SFPD incident reports, January 2019 to December 2021",
        transform=ax.transAxes,
        fontsize=12,
        color="#5b6570",
    )

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=1, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_ylabel("Incidents per month", color=PALETTE["ink"], fontsize=12)
    ax.set_xlabel("")
    ax.tick_params(axis="x", colors="#55606a")
    ax.tick_params(axis="y", colors="#55606a")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_ylim(7000, 13500)

    fig.tight_layout()
    STATIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STATIC_OUT, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def district_comparison(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[(df["dt"] >= "2019-03-17") & (df["dt"] < "2019-09-01")]
    during = df[(df["dt"] >= "2020-03-17") & (df["dt"] < "2020-09-01")]

    comp = pd.concat(
        [
            pre.groupby("district_upper").size().rename("incidents_2019"),
            during.groupby("district_upper").size().rename("incidents_2020"),
        ],
        axis=1,
    ).fillna(0)
    comp["pct_change"] = (comp["incidents_2020"] - comp["incidents_2019"]) / comp["incidents_2019"] * 100
    comp = comp.reset_index().rename(columns={"district_upper": "district"})
    return comp


def build_map(df: pd.DataFrame) -> None:
    comp = district_comparison(df)
    comp = comp[comp["district"] != "OUT OF SF"].copy()

    fig = px.choropleth(
        comp,
        geojson=json.loads(DISTRICT_GEOJSON.read_text()),
        locations="district",
        featureidkey="properties.DISTRICT",
        color="pct_change",
        color_continuous_scale=[
            [0.0, "#8f2d21"],
            [0.45, "#d96d48"],
            [0.8, "#f0d9b9"],
            [1.0, "#f7f3ed"],
        ],
        range_color=(-55, 0),
        hover_data={"incidents_2019": ":,.0f", "incidents_2020": ":,.0f", "pct_change": ":.1f"},
    )

    fig.update_traces(
        marker_line_color="#fbf7f1",
        marker_line_width=1.6,
        hovertemplate=(
            "<b>%{location}</b><br>"
            "2019 window: %{customdata[0]:,.0f}<br>"
            "2020 window: %{customdata[1]:,.0f}<br>"
            "Change: %{z:.1f}%<extra></extra>"
        ),
    )
    fig.update_geos(fitbounds="locations", visible=False, bgcolor=PALETTE["paper"])
    fig.update_layout(
        paper_bgcolor=PALETTE["paper"],
        plot_bgcolor=PALETTE["paper"],
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        title={
            "text": "Downtown districts saw the largest reporting drops",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 24, "color": PALETTE["ink"]},
        },
        font={"family": "Arial, sans-serif", "color": PALETTE["ink"]},
        coloraxis_colorbar={
            "title": "% change",
            "tickvals": [-50, -40, -30, -20, -10, 0],
            "ticksuffix": "%",
        },
    )

    MAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(MAP_OUT, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": False})


def build_interactive_heatmap(df: pd.DataFrame) -> None:
    pre = df[(df["dt"] >= "2019-03-17") & (df["dt"] < "2019-09-01")]
    during = df[(df["dt"] >= "2020-03-17") & (df["dt"] < "2020-09-01")]

    district_order = (
        district_comparison(df)
        .query("district != 'OUT OF SF'")
        .sort_values("pct_change")
        ["district"]
        .tolist()
    )

    citywide = pd.concat(
        [
            pre.groupby("Incident Category").size().rename("pre"),
            during.groupby("Incident Category").size().rename("during"),
        ],
        axis=1,
    ).fillna(0)
    citywide["net"] = citywide["during"] - citywide["pre"]
    citywide["pct_change"] = citywide["net"] / citywide["pre"] * 100

    category_order = [
        category
        for category in citywide.sort_values("pre", ascending=False).index.tolist()
        if category in {
            "Larceny Theft",
            "Burglary",
            "Motor Vehicle Theft",
            "Recovered Vehicle",
            "Lost Property",
            "Assault",
            "Drug Offense",
            "Fraud",
            "Warrant",
            "Malicious Mischief",
        }
    ]
    category_order = sorted(category_order, key=lambda category: citywide.loc[category, "pct_change"])

    rows = []
    for district in district_order:
        for category in category_order:
            pre_count = int(((pre["district_upper"] == district) & (pre["Incident Category"] == category)).sum())
            during_count = int(((during["district_upper"] == district) & (during["Incident Category"] == category)).sum())
            pct = None if pre_count == 0 else (during_count - pre_count) / pre_count * 100
            rows.append(
                {
                    "district": district,
                    "category": category,
                    "incidents_2019": pre_count,
                    "incidents_2020": during_count,
                    "abs_change": during_count - pre_count,
                    "pct_change": pct,
                }
            )

    heat = pd.DataFrame(rows)
    heat = heat[heat["district"] != "OUT OF SF"].copy()

    pct_matrix = (
        heat.pivot(index="category", columns="district", values="pct_change")
        .reindex(index=category_order, columns=district_order)
    )
    abs_matrix = (
        heat.pivot(index="category", columns="district", values="abs_change")
        .reindex(index=category_order, columns=district_order)
    )
    count_matrix = (
        heat.pivot(index="category", columns="district", values="incidents_2020")
        .reindex(index=category_order, columns=district_order)
    )
    pre_matrix = (
        heat.pivot(index="category", columns="district", values="incidents_2019")
        .reindex(index=category_order, columns=district_order)
    )

    custom = []
    for category in category_order:
        row = []
        for district in district_order:
            row.append(
                [
                    int(pre_matrix.loc[category, district]),
                    int(count_matrix.loc[category, district]),
                    int(abs_matrix.loc[category, district]),
                ]
            )
        custom.append(row)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=pct_matrix.values,
            x=district_order,
            y=category_order,
            customdata=custom,
            visible=True,
            zmid=0,
            zmin=-80,
            zmax=150,
            colorscale=[
                [0.0, "#8f2d21"],
                [0.45, "#dc7d59"],
                [0.5, "#f8f4ed"],
                [0.7, "#80c7b3"],
                [1.0, "#12756b"],
            ],
            colorbar={"title": "% change", "ticksuffix": "%"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "District: %{x}<br>"
                "2019 window: %{customdata[0]:,}<br>"
                "2020 window: %{customdata[1]:,}<br>"
                "Absolute change: %{customdata[2]:+,}<br>"
                "Percent change: %{z:.1f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=abs_matrix.values,
            x=district_order,
            y=category_order,
            customdata=custom,
            visible=False,
            zmid=0,
            colorscale=[
                [0.0, "#8f2d21"],
                [0.45, "#dc7d59"],
                [0.5, "#f8f4ed"],
                [0.7, "#80c7b3"],
                [1.0, "#12756b"],
            ],
            colorbar={"title": "Net change"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "District: %{x}<br>"
                "2019 window: %{customdata[0]:,}<br>"
                "2020 window: %{customdata[1]:,}<br>"
                "Absolute change: %{z:+,}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=count_matrix.values,
            x=district_order,
            y=category_order,
            customdata=custom,
            visible=False,
            colorscale=[
                [0.0, "#f4e8d7"],
                [0.35, "#d8a047"],
                [0.7, "#628aa6"],
                [1.0, "#244d66"],
            ],
            colorbar={"title": "2020 incidents"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "District: %{x}<br>"
                "2020 window incidents: %{z:,}<br>"
                "2019 window incidents: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor=PALETTE["paper"],
        plot_bgcolor=PALETTE["paper"],
        margin={"l": 20, "r": 20, "t": 110, "b": 30},
        title={
            "text": "The offense mix changed as daily routines moved",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 24, "color": PALETTE["ink"]},
        },
        font={"family": "Arial, sans-serif", "color": PALETTE["ink"]},
        xaxis={"side": "top", "tickangle": -35},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.03,
                "y": 1.16,
                "showactive": True,
                "buttons": [
                    {"label": "Percent change", "method": "update", "args": [{"visible": [True, False, False]}]},
                    {"label": "Net change", "method": "update", "args": [{"visible": [False, True, False]}]},
                    {"label": "2020 counts", "method": "update", "args": [{"visible": [False, False, True]}]},
                ],
            }
        ],
        annotations=[
            {
                "text": "Compare March 17 to August 31, 2020 with the same 2019 window. Hover for exact counts.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.03,
                "y": 1.08,
                "showarrow": False,
                "font": {"size": 12, "color": "#5b6570"},
            }
        ],
    )

    INTERACTIVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(INTERACTIVE_OUT, include_plotlyjs="cdn", config={"responsive": True})


def main() -> None:
    df = load_data()
    build_static_chart(df)
    build_map(df)
    build_interactive_heatmap(df)
    print(f"Created {STATIC_OUT.relative_to(PROJECT_ROOT)}")
    print(f"Created {MAP_OUT.relative_to(PROJECT_ROOT)}")
    print(f"Created {INTERACTIVE_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
