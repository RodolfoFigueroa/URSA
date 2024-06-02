import geemap

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import ursa.ghsl as ghsl
import ursa.translations as ut
import ursa.utils as utils

from PIL import Image, ImageOps


HEIGHT = 600
HIGH_RES = True

URL_BUILT = "https://doi.org/10.2905/D07D81B4-7680-4D28-B896-583745C27085"
URL_POP = "https://doi.org/10.2905/D6D86A90-4351-4508-99C1-CB074B022C4A"
URL_SMOD = "https://doi.org/10.2905/4606D58A-DC08-463C-86A9-D49EF461C47F"

YEARS = [
    "1975",
    "1980",
    "1985",
    "1990",
    "1995",
    "2000",
    "2005",
    "2010",
    "2015",
    "2020",
]
YEARS_UINT8 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="uint8")


def _update_figure(fig, centroid_mollweide, smod, language):
    smod_p = ghsl.smod_polygons(smod, centroid_mollweide)
    clusters_2020 = smod_p[(smod_p.year == 2020) & (smod_p["class"] == 2)]
    clusters_2020 = clusters_2020.to_crs(4326)

    n_mains = 0
    n_other = 0
    for _, row in clusters_2020.iterrows():
        if row.is_main:
            name = ut.central_zone[language]
            n_mains += 1
        else:
            name = ut.peripheral_zones[language]
            n_other += 1

        linestring = row.geometry.exterior
        x, y = linestring.xy
        p_df = pd.DataFrame({"lats": y, "lons": x})
        p_fig = px.line_mapbox(
            p_df,
            lat="lats",
            lon="lons",
            color=[name] * len(x),
            color_discrete_map={
                ut.central_zone[language]: "maroon",
                ut.peripheral_zones[language]: "orange",
            },
        )

        p_fig.update_traces(hovertemplate=None, hoverinfo="skip")

        if row.is_main and n_mains > 1:
            p_fig.update_traces(showlegend=False)
        if not row.is_main and n_other > 1:
            p_fig.update_traces(showlegend=False)

        fig.add_traces(p_fig.data)


def _add_bbox_trace(fig, bbox_mollweide, language):
    bbox_temp = (
        gpd.GeoDataFrame({"geometry": bbox_mollweide}, index=[0], crs="ESRI:54009")
        .to_crs(4326)
        .geometry.iloc[0]
    )
    x, y = bbox_temp.exterior.xy
    p_df = pd.DataFrame({"lats": y, "lons": x})
    p_fig = px.line_mapbox(
        p_df,
        lat="lats",
        lon="lons",
        color=[ut.analysis_zone[language]] * len(x),
        color_discrete_map={ut.analysis_zone[language]: "blue"},
    )
    p_fig.update_traces(hovertemplate=None, hoverinfo="skip")
    fig.add_traces(p_fig.data)


def _resize_image(img):
    return img.resize([hw * 10 for hw in img.size], resample=Image.Resampling.NEAREST)


def plot_built_poly(built_gdf, bbox_latlon, year=2020):
    """Plots a map with built information for year with polygons.
    May be slow and memory heavy."""

    west, south, east, north = bbox_latlon.bounds

    Map = geemap.Map()

    gdf = built_gdf[built_gdf.year == year].to_crs(4326).reset_index(drop=True)
    gdf["id"] = list(gdf.index)
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry,
        color="fraction",
        locations="id",
        color_continuous_scale="viridis",
        hover_data={"fraction": True, "id": False},
        opacity=0.5,
    )
    fig.update_traces(marker_line_width=0)
    Map.add_traces(fig.data)

    Map.update_layout(
        mapbox_bounds={"west": west, "east": east, "south": south, "north": north},
        height=HEIGHT,
    )

    return Map


def _built_to_bin(built, thresh):
    resolution = built.rio.resolution()
    pixel_area = abs(np.prod(resolution))

    # Create a density array
    # Only densities can be safely reprojected
    built = built / pixel_area

    # Reproject
    built.rio.set_nodata(0)
    built = built.rio.reproject("EPSG:4326")

    # Create a yearly coded binary built array
    built_bin = (built > thresh).astype("uint8")
    built_bin *= YEARS_UINT8[:, None, None]
    built_bin.values[built_bin.values == 0] = 200

    # Aggregate yearly binary built data
    # Keep earliest year of observed urbanization
    built_bin_agg = np.min(built_bin, axis=0)
    built_bin_agg.values[built_bin_agg == 200] = 0

    return built_bin_agg


def _raster_to_image(built, thresh):
    built_bin_agg = _built_to_bin(built, thresh)

    # Create array to hold colorized image
    built_img = np.zeros((*built_bin_agg.shape, 4), dtype="uint8")

    # Set colormap
    colors_rgba = [plt.get_cmap("cividis", 10)(i) for i in range(10)]
    colors = (np.array(colors_rgba) * 255).astype("uint8")
    cmap = {y: c for y, c in zip(YEARS_UINT8, colors)}
    cmap_cat = {y: mpl.colors.rgb2hex(c) for y, c in zip(YEARS, colors_rgba)}

    # Set colors manually on image array
    for year, color in cmap.items():
        mask = built_bin_agg == year
        built_img[mask] = color

    # Create image bounding box
    lonmin, latmin, lonmax, latmax = built_bin_agg.rio.bounds()
    coordinates = np.array(
        [
            [lonmin, latmin],
            [lonmax, latmin],
            [lonmax, latmax],
            [lonmin, latmax],
        ]
    )

    # Create Image object (memory haevy)
    img = ImageOps.flip(Image.fromarray(built_img))

    # High res image
    if HIGH_RES:
        img = _resize_image(img)

    return img, cmap_cat, coordinates


def plot_built_agg_img(
    smod, built, bbox_mollweide, centroid_mollweide, thresh=0.2, language="es"
):
    """Plots historic built using an image overlay."""
    img, cmap_cat, coordinates = _raster_to_image(built, thresh)

    lon = coordinates[:, 0]
    lat = coordinates[:, 1]

    dummy_df = pd.DataFrame({"lat": [0] * 10, "lon": [0] * 10, "Year": YEARS})
    fig = px.scatter_mapbox(
        dummy_df,
        lat="lat",
        lon="lon",
        color="Year",
        color_discrete_map=cmap_cat,
        mapbox_style="carto-positron",
    )

    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=HEIGHT,
        legend_title=ut.year[language],
        legend_orientation="h",
        mapbox_center={
            "lat": (lat.min() + lat.max()) / 2,
            "lon": (lon.min() + lon.max()) / 2,
        },
    )

    _update_figure(fig, centroid_mollweide, smod, language)
    _add_bbox_trace(fig, bbox_mollweide, language)

    fig.update_layout(
        mapbox_layers=[
            {
                "sourcetype": "image",
                "source": img,
                "coordinates": coordinates,
                "opacity": 0.7,
                "below": "traces",
            }
        ]
    )

    fig.add_annotation(
        text=f'Datos de: <a href="{URL_BUILT}"">GHS-BUILT-S</a>',
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1,
        y=0,
    )

    return fig


def _smod_to_frame(smod, feature, language):
    if feature == "clusters":
        c_code = 2
    elif feature == "centers":
        c_code = 3
    else:
        raise ValueError("Feature must be either clusters or centers.")

    smod_lvl_1 = smod // 10

    smod_lvl_1_df = smod_lvl_1.to_dataframe(name="smod").reset_index()
    smod_lvl_1_df = smod_lvl_1_df.drop(columns="spatial_ref")

    df = smod_lvl_1_df[smod_lvl_1_df.smod >= c_code].drop(columns="smod")

    df = df.groupby(["x", "y"]).min().reset_index()
    df = df.rename(columns={"band": ut.year[language]})
    df = df.sort_values(ut.year[language]).reset_index(drop=True)

    df["geometry"] = df.apply(
        utils.raster.row2cell, res_xy=smod.rio.resolution(), axis=1
    )

    gdf = gpd.GeoDataFrame(df.drop(columns=["x", "y"]), crs=smod.rio.crs)

    gdf[ut.year[language]] = gdf[ut.year[language]].astype(str)
    return gdf


def plot_smod_clusters(smod, bbox_latlon, feature="clusters", language="es"):
    gdf = _smod_to_frame(smod, feature, language)

    colors_rgba = [plt.get_cmap("cividis", 10)(i) for i in range(10)]
    cmap_cat = {y: mpl.colors.rgb2hex(c) for y, c in zip(YEARS, colors_rgba)}

    gdf = gdf.to_crs("EPSG:4326").reset_index()
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry,
        color=ut.year[language],
        locations="index",
        hover_name=None,
        hover_data={ut.year[language]: True, "index": False},
        color_discrete_map=cmap_cat,
        opacity=0.5,
        mapbox_style="carto-positron",
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        mapbox_center={
            "lat": bbox_latlon.centroid.xy[1][0],
            "lon": bbox_latlon.centroid.xy[0][0],
        }
    )

    fig.add_annotation(
        text=f'Datos de: <a href="{URL_SMOD}"">GHS-SMOD</a>',
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1,
        y=0,
    )

    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=HEIGHT,
        # width=600,
        legend_orientation="h",
    )

    return fig


def plot_built_year_img(
    smod,
    built,
    bbox_latlon,
    bbox_mollweide,
    centroid_mollweide,
    year=2020,
    language="es",
):
    """Plots built for year using an image overlay."""

    resolution = built.rio.resolution()
    pixel_area = abs(np.prod(resolution))

    # Select specific year and transform into density
    # Only densitities can be safely reprojected
    built = built.sel(band=year) / pixel_area
    built.rio.set_nodata(0)

    # Reprojecto to lat lon
    built = built.rio.reproject(dst_crs=4326)

    # Get colorized image.
    cmap = plt.get_cmap("cividis").copy()
    colorized = cmap(built)
    mask = built.values == 0
    colorized[mask] = (0, 0, 0, 0)
    colorized = np.uint8(colorized * 255)
    img = ImageOps.flip(Image.fromarray(colorized))

    # Create image bounding box
    lonmin, latmin, lonmax, latmax = built.rio.bounds()
    coordinates = [
        [lonmin, latmin],
        [lonmax, latmin],
        [lonmax, latmax],
        [lonmin, latmax],
    ]

    # Create figure
    west, south, east, north = bbox_latlon.bounds

    c_col = f"{ut.fraction[language]} <br> {ut.of_construction[language]} <br> {year}"

    dummy_df = pd.DataFrame({"lat": [0] * 2, "lon": [0] * 2, c_col: [0.0, 1.0]})
    fig = px.scatter_mapbox(
        dummy_df,
        lat="lat",
        lon="lon",
        color=c_col,
        color_continuous_scale="cividis",
        mapbox_style="carto-positron",
    )
    fig.update_layout(
        mapbox_center={"lat": (latmin + latmax) / 2, "lon": (lonmin + lonmax) / 2}
    )

    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=HEIGHT, legend_orientation="h"
    )

    _update_figure(fig, centroid_mollweide, smod, language)
    _add_bbox_trace(fig, bbox_mollweide, language)

    # High res image
    if HIGH_RES:
        img = _resize_image(img)

    fig.update_layout(coloraxis_colorbar_orientation="h")
    fig.update_layout(
        mapbox_layers=[
            {
                "sourcetype": "image",
                "source": img,
                "coordinates": coordinates,
                "opacity": 0.7,
                "below": "traces",
            }
        ]
    )

    fig.add_annotation(
        text=f'Datos de: <a href="{URL_BUILT}"">GHS-BUILT-S</a>',
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1,
        y=0,
    )

    return fig


def plot_pop_year_img(
    smod, pop, bbox_mollweide, centroid_mollweide, year=2020, language="es"
):
    resolution = pop.rio.resolution()
    pixel_area = abs(np.prod(resolution)) / 1e6

    # Select specific year and transform into density
    # Only densitities can be safely reprojected
    pop = pop.sel(band=year) / pixel_area
    pop.rio.set_nodata(0)

    # Reprojecto to lat lon
    pop = pop.rio.reproject(dst_crs=4326)

    # Get back counts
    pop = pop * utils.raster.get_area_grid(pop, "km")

    # Normalize values for colormap
    n_classes = 7
    pop_min = np.unique(pop)[1]
    bounds = np.array(
        [-1, pop_min / 2.0, 5.5, 20.5, 100.5, 300.5, 500.5, 1000.5, 10000]
    )
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=n_classes + 1)
    pop_norm = norm(pop).data / n_classes

    # Get colorized image.
    cmap = plt.get_cmap("cividis").copy()
    colorized = cmap(pop_norm)
    mask = pop_norm == 0
    colorized[mask] = (0, 0, 0, 0)
    colorized = np.uint8(colorized * 255)
    img = ImageOps.flip(Image.fromarray(colorized))

    # Create image bounding box
    lonmin, latmin, lonmax, latmax = pop.rio.bounds()
    coordinates = [
        [lonmin, latmin],
        [lonmax, latmin],
        [lonmax, latmax],
        [lonmin, latmax],
    ]

    mid_vals = ["3", "10", "50", "200", "400", "750", "2000"]
    cls_names = [
        "0 - 5",
        "6 - 20",
        "21 - 100",
        "101 - 300",
        "301 - 500",
        "501 - 1,000",
        "1,001 - Max",
    ]
    names = {v: n for v, n in zip(mid_vals, cls_names)}
    colors_d = [mpl.colors.rgb2hex(c) for c in cmap([int(v) for v in mid_vals])]
    cmap_d = {v: c for v, c in zip(mid_vals, colors_d)}

    dummy_df = pd.DataFrame(
        {
            "lat": [0] * n_classes,
            "lon": [0] * n_classes,
            ut.population[language]: mid_vals,
        }
    )

    fig = px.scatter_mapbox(
        dummy_df,
        lat="lat",
        lon="lon",
        color=ut.population[language],
        color_discrete_map=cmap_d,
        mapbox_style="carto-positron",
    )
    fig.update_layout(
        mapbox_center={"lat": (latmin + latmax) / 2, "lon": (lonmin + lonmax) / 2}
    )

    fig.for_each_trace(lambda t: t.update(name=names[t.name]))

    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=HEIGHT, legend_orientation="h"
    )

    _update_figure(fig, centroid_mollweide, smod, language)
    _add_bbox_trace(fig, bbox_mollweide, language)

    if HIGH_RES:
        img = _resize_image(img)

    fig.update_layout(
        mapbox_layers=[
            {
                "sourcetype": "image",
                "source": img,
                "coordinates": coordinates,
                "opacity": 0.7,
                "below": "traces",
            }
        ]
    )

    fig.add_annotation(
        text=f'Datos de: <a href="{URL_POP}"">GHS-POP</a>',
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1,
        y=0,
    )

    return fig


def plot_growth(growth_df, *, y_cols, title, ylabel, var_type, language="es"):
    if var_type == "extensive":
        p_func = px.area
    elif var_type == "intensive":
        p_func = px.line

    fig = p_func(growth_df, x="year", y=y_cols, markers=True)

    fig.update_layout(
        yaxis_title=ylabel,
        yaxis_tickformat=",.3~f",
        xaxis_title=ut.year[language],
        legend_title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_layout(hovermode="x")
    if "pop" in y_cols[0] or "urban" in y_cols[0]:
        fig.update_traces(hovertemplate="%{y:.0f}<extra></extra>")
    else:
        fig.update_traces(hovertemplate="%{y:.2f}<extra></extra>")

    name_dict = dict(
        all=f"{ut.all_zones[language]} {{}} {{:.2f}}%",
        main=f"{ut.central_zone[language]} {{}} {{:.2f}}%",
        other=f"{ut.peripheral_zones[language]} {{}} {{:.2f}}%",
    )

    color_dict = {"all": "black", "main": "maroon", "other": "orange"}
    options = ["all", "main", "other"]

    names = {}
    colors = {}

    for col in y_cols:
        i = 0
        c0 = growth_df[col].iloc[i]
        while np.isnan(c0):
            i += 1
            c0 = growth_df[col].iloc[i]

        cf = growth_df[col].iloc[-1]
        delta = (cf - c0) / c0 * 100
        if delta > 0:
            up_down = "▲"
        else:
            up_down = "▼"
        for option in options:
            if option in col:
                names[col] = name_dict[option].format(up_down, delta)
                colors[col] = color_dict[option]

    fig.for_each_trace(
        lambda t: t.update(line_color=colors[t.name], name=names[t.name])
    )

    return fig
