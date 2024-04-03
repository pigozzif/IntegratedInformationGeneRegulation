from fractions import Fraction

import jax.numpy as jnp
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
from addict import Dict
from numpy.linalg import det
import matplotlib.pyplot as plt


def downsample_traj(traj, scaling_vector=np.ones((2,)), eps=0.05):
    """
    traj: (2, T) array
    scaling_vector: 2D array
    eps: we discard all points that are closer to eps
    """

    scaling_vector = scaling_vector[:, np.newaxis]
    traj = traj / scaling_vector
    traj_filt = [traj[:, i] for i in range(10)]
    ids_filt = [i for i in range(10)]

    for i, e in enumerate(traj.T):
        if i >= 10 and (np.linalg.norm(e - traj_filt[-1], ord=2) > eps or i > len(traj.T) - 10):
            traj_filt.append(e)
            ids_filt.append(i)
    traj_filt = np.array(traj_filt).T
    ids_filt = np.array(ids_filt)
    return traj_filt * scaling_vector, ids_filt


# color to visualize time from trajectory start point A (red) to trajectory end point B (cyan)
n_points = 100
c = [mcolors.hsv_to_rgb((step / (2 * n_points), 1, 1)) for step in range(n_points)]
traj_cscale = [(x.item(), f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})') for (x, color) in
               zip(jnp.linspace(0., 1., len(c)), c)]

# default colors
default_colors = ['rgb(204,121,167)',
                  'rgb(0,114,178)',
                  'rgb(230,159,0)',
                  'rgb(0,158,115)',
                  'rgb(127,127,127)',
                  'rgb(240,228,66)',
                  'rgb(148,103,189)',
                  'rgb(86,180,233)',
                  'rgb(213,94,0)',
                  'rgb(140,86,75)',
                  'rgb(214,39,40)',
                  'rgb(0,0,0)']
transparency = 0.6
default_colors_shade = ['rgba' + color[3:-1] + ', ' + str(transparency) + ')' for color in default_colors]

default_dashes = ['solid', 'longdash', 'dot', 'dash', 'dashdot', 'longdashdot',
                  'solid', 'longdash', 'dot', 'dash', 'dashdot', 'longdashdot']

# plotly default layout
default_layout = Dict(
    font=Dict(
        size=10,
    ),
    title=Dict(font_size=10),

    xaxis=Dict(
        titlefont=Dict(size=10),
        tickfont=Dict(size=10),
        title_standoff=5,
        linecolor='rgba(0, 0, 0, .1)',
    ),

    yaxis=Dict(
        titlefont=Dict(size=10),
        tickfont=Dict(size=10),
        title_standoff=5,
        gridcolor="rgba(0, 0, 0, .1)",
        linecolor="rgba(0, 0, 0, .1)"
    ),

    updatemenus=[],
    autosize=True,

    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',

    margin=Dict(
        l=20,
        r=20,
        b=20,
        t=20
    ),

    legend=Dict(
        xanchor='left',
        yanchor='top',
        y=1,
        x=1,
        font_size=10,
    ),

)

default_annotation_layout = Dict(
    font_size=10,
)


def make_html_fig(fig_idx, fig, width, height, title, size_unit="px", figtitle_fontsize="1em", title_fontsize="0.8em",
                  full_html=False, include_plotlyjs=False, config=None):
    if config is None:
        config = {}
    if isinstance(fig, go.Figure):
        # autosize fig
        fig.layout.autosize = True
        fig.layout.width = None
        fig.layout.height = None

        # convert to html
        default_config = {'displaylogo': False, 'modeBarButtonsToRemove': ['select', 'lasso2d', 'autoScale']}
        for k, v in config.items():
            default_config[k] = v
        html_fig = fig.to_html(config=default_config, full_html=full_html, include_plotlyjs=include_plotlyjs,
                               default_width='100%', default_height='100%')
        html_fig = html_fig[:4] + f' style="aspect-ratio: {str(Fraction(width, height))};"' + html_fig[
                                                                                              4:]  # add aspect ratio

    elif isinstance(fig, str) and fig.split(".")[-1] in ["png", "jpg", "jpeg"]:
        html_fig = f'<div><img src="{fig}" alt="Figure {fig_idx}" style="aspect-ratio: {str(Fraction(width, height))}; width:100%;"></div> '

    # change div style and append title
    div_tag = f'<div id="figure-{fig_idx}" style="margin: 50px auto; max-width: {width}{size_unit};">'
    title_tag = f'<span style="font-size: {title_fontsize}; color: rgba(0, 0, 0, 0.6)">' + f'<b style="font-size: {figtitle_fontsize};">Figure {fig_idx}: </b>' + title + '</span> '
    html_fig = div_tag + html_fig + title_tag + '</div>'

    return html_fig


def make_img_fig(fig_idx, fig, width, height, title, img_format="png", scale=1, size_unit="px", title_fontsize="1em"):
    if isinstance(fig, go.Figure):
        # autosize fig
        fig.layout.autosize = True
        fig.layout.width = None
        fig.layout.height = None

        if size_unit != "px":
            raise NotImplementedError

        # convert to img
        img_fig = fig.to_image(width=width, height=height, scale=scale, format=img_format)

    elif isinstance(fig, str) and fig.split(".")[-1] in ["png", "jpg", "jpeg"]:
        img_fig = fig

    img_title = f'Figure {fig_idx}: ' + title

    return img_fig, img_title


def plot_states_trajectory(fig_name, system_rollout, system_outputs, nb_save_outputs=True, observed_node_ids=None,
                           observed_node_names=None):
    assert (observed_node_names is None and observed_node_ids is None) or\
           (len(observed_node_names) == len(observed_node_ids))
    fig = go.Figure(layout=default_layout)

    if observed_node_ids is None:
        observed_node_ids = system_rollout.grn_step.y_indexes.values()
    for y_name, y_idx in system_rollout.grn_step.y_indexes.items():
        if y_idx not in observed_node_ids:
            continue
        if observed_node_names is not None:
            y_name = observed_node_names[y_idx]
        fig.add_trace(go.Scatter(x=system_outputs.ts[0:-1:10], y=system_outputs.ys[y_idx, 0:-1:10],
                                 name=y_name,
                                 line=dict(color=default_colors[y_idx]),
                                 hovertemplate=y_name + ' (t=%{x:.0f}): %{y:.2f} &mu;M <extra></extra>'))

    fig.update_xaxes(title_text="reaction time [sec]")
    fig.update_yaxes(title_text="concentration of proteins [&mu;M]")

    # Serialize fig to json and save
    if nb_save_outputs:
        fig.write_image(fig_name)

    return fig


def plot_observed_nodes_trajectory(fig_name, system_outputs, observed_node_ids, system_rollout_config,
                                   observed_node_names,
                                   nb_save_outputs=True):
    fig = go.Figure(layout=default_layout)

    A = [system_outputs.ys[observed_node_ids[0], 0], system_outputs.ys[observed_node_ids[1], 0]]
    B = [system_outputs.ys[observed_node_ids[0], -1], system_outputs.ys[observed_node_ids[1], -1]]
    # downsample traj
    default_trajectory = system_outputs.ys[jnp.array(observed_node_ids)]
    scaling_vector = np.nanmax(default_trajectory, 1) - np.nanmin(default_trajectory, 1)
    default_trajectory, display_ts = downsample_traj(default_trajectory, scaling_vector, 1e-3)
    # plot
    fig.add_trace(go.Scatter(x=default_trajectory[0], y=default_trajectory[1], showlegend=False,
                             mode="markers",
                             marker=Dict(color=system_outputs.ts[display_ts], colorscale=traj_cscale, size=4,
                                         colorbar=Dict(title="time [sec]", thickness=10)),
                             hovertemplate='(%{x:.02f}, %{y:.02f})<extra></extra>'
                             ))
    fig.add_annotation(x=A[0], y=A[1], ay=-30, ax=20, text="init state <br> A(t=0)")
    fig.add_annotation(x=B[0], y=B[1], ay=30, text=f" <b>endpoint B<br> (t={system_rollout_config.n_secs})</b>")
    # fig.add_trace(go.Scatter(x=[B[0]], y=[B[1]], showlegend=False, marker=dict(size=12, color=traj_cscale[-1][1])))
    fig.update_xaxes(title_text=f"{observed_node_names[0]} [&mu;M]")
    fig.update_yaxes(title_text=f"{observed_node_names[1]} [&mu;M]", gridcolor='rgba(0,0,0,0)')

    # Serialize fig to json and save
    if nb_save_outputs:
        fig.write_json(fig_name)

    return fig
