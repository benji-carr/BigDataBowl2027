"""NFL Combine tracking drill-group animations using nfl.visuals.field.

Dependencies: numpy, pandas, matplotlib, nfl-tracks (``from nfl import visuals``).

This module does not modify its input DataFrame. Each event is aligned by its
first recorded timestamp; player trajectories are separate per event and NFL ID.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


METRICS: dict[str, tuple[str, str]] = {
    "dir": ("Direction", "deg"),
    "a": ("Acceleration", "yd/s²"),
    "y": ("Lateral Y", "yd"),
    "USER_delta_dir": ("Angular Change", "deg"),
    "USER_angular_velocity": ("Angular Velocity", "deg/s"),
    "USER_angular_acceleration": ("Angular Accel.", "deg/s²"),
    "USER_curvature": ("Curvature", "1/yd"),
    "USER_turning_radius": ("Turning Radius", "yd"),
    "USER_normal_acceleration": ("Normal Accel.", "yd/s²"),
}
USER_METRICS = tuple(c for c in METRICS if c.startswith("USER_"))
REQUIRED = {"event_id", "nfl_id", "entity_type", "drill_name", "time", "x", "y", "s", "dir", "a"}


@dataclass
class DrillFilters:
    """Filters applied *before* derived-metric calculation."""
    draft_years: Sequence[int] | None = None
    nfl_ids: Sequence[int] | None = None
    event_ids: Sequence[str] | None = None
    attempts: Sequence[int] | None = None
    query: str | None = None
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None


@dataclass
class AnimationConfig:
    fps: int = 10
    min_speed: float = 0.5
    flip_x: bool = False
    flip_y: bool = False
    swap_xy: bool = False
    field_length: float = 120.0
    field_width: float = 53.33
    trail_alpha: float = 0.07
    trail_width: float = 1.1
    marker_size: float = 30
    show_ball: bool = True
    ball_marker_size: float = 65
    ball_trail_alpha: float = 0.12
    max_frames: int | None = None
    # Optional player-only histogram below the field, synchronized to playback.
    histogram_column: str | None = None
    histogram_bins: int = 25
    histogram_range: tuple[float, float] | None = None
    histogram_color: str = "#2d81ba"
    # Customize stats without changing rendering code.
    metrics: Mapping[str, tuple[str, str]] = field(default_factory=lambda: METRICS.copy())


def add_turning_metrics(df: pd.DataFrame, min_speed: float = 0.5) -> pd.DataFrame:
    """Compute 6 finite-difference USER_ metrics per (event_id, nfl_id).

    Derivatives are based on successive player observations. Undefined/unstable
    metrics are NaN. No extrapolation, interpolation, or smoothing is performed.
    """
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    for col in USER_METRICS:
        out[col] = np.nan
    mask = out["entity_type"].eq("PLAYER") & out["nfl_id"].notna() & out["time"].notna()
    players = out.loc[mask].sort_values(["event_id", "nfl_id", "time"])
    for _, g in players.groupby(["event_id", "nfl_id"], sort=False):
        dt = g["time"].diff().dt.total_seconds()
        dt = dt.where(dt > 0)
        angle = pd.to_numeric(g["dir"], errors="coerce")
        speed = pd.to_numeric(g["s"], errors="coerce")
        delta = ((angle.diff() + 180) % 360) - 180
        omega = delta / dt
        alpha = omega.diff() / dt
        radians_per_s = np.deg2rad(omega)
        moving_speed = speed.where(speed > min_speed)
        curvature = radians_per_s.abs() / moving_speed
        turning_radius = 1 / curvature.where(curvature > 0)
        normal_accel = moving_speed * radians_per_s.abs()
        values = [delta, omega, alpha, curvature, turning_radius, normal_accel]
        out.loc[g.index, list(USER_METRICS)] = np.column_stack([v.to_numpy() for v in values])
    return out


def prepare_drill(
    tracking_df: pd.DataFrame,
    drill_name: str,
    *,
    filters: DrillFilters | None = None,
    config: AnimationConfig | None = None,
) -> pd.DataFrame:
    """Return selected players and their event-linked ball observations.

    Player filters select the eligible attempts; matching BALL frames are
    retained from those events only when show_ball=True. Ball records do not
    contribute to metrics or player statistics.
    """
    config = config or AnimationConfig()
    filters = filters or DrillFilters()
    missing = REQUIRED - set(tracking_df.columns)
    if missing:
        raise ValueError(f"Tracking data missing required columns: {sorted(missing)}")
    if config.fps <= 0:
        raise ValueError("fps must be positive")

    df = tracking_df.loc[
        tracking_df["drill_name"].eq(drill_name) & tracking_df["entity_type"].eq("PLAYER")
        & tracking_df["nfl_id"].notna()
    ].copy()
    for name, values in (
        ("draft_year", filters.draft_years),
        ("nfl_id", filters.nfl_ids),
        ("event_id", filters.event_ids),
        ("attempt", filters.attempts),
    ):
        if values is not None:
            if name not in df.columns:
                raise ValueError(f"Filter requires missing column: {name}")
            df = df[df[name].isin(values)].copy()
    if filters.query:
        df = df.query(filters.query).copy()
    if filters.predicate is not None:
        keep = filters.predicate(df)
        if not isinstance(keep, pd.Series) or not keep.index.equals(df.index):
            raise ValueError("predicate must return a boolean Series with the input index")
        df = df.loc[keep.fillna(False)].copy()
    if df.empty:
        raise ValueError(f"No player observations matched drill={drill_name!r} and filters")

    # Derivatives must never be computed across event/player boundaries.
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "x", "y", "s"])
    if df.empty:
        raise ValueError("No observations with valid time, x, y, and speed")
    df = add_turning_metrics(df, min_speed=config.min_speed)
    df = df.sort_values(["event_id", "nfl_id", "time"])

    # Align to selected player's first timestamp in each event. Keep that
    # origin when adding BALL records, otherwise their arrival might shift
    # the entire animation timeline.
    starts = df.groupby("event_id")["time"].min()
    ends = df.groupby("event_id")["time"].max()

    if config.show_ball:
        balls = tracking_df.loc[
            tracking_df["drill_name"].eq(drill_name)
            & tracking_df["entity_type"].eq("BALL")
            & tracking_df["event_id"].isin(starts.index)
        ].copy()
        balls["time"] = pd.to_datetime(balls["time"], errors="coerce")
        balls = balls.dropna(subset=["time", "x", "y"])
        # Restrict to the selected players' observation windows.
        balls = balls.loc[
            balls["time"].ge(balls["event_id"].map(starts))
            & balls["time"].le(balls["event_id"].map(ends))
        ].copy()
        # Avoid duplicate index labels during later assignment/grouping.
        df = pd.concat([df, balls], ignore_index=True, sort=False)

    df["USER_elapsed_seconds"] = (
        df["time"] - df["event_id"].map(starts)
    ).dt.total_seconds()
    df["USER_aligned_frame"] = np.rint(df["USER_elapsed_seconds"] * config.fps).astype(int)
    # BALL nfl_id may be nullable or may identify the associated player.
    # Deduplicate each event's ball samples separately from player samples.
    players_mask = df["entity_type"].eq("PLAYER")
    selected_players = df.loc[players_mask].drop_duplicates(
        ["event_id", "nfl_id", "USER_aligned_frame"], keep="first"
    )
    selected_balls = df.loc[~players_mask].drop_duplicates(
        ["event_id", "USER_aligned_frame"], keep="first"
    )
    df = pd.concat([selected_players, selected_balls], ignore_index=True, sort=False)
    if config.max_frames is not None:
        df = df[df["USER_aligned_frame"] < config.max_frames].copy()
    if df.empty:
        raise ValueError("No observations remain after applying max_frames")

    # This transform affects display only; physical USER_ metrics use original data.
    x, y = df["x"].astype(float), df["y"].astype(float)
    if config.swap_xy:
        x, y = y, x
        length, width = config.field_width, config.field_length
    else:
        length, width = config.field_length, config.field_width
    df["PLOT_x"] = length - x if config.flip_x else x
    df["PLOT_y"] = width - y if config.flip_y else y
    return df


def _frame_statistics(df: pd.DataFrame, metrics: Mapping[str, tuple[str, str]]):
    """Per-frame mean/median/quartiles; directions handled on a local circle."""
    frame = "USER_aligned_frame"
    summaries = {}
    for name in metrics:
        if name not in df.columns:
            raise ValueError(f"Requested statistic column not present: {name}")
        scratch = df[[frame, name]].copy()
        scratch[name] = pd.to_numeric(scratch[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if name == "dir":
            # Circular mean (degrees), then local signed residuals for quartiles.
            radians = np.deg2rad(scratch[name])
            scratch["sin"] = np.sin(radians)
            scratch["cos"] = np.cos(radians)
            centers = np.rad2deg(np.arctan2(
                scratch.groupby(frame)["sin"].transform("mean"),
                scratch.groupby(frame)["cos"].transform("mean"),
            )) % 360
            scratch[name] = centers + ((scratch[name] - centers + 180) % 360 - 180)
        agg = scratch.groupby(frame)[name].agg(
            mean="mean", median="median",
            q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75),
            count="count",
        )
        if name == "dir":
            for key in ["mean", "median", "q1", "q3"]:
                agg[key] %= 360
        summaries[name] = agg
    return summaries


def animate_drill(
    tracking_df: pd.DataFrame,
    drill_name: str,
    *,
    filters: DrillFilters | None = None,
    config: AnimationConfig | None = None,
):
    """Return ``(animation, prepared_dataframe)`` for a selected drill.

    Usage in Jupyter::
        from IPython.display import HTML, display
        anim, prepared = animate_drill(data, "END_ZONE_FADE_RIGHT")
        display(HTML(anim.to_jshtml(fps=10)))

    Keep the returned animation alive for playback/export.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from nfl import visuals

    config = config or AnimationConfig()
    df = prepare_drill(tracking_df, drill_name, filters=filters, config=config)
    players_df = df.loc[df["entity_type"].eq("PLAYER")].copy()
    balls_df = df.loc[df["entity_type"].eq("BALL")].copy()
    # Distribution summaries are strictly player-only. A BALL row may carry
    # nfl_id metadata, but it must never contribute to the sample.
    stats = _frame_statistics(players_df, config.metrics)
    frame_col = "USER_aligned_frame"
    trajectories = [
        (g[frame_col].to_numpy(dtype=int), g["PLOT_x"].to_numpy(dtype=float), g["PLOT_y"].to_numpy(dtype=float))
        for _, g in players_df.groupby(["event_id", "nfl_id"], sort=False)
    ]
    frame_groups = {int(f): g for f, g in players_df.groupby(frame_col)}
    ball_frame_groups = {int(f): g for f, g in balls_df.groupby(frame_col)}
    max_frame = int(df[frame_col].max())

    # Fixed edges let every frame's histogram use the same horizontal scale.
    # This represents CURRENT player observations, not historical observations.
    hist_values_by_frame = {}
    hist_edges = None
    if config.histogram_column is not None:
        name = config.histogram_column
        if name not in players_df.columns:
            raise ValueError(f"Histogram column not present: {name}")
        if not isinstance(config.histogram_bins, int) or config.histogram_bins < 1:
            raise ValueError("histogram_bins must be a positive integer")
        values = pd.to_numeric(players_df[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = values.dropna().to_numpy(dtype=float)
        if not len(valid):
            raise ValueError(f"Histogram column {name!r} has no finite numeric player values")
        hist_edges = np.histogram_bin_edges(
            valid, bins=config.histogram_bins, range=config.histogram_range
        )
        for f, g in players_df.groupby(frame_col):
            vals = pd.to_numeric(g[name], errors="coerce").to_numpy(dtype=float)
            hist_values_by_frame[int(f)] = vals[np.isfinite(vals)]

    fig, ax = visuals.field(
        yard_numbers=True, touchdown_markings=True,
        fifty_yard=False, fig_size=(19, 9),
    )
    if hist_edges is not None:
        fig.set_size_inches(19, 12)
        # The upper field and right-hand table remain separate from the
        # lower histogram. The field axis retains its football aspect ratio.
        ax.set_position([.06, .34, .59, .61])
    else:
        fig.set_size_inches(19, 9)
        fig.subplots_adjust(right=.67)
    lines = [
        ax.plot([], [], color="cyan", alpha=config.trail_alpha,
                linewidth=config.trail_width, zorder=20)[0]
        for _ in trajectories
    ]
    scatter = ax.scatter([], [], s=config.marker_size, color="dodgerblue",
                         alpha=.8, zorder=30)
    ball_scatter = ax.scatter(
        [], [], s=config.ball_marker_size, marker="D",
        color="orange", edgecolors="black", linewidths=.6,
        zorder=31, label="Ball"
    ) if config.show_ball else None
    ball_trajectories = [
        (g[frame_col].to_numpy(dtype=int),
         g["PLOT_x"].to_numpy(dtype=float),
         g["PLOT_y"].to_numpy(dtype=float))
        for _, g in balls_df.sort_values(frame_col).groupby("event_id", sort=False)
    ] if config.show_ball else []
    ball_lines = [
        ax.plot([], [], color="orange", alpha=config.ball_trail_alpha,
                linewidth=config.trail_width, zorder=21)[0]
        for _ in ball_trajectories
    ]
    # Use four separately positioned columns. The previous single monospaced
    # line was clipped past the right edge, hiding Q1 and Q3.
    fig.text(.69, .92, "PLAYER DISTRIBUTIONS", fontsize=15, weight="bold")
    fig.text(.69, .885, drill_name, fontsize=10)
    time_label = fig.text(.69, .85, "", fontsize=9)

    columns = ("mean", "median", "q1", "q3")
    column_x = (.735, .812, .887, .965)
    for name, xpos in zip(("Mean", "Median", "Q1", "Q3"), column_x):
        fig.text(xpos, .804, name, ha="right", fontsize=9, weight="bold")

    metrics = list(config.metrics.items())
    top, bottom = .765, (.355 if hist_edges is not None else .07)
    spacing = (top - bottom) / max(1, len(metrics))
    value_labels = {}
    for i, (name, (title, unit)) in enumerate(metrics):
        y = top - i * spacing
        fig.text(.69, y, f"{title} ({unit})", fontsize=8.6, weight="bold")
        value_labels[name] = {
            col: fig.text(xpos, y - min(.032, spacing * .46), "",
                          fontsize=8.7, ha="right", family="monospace")
            for col, xpos in zip(columns, column_x)
        }

    histogram_ax = None
    histogram_bars = None
    histogram_title = None
    if hist_edges is not None:
        histogram_ax = fig.add_axes([.07, .075, .57, .205])
        widths = np.diff(hist_edges)
        histogram_bars = histogram_ax.bar(
            hist_edges[:-1], np.zeros(len(widths)), width=widths,
            align="edge", color=config.histogram_color,
            edgecolor="white", linewidth=.4,
        )
        histogram_ax.set_xlim(hist_edges[0], hist_edges[-1])
        max_count = max(
            (np.histogram(vals, bins=hist_edges)[0].max(initial=0)
             for vals in hist_values_by_frame.values()), default=0
        )
        histogram_ax.set_ylim(0, max(1, int(max_count)) * 1.12)
        histogram_ax.set_xlabel(config.histogram_column)
        histogram_ax.set_ylabel("Player count")
        histogram_title = histogram_ax.set_title("", fontsize=10)

    def fmt(x):
        return f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "N/A"

    def update(frame):
        for (f, x, y), line in zip(trajectories, lines):
            end = np.searchsorted(f, frame, side="right")
            line.set_data(x[:end], y[:end])
        for (f, x, y), line in zip(ball_trajectories, ball_lines):
            end = np.searchsorted(f, frame, side="right")
            line.set_data(x[:end], y[:end])
        if ball_scatter is not None:
            ball_current = ball_frame_groups.get(frame)
            ball_points = (
                ball_current[["PLOT_x", "PLOT_y"]].to_numpy(dtype=float).reshape(-1, 2)
                if ball_current is not None else np.empty((0, 2))
            )
            ball_scatter.set_offsets(ball_points)
        current = frame_groups.get(frame)
        points = (
            current[["PLOT_x", "PLOT_y"]].to_numpy(dtype=float).reshape(-1, 2)
            if current is not None else np.empty((0, 2))
        )
        scatter.set_offsets(points)
        for name, summary in stats.items():
            row = summary.loc[frame] if frame in summary.index else None
            for col in columns:
                value_labels[name][col].set_text(fmt(row[col]) if row is not None else "N/A")
        time_label.set_text(f"Elapsed: {frame/config.fps:.1f}s | Active: {len(points)} | Trajectories: {len(trajectories)}")
        if histogram_ax is not None:
            values = hist_values_by_frame.get(frame, np.empty(0))
            counts, _ = np.histogram(values, bins=hist_edges)
            for bar, count in zip(histogram_bars, counts):
                bar.set_height(int(count))
            histogram_title.set_text(
                f"{config.histogram_column} distribution — elapsed {frame/config.fps:.1f}s "
                f"(n={int(counts.sum())})"
            )
        ax.set_title(f"{drill_name} — player trajectories")
        return (*lines, *ball_lines, scatter, *((ball_scatter,) if ball_scatter is not None else ()), time_label, *(artist for row in value_labels.values() for artist in row.values()))

    animation = FuncAnimation(
        fig, update, frames=range(max_frame + 1),
        interval=1000/config.fps, blit=False, repeat=True,
    )
    plt.close(fig)
    return animation, df
