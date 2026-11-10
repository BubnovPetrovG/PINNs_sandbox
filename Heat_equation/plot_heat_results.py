"""Plot saved heat-equation results using NumPy, SciPy, and Matplotlib.

From the repository root:
    python Heat_equation/plot_heat_results.py

STALE: Reads comparison files (*_heat_NTK_seed_*.mat) and data-size experiments
(dg_pinn_heat_NTK_Nd_*.mat) from the results directory and Heat_equation/.
Reads those temperature results plus *_heat_flux_seed_*.mat flux results.
Also reads *_heat_point_flux_seed_*.mat for pointwise sensor measurements.
STALE: For flux runs, pass --results-dir results/flux/original (or results/flux/uniform).
For regional flux runs, pass --results-dir results/flux/region/original (or results/flux/region/uniform).
For random pointwise flux runs, pass --results-dir results/flux/pointwise/random/original (or uniform).
STALE: Flux files also produce observed/predicted average-flux curves for each region.
Flux files also produce observed/predicted curves for each region or point sensor.
The legacy metrics.csv column relative_L2_test is a temperature metric; for
flux runs it evaluates the full temperature reference grid, not held-out flux.
Saves PNG figures and metrics.csv; it does not run or modify training.
Use --results-dir and --output-dir to choose other locations.

The files save a history of beta, rather than a final model checkpoint.
Reported diffusivity therefore uses the LAST LOGGED beta squared.
The original files do not record the observation count for comparison runs.
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening blocking desktop windows.
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


PATTERN = re.compile(
    r"(?P<method>dgpinn|dg_pinn|pinn)_heat_(?P<operator>NTK|flux|point_flux)_"
    r"(?P<kind>seed|Nd)_(?P<value>\d+)\.mat"
)
COLORS = {"DG-PINN": "#0072B2", "PINN": "#D55E00"}


def vector(data, key):
    return np.asarray(data.get(key, []), dtype=float).reshape(-1)


def load_result(path, match, beta_true):
    data = loadmat(path)
    for key in ("U_pred", "u_true", "u_pred", "u_test", "beta", "time"):
        if key not in data or np.asarray(data[key]).size == 0:
            raise ValueError(f"{path.name}: missing or empty '{key}'")
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f"{path.name}: '{key}' contains nonfinite values")
    predicted = np.asarray(data["U_pred"])
    if predicted.ndim != 2 or predicted.shape != np.asarray(data["u_true"]).shape:
        raise ValueError(f"{path.name}: temperature grids must have matching 2D shapes")
    observed, estimated = vector(data, "u_test"), vector(data, "u_pred")
    if observed.shape != estimated.shape or np.linalg.norm(observed) == 0:
        raise ValueError(f"{path.name}: invalid test temperatures")
    diffusivity = vector(data, "beta")[-1] ** 2
    method = "PINN" if match["method"] == "pinn" else "DG-PINN"
    kind, value = match["kind"], int(match["value"])
    measurement = match["operator"] if match["operator"] != "NTK" else "temperature"
    flux_relative_l2, flux_rmse, flux_count = "", "", ""
    if measurement in ("flux", "point_flux"):
        coordinates = ("flux_x", "flux_t") if measurement == "point_flux" else (
            "flux_x_start", "flux_x_end", "flux_t"
        )
        for key in ("q_pred", "q_true", *coordinates):
            values = vector(data, key)
            if not values.size or not np.all(np.isfinite(values)):
                raise ValueError(f"{path.name}: missing, empty or nonfinite '{key}'")
        flux_count = len(vector(data, "q_true"))
        if any(len(vector(data, key)) != flux_count
               for key in ("q_pred", *coordinates)):
            raise ValueError(f"{path.name}: flux predictions and coordinates must match")
        flux_error = vector(data, "q_pred") - vector(data, "q_true")
        flux_norm = np.linalg.norm(vector(data, "q_true"))
        flux_relative_l2 = np.linalg.norm(flux_error) / flux_norm if flux_norm else float("nan")
        flux_rmse = np.sqrt(np.mean(flux_error**2))
    metrics = {
        "method": method,
        "seed": value if kind == "seed" else "",
        "N_d": value if kind == "Nd" else flux_count,
        "relative_L2_test": np.linalg.norm(estimated - observed) / np.linalg.norm(observed),
        "diffusivity_last_logged": diffusivity,
        "diffusivity_APE_percent": 100 * abs(diffusivity - beta_true**2) / beta_true**2,
        "training_time_seconds": vector(data, "time")[0],
        "source_file": str(path),
        "measurement": measurement,
        "flux_relative_L2_training": flux_relative_l2,
        "flux_RMSE_training": flux_rmse,
    }
    return {"data": data, "method": method, "kind": kind, "value": value,
            "measurement": measurement, "metrics": metrics}


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=250)
    plt.close(fig)


def history_steps(data, values):
    # STALE DG-PINN's data loss includes pretraining; other histories start later.
    # Align each history to the data-loss timeline; flux beta also includes pretraining.
    total = max(len(vector(data, "loss_d")), len(values))
    return np.arange(total - len(values) + 1, total + 1)


def positive(values):
    # Exact zero is invisible on a logarithmic axis.
    return np.maximum(values, np.finfo(float).tiny)


def plot_seed(records, output, seed, beta_true):
    n = len(records)
    measurement = records[0]["measurement"]
    flux_data = measurement in ("flux", "point_flux")
    point_data = measurement == "point_flux"
    tag = f"{measurement}_seed_{seed}" if flux_data else f"seed_{seed}"
    observation_title = (" | pointwise-flux observations" if point_data else
                         " | average-flux observations" if flux_data else "")
    data_label = ("Pointwise flux data" if point_data else
                  "Average flux data" if flux_data else "Temperature data")
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n), squeeze=False,
                             layout="constrained")
    temperatures = [
        np.asarray(record["data"][key])
        for record in records for key in ("u_true", "U_pred")
    ]
    vmin, vmax = min(a.min() for a in temperatures), max(a.max() for a in temperatures)
    error_max = max(
        np.abs(record["data"]["U_pred"] - record["data"]["u_true"]).max()
        for record in records
    )
    error_max = max(error_max, np.finfo(float).eps)
    for row, record in enumerate(records):
        data, method = record["data"], record["method"]
        panels = (
            (data["u_true"], "Exact temperature", "RdBu_r", vmin, vmax),
            (data["U_pred"], f"{method} prediction", "RdBu_r", vmin, vmax),
            (np.abs(data["U_pred"] - data["u_true"]),
             "Absolute error", "magma", 0, error_max),
        )
        for ax, (field, title, cmap, low, high) in zip(axes[row], panels):
            # Original meshgrid stores time in rows and position in columns.
            im = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1),
                           aspect="auto", cmap=cmap, vmin=low, vmax=high)
            ax.set(title=title, xlabel="Position x", ylabel="Time t")
            fig.colorbar(im, ax=ax, label="Absolute temperature error"
                         if title == "Absolute error" else "Temperature u")
    fig.suptitle(f"Heat equation{observation_title} | seed {seed}")
    save_figure(fig, output, f"temperature_fields_{tag}")

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False,
                             layout="constrained")
    loss_specs = (
        ("loss_d", data_label, "#0072B2"),
        ("loss_r", "PDE residual", "#D55E00"),
        ("loss_i", "Initial condition", "#009E73"),
        ("loss_b", "Boundary condition", "#CC79A7"),
    )
    for ax, record in zip(axes[0], records):
        data = record["data"]
        for key, label, color in loss_specs:
            values = vector(data, key)
            if values.size:
                ax.semilogy(history_steps(data, values), positive(values),
                            label=label, color=color, linewidth=1.1)
        pretrain = len(vector(data, "loss_d")) - len(vector(data, "loss_r"))
        if record["method"] == "DG-PINN" and pretrain > 0:
            ax.axvline(pretrain, color="0.35", linestyle="--",
                       linewidth=1, label="Fine-tuning begins")
        ax.set(title=record["method"], xlabel="Recorded training evaluation",
               ylabel="Unweighted mean squared loss")
        ax.grid(True, which="major", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Training losses{observation_title} | seed {seed}")
    save_figure(fig, output, f"losses_{tag}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for record in records:
        data, method = record["data"], record["method"]
        diffusivity = vector(data, "beta") ** 2
        steps = history_steps(data, diffusivity)
        ape = 100 * np.abs(diffusivity - beta_true**2) / beta_true**2
        axes[0].plot(steps, diffusivity, label=method, color=COLORS[method])
        axes[1].semilogy(steps, positive(ape), label=method, color=COLORS[method])
    axes[0].axhline(beta_true**2, color="black", linestyle="--",
                    linewidth=1, label=f"True diffusivity = {beta_true**2:g}")
    axes[0].set(title="Estimated thermal diffusivity", ylabel=r"$\beta^2$")
    axes[1].set(title="Diffusivity estimation error", ylabel="Absolute percentage error (%)")
    for ax in axes:
        ax.set_xlabel("Recorded training evaluation")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Parameter recovery{observation_title} | seed {seed}")
    save_figure(fig, output, f"diffusivity_{tag}")

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False,
                             layout="constrained")
    for ax, record in zip(axes[0], records):
        for index, (_, label, color) in enumerate(
                (loss_specs[1], loss_specs[2], loss_specs[3], loss_specs[0]), start=1):
            values = vector(record["data"], f"lambda{index}")
            if values.size:
                ax.semilogy(history_steps(record["data"], values), positive(values),
                            color=color, label=label)
        ax.set(title=record["method"], xlabel="Recorded training evaluation",
               ylabel="Loss weight")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Loss weights{observation_title} | seed {seed}")
    save_figure(fig, output, f"loss_weights_{tag}")
    if flux_data:
        plot_flux_observations(records, output, seed)


def plot_flux_observations(records, output, seed):
    point_data = records[0]["measurement"] == "point_flux"
    if point_data:
        regions = sorted({
            (x, x) for record in records for x in vector(record["data"], "flux_x")
        })
    else:
        regions = sorted({
            (left, right)
            for record in records
            for left, right in zip(vector(record["data"], "flux_x_start"),
                                   vector(record["data"], "flux_x_end"))
        })
    columns = min(2, len(regions))
    rows = (len(regions) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.8 * rows),
                             squeeze=False, layout="constrained")
    for ax, (left, right) in zip(axes.flat, regions):
        reference_drawn = False
        for record in records:
            data, method = record["data"], record["method"]
            mask = (vector(data, "flux_x") == left) if point_data else (
                (vector(data, "flux_x_start") == left)
                & (vector(data, "flux_x_end") == right)
            )
            if not mask.any():
                continue
            times = vector(data, "flux_t")[mask]
            order = np.argsort(times)
            if not reference_drawn:
                ax.plot(times[order], vector(data, "q_true")[mask][order],
                        "k--", linewidth=1.4, label="Observed (training)")
                reference_drawn = True
            ax.plot(times[order], vector(data, "q_pred")[mask][order],
                    color=COLORS[method], label=method)
        ax.set(title=f"Sensor x = {left:g}" if point_data else f"Region [{left:g}, {right:g}]",
               xlabel="Time t",
               ylabel="Signed pointwise flux" if point_data else "Signed spatial-average flux")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(regions):]:
        ax.set_visible(False)
    fig.suptitle(f"{'Pointwise' if point_data else 'Average'} flux: "
                 f"observations and predictions | seed {seed}")
    name = "point_flux_observations" if point_data else "flux_observations"
    save_figure(fig, output, f"{name}_seed_{seed}")


def plot_data_sweep(records, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for method in sorted({record["method"] for record in records}):
        selected = sorted(
            (record for record in records if record["method"] == method),
            key=lambda record: record["value"],
        )
        sizes = [record["value"] for record in selected]
        axes[0].plot(sizes, [r["metrics"]["relative_L2_test"] for r in selected],
                     "o-", label=method, color=COLORS[method])
        axes[1].plot(sizes, [r["metrics"]["diffusivity_APE_percent"] for r in selected],
                     "o-", label=method, color=COLORS[method])
    axes[0].set(ylabel="Relative L2 test error", title="Temperature prediction")
    axes[1].set(ylabel="Absolute percentage error (%)", title="Diffusivity estimation")
    for ax in axes:
        ax.set_xlabel(r"Number of temperature observations $N_d$")
        ax.grid(True, alpha=0.2)
        ax.legend()
    fig.suptitle("Heat equation | sensitivity to observation count")
    save_figure(fig, output, "data_size_sweep")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--beta-true", type=float, default=0.05)
    args = parser.parse_args()
    if not np.isfinite(args.beta_true) or args.beta_true <= 0:
        parser.error("--beta-true must be finite and positive")
    root = args.results_dir.resolve()
    if not root.is_dir():
        parser.error(f"Results directory does not exist: {root}")
    records = []
    for directory in (root, root / "Heat_equation"):
        for path in sorted(directory.glob("*.mat")):
            match = PATTERN.fullmatch(path.name)
            if match:
                records.append(load_result(path, match, args.beta_true))
    if not records:
        parser.error("No temperature, flux or Nd result files found. Run a training script first "
                     "or pass --results-dir with the directory containing the .mat files.")
    output = (args.output_dir or root / "plots" / "heat_equation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    comparisons = [record for record in records if record["kind"] == "seed"]
    groups = sorted({(record["measurement"], record["value"]) for record in comparisons})
    for measurement, seed in groups:
        selected = sorted(
            (record for record in comparisons
             if record["value"] == seed and record["measurement"] == measurement),
            key=lambda record: record["method"],
        )
        if len({record["method"] for record in selected}) != len(selected):
            parser.error(f"Duplicate {measurement} results for seed {seed}; "
                         "choose one policy/results directory")
        plot_seed(selected, output, seed, args.beta_true)
        print(f"{measurement.replace('_', ' ').capitalize()} seed {seed}: plotted "
              + ", ".join(record["method"] for record in selected))
    sweep = [record for record in records if record["kind"] == "Nd"]
    if sweep:
        plot_data_sweep(sweep, output)
        print(f"Data-size sweep: plotted {len(sweep)} result files")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]["metrics"]))
        writer.writeheader()
        writer.writerows(record["metrics"] for record in records)
    for record in records:
        metric = record["metrics"]
        print(f"{record['method']} {record['measurement']} {record['kind']}={record['value']}: "
              f"temperature L2={metric['relative_L2_test']:.4g}, "
              f"last logged beta^2={metric['diffusivity_last_logged']:.7g}, "
              f"APE={metric['diffusivity_APE_percent']:.4g}%, "
              f"time={metric['training_time_seconds']:.1f}s")
        if record["measurement"] in ("flux", "point_flux"):
            print(f"  Flux training L2={metric['flux_relative_L2_training']:.4g}, "
                  f"flux training RMSE={metric['flux_RMSE_training']:.4g}")
    print(f"Saved PNG figures and metrics.csv in: {output}")


if __name__ == "__main__":
    main()
