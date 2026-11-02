# Supplementary Experiments

Run commands from the repository root. Image capture and closed-loop tests
require the experiment-1 Unity scene and `AFPV_Kit`.

## VisLoc-GVF

The position CNN predicts normalized `(x, y)` for the analytical Cassini GVF.
World coordinates are `[576, 275] + 100 * (x, y)`.

Training requires
`src/path_following/vector_field_with_gradients_cassini.csv` and
`src/path_following/rectified_panorama_cassini/`. The CSV must contain
`frame_id`, `x`, and `y`; image names use `rectified_color_panorama_0001.jpg`.

```powershell
python -m src.baseline.position_localization
python -m src.baseline.position_localization --evaluate
```

The checkpoint is saved to `src/baseline/models/position_cassini_cnn/position_best.pth`.
Full-grid evaluation reports training fit; held-out accuracy is evaluated below.
Set `CONTROL_METHOD = "position_gvf"` in `exp1_basic_path_following.py`, then run:

```powershell
python exp1_basic_path_following.py
python -m src.baseline.diagnose_position_gvf
python plot_data_cassini.py
```

For JSBSim, install the Skywalker X8 model in the JSBSim environment,
start the Unity scene, and run:

```powershell
python exp3_high_fidelity_fixed_wing_jsbsim.py
python plot_data_cassini_jsbsim.py
```

Results use `data/cassini_path_following_data/position_gvf/` and
`jsbsim_data/cassini_path_following_data/position_gvf/`. The plotting scripts
include available baseline logs and export localization and timing statistics.

## Visual Robustness

Provide the following checkpoints before running the four-method comparison:

```text
src/generative_model/models/flow_matching_cassini_models/flow_matching_policy_final.pth
src/baseline/models/cnn_cassini_models/cnn_final.pth
src/baseline/models/position_cassini_cnn/position_best.pth
src/generative_model/models/diffusion_policy_cassini_models/diffusion_policy_final.pth
```

`visual_robustness/common.py` sets the image conditions, seeds, and repetitions.
Each run uses 2,000 frames; tracking statistics use `[500:2000]`.

```powershell
python -m supplementary.visual_robustness.collect_heldout
python -m supplementary.visual_robustness.run_experiment1
python -m supplementary.visual_robustness.retry_incomplete
python -m supplementary.visual_robustness.evaluate_offline
python -m supplementary.visual_robustness.report
```

Closed-loop logs and reports are written to `data/visual_robustness/run01/`;
the shared held-out images are in `data/visual_robustness/offline_heldout/`.
The retry script records new attempts for interrupted or timed-out trials.
Change `RUN_TAG` when changing the protocol.

| File in `visual_robustness/` | Purpose |
| --- | --- |
| `common.py` | Settings and image corruptions. |
| `camera.py`, `bridge.py` | Six-camera capture. |
| `collect_heldout.py` | Collect held-out images. |
| `run_experiment1.py` | Run closed-loop trials. |
| `retry_incomplete.py` | Retry Unity transport failures. |
| `evaluate_offline.py` | Evaluate GVF direction errors. |
| `report.py` | Generate tables and figures. |

## GVF Diagnostics

Collect off-grid images, then evaluate the GVC and localization models:

```powershell
python -m supplementary.gvf_diagnostics.collect
python -m supplementary.gvf_diagnostics.evaluate
python -m supplementary.gvf_diagnostics.report
python -m supplementary.gvf_diagnostics.trajectory
python -m supplementary.gvf_diagnostics.latent_consistency
python -m supplementary.gvf_diagnostics.one_meter
python -m supplementary.gvf_diagnostics.analytic_step
```

Run in the listed order. `collect` and `one_meter` require Unity.
Trajectory analysis reads `data/visual_robustness/run01/gvc/clean/rep_00/`;
reports also use available experiment-3 logs.

Outputs use `data/gvf_diagnostics/<diagnostic>/run01/`.
These sampled checks do not certify uniform error bounds.

| File in `gvf_diagnostics/` | Purpose |
| --- | --- |
| `settings.py` | Sampling and evaluation settings. |
| `collect.py` | Collect off-grid images and local stencils. |
| `evaluate.py` | Evaluate direction and gradient consistency. |
| `report.py` | Generate reports and inspect JSBSim logs. |
| `trajectory.py` | Analyze a recorded clean trajectory. |
| `latent_consistency.py` | Compare fixed and resampled latents. |
| `one_meter.py` | Check one-meter direction changes. |
| `analytic_step.py` | Compare with the analytical Cassini field. |
