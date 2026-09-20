# Generative Visual Control

Official repository for the paper **"Generative Visual Control: Flow Matching for Vector Field Guidance of Fixed-wing UAVs from Visual Inputs"**.

> **Code release notice**
>
> The source code, training scripts, evaluation tools, trained-model interfaces, and experiment documentation are available in this repository. The real-time benchmark package includes separate PC PyTorch, Orin NX PyTorch, and Orin NX TensorRT entry points.

## Overview

Generative Visual Control (GVC) learns to generate guidance vector field parameters for fixed-wing UAVs directly from visual inputs. The method uses flow matching to model the conditional distribution of vector field parameters, enabling visual-input-driven guidance that can be deployed with the UAV dynamics model.

<p align="center">
  <img src="figures/overview.png" alt="Overview of Generative Visual Control" width="95%">
</p>

## Visual Results

The following figures compare the proposed GVC method with classical and learning-based baselines on representative fixed-wing UAV guidance tasks. The desired curve is shown in green, while the proposed GVC trajectory is shown in red.

### Basic Path Following

<p align="center">
  <img src="figures/trajectory_comparison_cassini.png" alt="Trajectory comparison on the Cassini curve" width="95%">
</p>

### Multimodal Obstacle Avoidance

<p align="center">
  <img src="figures/trajectory_comparison_quartic_obstacle.png" alt="Trajectory comparison in the multimodal quartic obstacle-avoidance experiment" width="95%">
</p>

### High-Fidelity Fixed-Wing Simulation with JSBSim

<p align="center">
  <img src="figures/trajectory_comparison_cassini_jsbsim.png" alt="Trajectory comparison with the JSBSim fixed-wing dynamics model" width="95%">
</p>

## Repository Status

The repository contains the reproducible implementation and visualization code for the experiments in the paper:

- Core GVC, CNN, Diffusion Policy, and classical vector-field controllers.
- Experiment scripts for basic path following, multimodal obstacle avoidance, and JSBSim fixed-wing simulation.
- Training, offline evaluation, closed-loop evaluation, and plotting utilities.
- Standalone benchmark code for PC PyTorch, Orin NX PyTorch, and Orin NX TensorRT.

Unity communication uses the separately maintained `AFPV_Kit` package, which
is not included in this repository.



## License

This project is released under the license provided in [LICENSE](LICENSE).
