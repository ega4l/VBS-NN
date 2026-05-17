# VertexByteStream (VBS-NN) — Docker Execution Guide

This repository contains the official Docker deployment setups for evaluating the **VertexByteStream (VBS-NN)** architecture. With these isolated environments, you can reproduce our extreme **512k context length needle-in-a-haystack stress tests** on local hardware using either AMD (ROCm) or NVIDIA (CUDA) graphics cards.

⚠️ **Disclaimer:** This is the very first trial release of the architecture, distributed strictly for demo, evaluation, and non-commercial research purposes. 

---

## 🖥️ System Requirements

Before running the stress tests, ensure your local environment meets the following criteria:

*   **OS:** Linux (Ubuntu 22.04+ or Debian Trixie recommended).
*   **Docker:** Installed and running.
*   **VRAM:** At least 12 GB of VRAM is recommended to comfortably handle extreme context scaling due to our integrated dynamic gradient checkpointing (`GC:True`).
*   **GPU Drivers:**
    *   **For AMD:** Properly installed ROCm kernel drivers on the host machine.
    *   **For NVIDIA:** Installed `NVIDIA Container Toolkit` on the host to enable the `--gpus` flag inside Docker.

---

## 🚀 Quick Start (Docker Installation)

Follow these steps to clone, build, and deploy the architecture evaluation container based on your GPU hardware vendor.

### 1. Clone the Repository
Open your terminal, clone the repository, and navigate into the root directory:
```bash
git clone [https://github.com/ega4l/VBS-NN.git](https://github.com/ega4l/VBS-NN.git) && cd VBS-NN/code

```

---

### 2. Deployment on AMD GPUs (ROCm)

The AMD setup utilizes direct device sharing to communicate with Radeon cards.

#### a. Build the ROCm Image:

```bash
docker build -f docker/Dockerfile.rocm -t local-vbs.nn:rocm .

```

#### b. Run the 512k Context Needle Stress-Test:

```bash
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --name needle-test \
  local-vbs.nn:rocm

```

> 💡 *Note: `--ipc=host` is critical for PyTorch to allow high-throughput shared memory allocation during deep hierarchical gradient aggregation.*

---

### 3. Deployment on NVIDIA GPUs (CUDA)

The NVIDIA setup relies on the standard unified container runtime for Pascal, Ampere, Ada Lovelace, or Hopper architectures.

#### a. Build the CUDA Image:

```bash
docker build -f docker/Dockerfile.cuda -t local-vbs.nn:cuda .

```

#### b. Run the 512k Context Needle Stress-Test:

```bash
docker run -it --rm \
  --gpus all \
  --ipc=host \
  --name needle-test \
  local-vbs.nn:cuda

```

---

## 📄 License

This project uses a dual-licensing model to protect core software assets while allowing public validation of academic data:

* **Source Code & Model Weights:** Licensed under a custom **Evaluation and Non-Commercial Research License**. Permitted strictly for academic research, educational use, and private benchmarks. Commercial deployment or derivation is strictly prohibited. Please see the [LICENSE](https://www.google.com/search?q=LICENSE) file in the main repository for full terms.
* **Documentation & Logs:** Licensed under the Creative Commons Attribution 4.0 International License ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). You are welcome to share and cite our benchmark data as long as proper attribution is provided.

```

```
