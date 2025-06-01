# 🌌 N_Body_Sim

A 2D gravitational N-body simulation implemented in both CPU (serial C++) and GPU (CUDA). It simulates the interactions of particles under a short-range repulsive force model. Includes correctness checking and configurable runtime parameters.

---

## 🧠 Description

This simulation models `N` particles interacting through a short-range repulsive force. It features both a CPU (serial) version and a GPU-accelerated CUDA implementation to compare correctness and performance. The simulation uses Velocity Verlet integration for improved numerical stability and allows configurable particle count and randomness.

---

## 🔧 Tech Stack

- CUDA C++
- Serial C++
- Python (for correctness check)

---

## ✨ Key Features

- ✅ Serial C++ implementation in `serial.cpp`
- ✅ CUDA-parallel implementation in `gpu.cu`
- ✅ Particle interaction with repulsion and wall bounce
- ✅ Position and velocity updates with Velocity Verlet
- ✅ Configurable particle count and seed from command line
- ✅ Python script `correctness-check.py` to compare CPU and GPU outputs

---

## 📂 Folder Structure

```
.
├── Makefile               # Builds both serial and GPU versions
├── common.h               # Shared constants, structs
├── main.cu                # CUDA version entry point
├── gpu.cu                 # CUDA simulation functions
├── serial.cpp             # Serial version
├── correctness-check.py   # Output comparison tool
```

---

## 🛠️ Setup & Usage

### ✅ Requirements

- CUDA Toolkit 12.x+
- NVIDIA GPU with Compute Capability ≥ 3.5
- Python 3.x

### ⚙️ Build

```bash
make
```

This creates two binaries:
- `nbody` (CUDA)
- `serial` (CPU)

### ▶️ Run

```bash
./nbody  -n 1000 -s 1 -o gpu_output.txt
./serial -n 1000 -s 1 -o serial_output.txt
python3 correctness-check.py gpu_output.txt serial_output.txt
```

Arguments:
- `-n`: number of particles
- `-s`: seed for random generator
- `-o`: output filename

---

## 📈 Sample Output

```
Simulation Time = 0.9458 seconds for 1000 particles.
```

Correctness script checks positional match between CPU and GPU runs.

---

## 👥 Contributors

- Can Ercan (@cann-e)

---

