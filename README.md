# hackathon-qiskit
Team name: Dream

# 🧩 Quantum Key Distribution Simulator (BB84 Protocol) — Qiskit Hackathon Project

This project is an **interactive simulator and visualization tool** for the **BB84 Quantum Key Distribution (QKD)** protocol, built using **Qiskit**, **Tkinter**, and **Matplotlib**.

It demonstrates how **Alice** and **Bob** can securely establish a shared secret key using the laws of **quantum mechanics**, detecting any **eavesdropping (Eve)** attempts via measurable disturbances in the qubit error rate.

---

## 🚀 Features

✅ **End-to-End BB84 Protocol Simulation**
- Alice encodes qubits using random bits and random bases (Z/X)
- Bob measures them using his own random bases
- Sifting phase filters out mismatched bases
- QBER (Quantum Bit Error Rate) quantifies potential eavesdropping

✅ **Multiple Security Scenarios**
- 🟢 *Ideal Channel:* No noise or attacks  
- 🔵 *Noisy Channel:* Depolarizing noise (random per iteration, 1–5%)  
- 🔴 *Eavesdropping Attack:* Eve intercepts and resends qubits

✅ **Dynamic Visualizations**
- 📊 **QBER Bar Chart** – compares error rates for Ideal, Noise, and Eve scenarios  
- 📈 **QBER Evolution** – tracks error rate trends across multiple iterations  
- 🥧 **Sifting Efficiency Pie Chart** – shows how many qubits are kept/discarded after basis comparison  

✅ **Message Encryption Demo**
- Demonstrates using the generated quantum key to encrypt/decrypt a text message via **One-Time Pad (XOR)**.

✅ **Interactive GUI**
- Built with **Tkinter** for simple button-based iteration (“Run Security Audit”)  
- Uses **Matplotlib embedded canvases** for live, per-iteration visualization updates.

---

## 🧮 Background

Classical cryptography (RSA, ECC) relies on mathematical hardness assumptions.  
Quantum computers threaten these by efficiently solving problems like integer factorization.

The **BB84 Protocol** (Bennett & Brassard, 1984) instead leverages *quantum physics*:
- Measurement disturbs the quantum state.
- Eavesdropping introduces detectable noise (increased QBER).
- If QBER exceeds a threshold (≈11%), Alice and Bob discard the key.

This gives **information-theoretic security** — safe even against quantum computers.

---

## 🖥️ Visualizations Explained

| Visualization | Description |
|----------------|-------------|
| **QBER Bar Chart** | Displays average error rates (QBER) under Ideal, Noisy, and Eavesdropping scenarios. A high QBER immediately signals a possible attack. |
| **QBER Evolution** | Tracks QBER across multiple protocol iterations to demonstrate stability and repeatability. |
| **Sifting Efficiency (Pie Chart)** | Shows how many qubits are kept after basis comparison (expected ≈50%) — visualizing the inherent efficiency of the BB84 process. |

---

## 🧰 Tech Stack

- 🧪 **Qiskit** – Quantum circuit creation, simulation, and noise modeling  
- ⚙️ **Python 3.10+**  
- 📈 **Matplotlib** – Real-time chart visualization  
- 🪟 **Tkinter** – Interactive graphical user interface  
- 🎲 **NumPy** – Randomized bit and basis generation

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Tomler123/hackathon-qiskit.git
cd hackathon-qiskit
