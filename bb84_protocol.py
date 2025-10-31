# bb84_protocol.py (CLEAN FINAL VERSION)

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer.noise import NoiseModel, depolarizing_error
import numpy as np
from qiskit_aer import Aer, AerSimulator
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import yfinance as yf
import pandas as pd
import warnings

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============== Hybrid Quantum–Market RNG (FAST) ==============

STOCKS = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "NFLX", "INTC", "AMD"]

def quantum_seed():
    """Generate an 8-bit random integer using Qiskit Aer simulator."""
    qc = QuantumCircuit(8, 8)
    qc.h(range(8))
    qc.measure(range(8), range(8))
    sim = AerSimulator()
    result = sim.run(qc, shots=1).result()
    bitstring = list(result.get_counts().keys())[0]
    return int(bitstring, 2)

def _fetch_last_cent_parity(symbol: str):
    """Return 0 if last cent digit even, 1 if odd."""
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=False)
        if data is None or len(data) == 0:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if "Close" not in data.columns or len(data["Close"]) == 0:
            return None
        last_price = float(data["Close"].iloc[-1])
        last_cent_digit = int(str(int(round(last_price * 100.0)))[-1])
        return 0 if (last_cent_digit % 2 == 0) else 1
    except Exception:
        return None

def build_market_parity_map(symbols):
    """Fetch parity bits (even/odd last cent digit) for all stocks."""
    parity = {}
    for sym in symbols:
        b = _fetch_last_cent_parity(sym)
        if b is not None:
            parity[sym] = b
    print(f"[Market Parity Map] {parity}")
    return parity

def hybrid_quantum_market_random_bits(n_bits=10):
    """Hybrid RNG: quantum picks stock; stock's price last cent defines bit parity."""
    parity_map = build_market_parity_map(STOCKS)
    bits = []
    for _ in range(n_bits):
        seed = quantum_seed()
        chosen = STOCKS[seed % len(STOCKS)]
        bit = parity_map.get(chosen, seed % 2)
        bits.append(bit)
    print(f"[Hybrid RNG] First bits: {bits[:16]}")
    return bits

# ============== BB84 SIMULATOR ==============

BASES = ['Z', 'X']
KEY_LENGTH = 2000
TOTAL_ITERATIONS = 10
NOISE_PROB = 0.05
SECURITY_THRESHOLD = 0.11

def get_state_symbol(bit, basis):
    if basis == 'Z':
        return "|0>" if bit == 0 else "|1>"
    if basis == 'X':
        return "|+>" if bit == 0 else "|->"
    return "??"

def alice_encode(bit, basis):
    qc = QuantumCircuit(1)
    if bit == 1:
        qc.x(0)
    if basis == 'X':
        qc.h(0)
    return qc

def bob_measure(qc_in, basis):
    qc_out = qc_in.copy()
    cbit = ClassicalRegister(1, name='cbit')
    qc_out.add_register(cbit)
    if basis == 'X':
        qc_out.h(0)
    qc_out.measure(0, 0)
    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(qc_out, shots=1)
    result = job.result()
    counts = result.get_counts()
    measured_bit = int(list(counts.keys())[0])
    return measured_bit, qc_out

def eve_eavesdrops(qc_in):
    eve_basis = random.choice(BASES)
    eve_measured_bit, _ = bob_measure(qc_in, eve_basis)
    return alice_encode(eve_measured_bit, eve_basis)

def create_noise_model(error_probability=NOISE_PROB):
    noise_model = NoiseModel()
    error = depolarizing_error(error_probability, 1)
    noise_model.add_quantum_error(error, ['x', 'h'], [0])
    return noise_model

def execute_protocol_run(key_length, scenario='ideal', noise_prob=0.0):
    alice_bits = np.array(hybrid_quantum_market_random_bits(key_length))
    alice_bases = [random.choice(BASES) for _ in range(key_length)]
    bob_bases = [random.choice(BASES) for _ in range(key_length)]
    bob_measured_bits = []
    sample_data, sample_circuit = [], None

    noise_model = create_noise_model(noise_prob) if scenario == 'noise' else None
    simulator = Aer.get_backend('qasm_simulator')

    for i in range(key_length):
        qc_in = alice_encode(alice_bits[i], alice_bases[i])
        qc_to_measure = eve_eavesdrops(qc_in) if scenario == 'eve' else qc_in.copy()
        qc_to_measure.add_register(ClassicalRegister(1, name='cbit'))
        if bob_bases[i] == 'X':
            qc_to_measure.h(0)
        qc_to_measure.measure(0, 0)

        job = simulator.run(qc_to_measure, shots=1, noise_model=noise_model)
        counts = job.result().get_counts()
        measured_bit = int(list(counts.keys())[0])
        bob_measured_bits.append(measured_bit)

        if i < 3:
            sift_status = "Match" if alice_bases[i] == bob_bases[i] else "Mismatch"
            sample_data.append({
                'A_bit': int(alice_bits[i]),
                'A_basis': alice_bases[i],
                'Q_state': get_state_symbol(alice_bits[i], alice_bases[i]),
                'B_basis': bob_bases[i],
                'B_bit': measured_bit,
                'Status': sift_status
            })
            if i == 0:
                sample_circuit = qc_in.draw(output='text', idle_wires=False)

    sifted = [(a, b) for a, b, ab, bb in zip(alice_bits, bob_measured_bits, alice_bases, bob_bases) if ab == bb]
    if sifted:
        a_bits, b_bits = zip(*sifted)
    else:
        a_bits, b_bits = [], []

    errors = sum(a != b for a, b in zip(a_bits, b_bits))
    sifted_len = len(a_bits)
    qber = errors / sifted_len if sifted_len else 0.0

    return qber, sifted_len, errors, sample_data, sample_circuit, a_bits, b_bits

# ============== GUI APPLICATION ==============

class BB84App:
    def __init__(self, master):
        self.master = master
        master.title("Interactive BB84 QKD Simulator: Corporate Spy Edition")

        self.iteration = 0
        self.qber_data = {'ideal': [], 'noise': [], 'eve': []}
        self.setup_ui()
        self.initialize_plots()
        self.print_initial_checks()

    def setup_ui(self):
        control = ttk.Frame(self.master, padding="10")
        control.pack(side=tk.TOP, fill=tk.X)

        self.status_text = tk.StringVar(value="Ready to run first security audit.")
        ttk.Label(control, textvariable=self.status_text, wraplength=400, justify=tk.LEFT).pack(side=tk.LEFT, padx=10)
        ttk.Button(control, text=f"Run Security Audit 1/{TOTAL_ITERATIONS}", command=self.run_next_iteration)\
            .pack(side=tk.RIGHT, padx=5)
        ttk.Button(control, text="Exit", command=self.master.quit).pack(side=tk.RIGHT, padx=5)

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.fig1, self.ax1 = plt.subplots(figsize=(6, 4))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.notebook)
        self.notebook.add(self.canvas1.get_tk_widget(), text="QBER Bar")

        self.fig2, self.ax2 = plt.subplots(figsize=(6, 4))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.notebook)
        self.notebook.add(self.canvas2.get_tk_widget(), text="QBER Evolution")

        self.fig3, self.ax3 = plt.subplots(figsize=(5, 4))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.notebook)
        self.notebook.add(self.canvas3.get_tk_widget(), text="Sifting Efficiency")

    def initialize_plots(self):
        self.ax1.clear()
        self.ax1.set_title('QBER Audit (Iteration 0)')
        self.ax1.set_ylabel('Quantum Bit Error Rate (QBER)')
        self.ax1.set_ylim(0, 0.3)
        self.ax1.axhline(y=SECURITY_THRESHOLD, color='orange', linestyle='--', label='Security Threshold')
        self.ax1.set_xticks(range(3))
        self.ax1.set_xticklabels(['Ideal', f'Noise ({NOISE_PROB*100:.0f}%)', 'Eve'])
        self.ax1.legend()
        self.canvas1.draw()

    def print_initial_checks(self):
        alice_bit, alice_basis, bob_basis = 0, 'X', 'Z'
        results = [bob_measure(alice_encode(alice_bit, alice_basis), bob_basis)[0] for _ in range(100)]
        correct = results.count(alice_bit)
        print("\n--- INITIAL QUANTUM CHECK ---")
        print(f"Bob measured |+> in Z-basis -> Correct {correct}/100 (Expected ~50%)")
        print("-" * 50)

    def run_next_iteration(self):
        if self.iteration >= TOTAL_ITERATIONS:
            self.status_text.set("Simulation complete.")
            return

        self.iteration += 1
        qber_ideal, sl_ideal, *_ = execute_protocol_run(KEY_LENGTH, 'ideal')
        noise_now = random.randint(10, 50) / 1000.0
        qber_noise, *_ = execute_protocol_run(KEY_LENGTH, 'noise', noise_prob=noise_now)
        qber_eve, *_ = execute_protocol_run(KEY_LENGTH, 'eve')

        self.qber_data['ideal'].append(qber_ideal)
        self.qber_data['noise'].append(qber_noise)
        self.qber_data['eve'].append(qber_eve)

        self.update_qber_bar(qber_ideal, qber_noise, qber_eve, noise_level=noise_now)
        self.update_qber_evolution()
        self.update_sifting_pie(sl_ideal)

        self.status_text.set(f"Audit {self.iteration}/{TOTAL_ITERATIONS} complete.")
        if self.iteration == TOTAL_ITERATIONS:
            self.status_text.set("All audits complete. Simulation finished.")

    def update_qber_bar(self, q1, q2, q3, noise_level=None):
        qbers = [q1, q2, q3]
        labels = ['Ideal', f'Noise p={noise_level:.3f}', 'Eve']
        colors = ['green' if q < SECURITY_THRESHOLD else 'red' for q in qbers]
        self.ax1.clear()
        bars = self.ax1.bar(labels, qbers, color=colors)
        self.ax1.axhline(y=SECURITY_THRESHOLD, color='orange', linestyle='--', label='Threshold')
        self.ax1.set_ylim(0, 0.3)
        for bar in bars:
            yval = bar.get_height()
            self.ax1.text(bar.get_x()+bar.get_width()/2, yval+0.005, f'{yval:.3f}', ha='center')
        self.ax1.legend()
        self.canvas1.draw()

    def update_qber_evolution(self):
        self.ax2.clear()
        x = np.arange(1, self.iteration + 1)
        self.ax2.plot(x, self.qber_data['ideal'], 'g-o', label='Ideal')
        self.ax2.plot(x, self.qber_data['noise'], 'b-o', label='Noise')
        self.ax2.plot(x, self.qber_data['eve'], 'r-o', label='Eve')
        self.ax2.axhline(y=SECURITY_THRESHOLD, color='orange', linestyle='--', label='Threshold')
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("QBER")
        self.ax2.legend()
        self.canvas2.draw()

    def update_sifting_pie(self, sifted_len):
        self.ax3.clear()
        kept, discarded = sifted_len, KEY_LENGTH - sifted_len
        total = kept + discarded
        kept_pct = (kept / total) * 100 if total > 0 else 0.0
        self.ax3.pie([kept, discarded],
                     labels=['Kept (Matching Bases)', 'Discarded'],
                     autopct='%1.1f%%', startangle=90,
                     colors=['#4CAF50', '#FF7043'])
        self.ax3.text(0, -1.25, f"Sifting Efficiency: {kept_pct:.2f}%", ha='center', fontsize=11, fontweight='bold')
        self.canvas3.draw()


if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = BB84App(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
