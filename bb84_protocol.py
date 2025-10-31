from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer.noise import NoiseModel, depolarizing_error
import numpy as np
from qiskit_aer import Aer
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# --- GLOBAL CONSTANTS ---
BASES = ['Z', 'X']
KEY_LENGTH = 2000
TOTAL_ITERATIONS = 10
NOISE_PROB = 0.05
SECURITY_THRESHOLD = 0.11
ALICE_SECRET_MESSAGE = "The blueprint is secure."  # Message to encrypt/decrypt


# Helper function to convert the bit/basis pair to a quantum state symbol
def get_state_symbol(bit, basis):
    if basis == 'Z':
        return "|0>" if bit == 0 else "|1>"
    if basis == 'X':
        return "|+>" if bit == 0 else "|->"
    return "??"


# --- MESSAGE ENCRYPTION/DECRYPTION ---
def text_to_bits(text):
    return [int(b) for char in text for b in bin(ord(char))[2:].zfill(8)]


def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = "".join(map(str, bits[i:i + 8]))
        chars.append(chr(int(byte, 2)))
    return "".join(chars)


def xor_message(bit_message, bit_key):
    key_length = len(bit_key)
    full_key = [bit_key[i % key_length] for i in range(len(bit_message))]
    return [m ^ k for m, k in zip(bit_message, full_key)]


# --- CORE BB84 FUNCTIONS ---
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
    qc_forwarded = alice_encode(eve_measured_bit, eve_basis)
    return qc_forwarded


def create_noise_model(error_probability=NOISE_PROB):
    noise_model = NoiseModel()
    error = depolarizing_error(error_probability, 1)
    noise_model.add_quantum_error(error, ['x', 'h'], [0])
    return noise_model


def execute_protocol_run(key_length, scenario='ideal', noise_prob=0.0):
    alice_bits = np.random.randint(2, size=key_length)
    alice_bases = [random.choice(BASES) for _ in range(key_length)]
    bob_bases = [random.choice(BASES) for _ in range(key_length)]
    bob_measured_bits = []

    sample_data = []
    sample_circuit = None

    noise_model = None
    if scenario == 'noise':
        noise_model = create_noise_model(noise_prob)

    simulator = Aer.get_backend('qasm_simulator')

    for i in range(key_length):
        qc_in = alice_encode(alice_bits[i], alice_bases[i])
        qc_to_measure = qc_in.copy()

        if scenario == 'eve':
            qc_to_measure = eve_eavesdrops(qc_in)

        qc_to_measure.add_register(ClassicalRegister(1, name='cbit'))
        if bob_bases[i] == 'X':
            qc_to_measure.h(0)
        qc_to_measure.measure(0, 0)

        job = simulator.run(qc_to_measure, shots=1, noise_model=noise_model)
        result = job.result()
        counts = result.get_counts()
        measured_bit = int(list(counts.keys())[0])
        bob_measured_bits.append(measured_bit)

        if i < 3:
            sift_status = "Match" if alice_bases[i] == bob_bases[i] else "Mismatch"
            state_symbol = get_state_symbol(alice_bits[i], alice_bases[i])
            sample_data.append({
                'A_bit': alice_bits[i],
                'A_basis': alice_bases[i],
                'Q_state': state_symbol,
                'B_basis': bob_bases[i],
                'B_bit': measured_bit,
                'Status': sift_status
            })
            if i == 0:
                sample_circuit = qc_in.draw(output='text', idle_wires=False)

    sifted_alice_key = []
    sifted_bob_key = []

    for i in range(key_length):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice_key.append(alice_bits[i])
            sifted_bob_key.append(bob_measured_bits[i])

    sifted_alice_key = np.array(sifted_alice_key)
    sifted_bob_key = np.array(sifted_bob_key)

    errors = np.sum(sifted_alice_key != sifted_bob_key)
    sifted_length = len(sifted_alice_key)

    qber = errors / sifted_length if sifted_length > 0 else 0.0

    return qber, sifted_length, errors, sample_data, sample_circuit, sifted_alice_key, sifted_bob_key


# --- GUI APPLICATION ---
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
        control_frame = ttk.Frame(self.master, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.status_text = tk.StringVar(value="Ready to run first security audit.")
        status_label = ttk.Label(control_frame, textvariable=self.status_text, wraplength=400, justify=tk.LEFT)
        status_label.pack(side=tk.LEFT, padx=10)

        self.generate_button = ttk.Button(control_frame, text=f"Run Security Audit 1/{TOTAL_ITERATIONS}",
                                          command=self.run_next_iteration)
        self.generate_button.pack(side=tk.RIGHT, padx=5)

        ttk.Button(control_frame, text="Exit", command=self.master.quit).pack(side=tk.RIGHT, padx=5)

        # --- Tabs for different visualizations ---
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: QBER Bar Plot
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 4))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.notebook)
        self.notebook.add(self.canvas1.get_tk_widget(), text="QBER Bar")

        # Tab 2: QBER Evolution
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 4))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.notebook)
        self.notebook.add(self.canvas2.get_tk_widget(), text="QBER Evolution")

        # Tab 3: Sifting Efficiency (Pie Chart)
        self.fig3, self.ax3 = plt.subplots(figsize=(5, 4))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.notebook)
        self.notebook.add(self.canvas3.get_tk_widget(), text="Sifting Efficiency")

    def initialize_plots(self):
        self.ax1.clear()
        self.ax1.set_title('QBER Audit (Iteration 0)')
        self.ax1.set_ylabel('Average Quantum Bit Error Rate (QBER)')
        self.ax1.set_ylim(0, 0.3)
        self.ax1.axhline(y=SECURITY_THRESHOLD, color='orange', linestyle='--', label='Security Threshold')
        self.ax1.set_xticks(range(3))
        self.ax1.set_xticklabels(['Ideal', f'Noise ({NOISE_PROB * 100:.0f}%)', 'Eve'])
        self.ax1.legend()
        self.canvas1.draw()

    def print_initial_checks(self):
        alice_bit = 0
        alice_basis = 'X'
        bob_basis = 'Z'
        mismatch_results = [bob_measure(alice_encode(alice_bit, alice_basis), bob_basis)[0] for _ in range(100)]
        correct_count = mismatch_results.count(alice_bit)
        print("\n--- INITIAL QUANTUM CHECK ---")
        print(f"Bob measured |+> in Z-basis -> Correct {correct_count}/100 (Expected ~50%)")
        print("-" * 50)

    def run_next_iteration(self):
        if self.iteration >= TOTAL_ITERATIONS:
            self.status_text.set("Simulation complete.")
            self.generate_button.config(state=tk.DISABLED, text="All Audits Complete")
            return

        self.iteration += 1

        # --- Run all scenarios ---
        qber_ideal, sl_ideal, _, _, _, _, _ = execute_protocol_run(KEY_LENGTH, 'ideal')
        noise_this_round = random.randint(10, 50) / 1000.0
        qber_noise, _, _, *_ = execute_protocol_run(KEY_LENGTH, 'noise', noise_prob=noise_this_round)
        qber_eve, _, _, *_ = execute_protocol_run(KEY_LENGTH, 'eve')

        self.qber_data['ideal'].append(qber_ideal)
        self.qber_data['noise'].append(qber_noise)
        self.qber_data['eve'].append(qber_eve)

        self.update_qber_bar(qber_ideal, qber_noise, qber_eve, noise_level=noise_this_round)
        self.update_qber_evolution()
        self.update_sifting_pie(sl_ideal)

        self.status_text.set(f"Audit {self.iteration}/{TOTAL_ITERATIONS} complete.")
        if self.iteration < TOTAL_ITERATIONS:
            self.generate_button.config(text=f"Run Security Audit {self.iteration + 1}/{TOTAL_ITERATIONS}")
        else:
            self.generate_button.config(text="SECURITY AUDIT COMPLETE!", state=tk.DISABLED)
            self.status_text.set(f"Simulation complete after {TOTAL_ITERATIONS} audits. Final averages stable.")

    def update_qber_bar(self, q1, q2, q3, noise_level=None):
        qbers = [q1, q2, q3]
        labels = [
            'Ideal',
            f'Channel Noise p={noise_level:.3f}' if noise_level else f'Noise ({NOISE_PROB * 100:.0f}% )',
            'Eve'
        ]
        colors = ['green' if q < SECURITY_THRESHOLD else 'red' for q in qbers]

        self.ax1.clear()
        bars = self.ax1.bar(labels, qbers, color=colors)
        self.ax1.axhline(y=SECURITY_THRESHOLD, color='orange', linestyle='--', label='Security Threshold')
        self.ax1.set_ylim(0, 0.3)
        self.ax1.set_title(f'Iteration {self.iteration} - QBER Summary')
        for bar in bars:
            yval = bar.get_height()
            self.ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom')
        self.ax1.legend()
        self.canvas1.draw()

    def update_qber_evolution(self):
        self.ax2.clear()
        x = np.arange(1, self.iteration + 1)
        self.ax2.plot(x, self.qber_data['ideal'], 'g-o', label='Ideal')
        self.ax2.plot(x, self.qber_data['noise'], 'b-o', label='Noise')
        self.ax2.plot(x, self.qber_data['eve'], 'r-o', label='Eve')
        self.ax2.axhline(y=SECURITY_THRESHOLD, color='orange', linestyle='--', label='Security Threshold')
        self.ax2.set_title("QBER Evolution Across Iterations")
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("QBER")
        self.ax2.set_ylim(0, 0.3)
        self.ax2.legend()
        self.canvas2.draw()

    def update_sifting_pie(self, sifted_length):
        self.ax3.clear()
        kept = sifted_length
        discarded = KEY_LENGTH - sifted_length
        total = kept + discarded
        kept_pct = (kept / total) * 100 if total > 0 else 0.0

        wedges, texts, autotexts = self.ax3.pie(
            [kept, discarded],
            labels=['Kept (Matching Bases)', 'Discarded'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#4CAF50', '#FF7043']
        )
        self.ax3.text(0, -1.25, f"Sifting Efficiency: {kept_pct:.2f}%", ha='center', va='center',
                      fontsize=11, fontweight='bold')
        self.ax3.set_title("Sifting Efficiency (Basis Match Rate)")
        self.canvas3.draw()


if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = BB84App(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
