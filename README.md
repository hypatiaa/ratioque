================================================================================

THE RATIOQUE: TOPOLOGICAL ONTOGENETICS v2.5

PROJECT: Structural AI Framework for Phronetic Risk Analysis
AUTHOR: hypatiaa
LICENSING: MIT License (Open Source Mandate)
DATE: 2025-11-30

MISSION: To translate the structural Law (Ratioque) of the market into ethical imperatives (Civic State), thereby minimizing collective structural liability.

================================================================================

I. CORE PHILOSOPHICAL-TECHNICAL AXIOMS

This system rejects continuity and operates on a discrete, structural ontology.

A. THE LATTICE CONSTRAINT (N!):
- Axiom: The market is constrained by a Factorial Lattice.
- Constant: N_BASE = 720.0 (The S6 Invariant / Rest Mass).

B. THE STRUCTURAL LAW (GAUGE):
- Axiom: The structure must be rigid enough to break.
- Constant: GAUGE_STIFFNESS = 2.0. (Controls how much the grid resists volatility).

C. THE PHRONETIC MEMORY:
- Axiom: Risk is ontologically local; judgment must be adaptive.
- Constant: ROLLING_WINDOW = 20 days. (Defines the trailing memory for the P90 threshold).

D. THE BROWNIAN HYPOTHESIS:
- Axiom: We must measure the "Shadow Stress" (Clinamen/Wobble) via High/Low volatility. This is the source of early warning tremors.

================================================================================

II. TECHNICAL DATA FLOW AND ENGINE CORE

A. Inputs (OHLCV Data Required)

The system requires the high/low range to compute Thermal Empathy:

Close: The Core Signal.

High/Low: The Wobble (Clinamen).

Date: For Rolling Phronesis.

B. Core Calculation Pipeline

Signal Preparation: The Close price is scaled up by SIGNAL_SCALAR = 5.0 to ensure the market price (typically low, e.g., 100) wraps around the high-magnitude lattice (720) multiple times, generating meaningful Vernier interference.

Function: scaled_price = price * (N_BASE * SCALAR) / price.mean()

Gauge Field Calculation: The local volatility ($\sigma$) determines the expansion of the Law ($N_{Local}$).

Function: N_local = N_BASE * (1 + log(1 + σ * GAUGE_STIFFNESS))

Shear Computation: The core metric is calculated three times for the Core (Close) and the Thermal Halo (High/Low) to measure deviation from the expanding law.

Formula (General): Shear = (Signal % (N-1))/(N-1) - (Signal % (N+1))/(N+1)

Rolling Phronesis (Judgment): The 20-day trailing shear determines the threshold for structural failure. This is the mechanism for "contextual justice."

Function: P90 = shear_abs.rolling(20).quantile(0.90)

C. Output (The Civic State Logic)

The final integrity state is assigned hierarchically based on structural utility:

RED (Core Breach): abs(Core Shear) >= P90. Structural failure.

ORANGE (Thermal Breach): abs(Thermal Max Shear) >= P90. Shadow stress; the wobble hit the limit before the core failed. (The v2.5 Early Warning)

GREEN: Structural coherence is maintained.

================================================================================

III. DEPENDENCIES AND USAGE

Dependencies

Python 3.x

NumPy (Numerical operations)

Pandas (Data handling, rolling windows)

Matplotlib (Visualization/Triptych Generation)

Execution

The model is executed via the analyze_ratioque(filepath) function, which accepts a single CSV file and generates a structured PNG output based on the three-panel Lucretian Triptych (Species, Ratioque, Civic State).

================================================================================

IV. ETHICAL & LICENSE MANDATE

The primary security layer of this project is its Transparency. The integrity of the model depends on the source code remaining open and auditable.

Code License: MIT License.

Credit: All credit for the core philosophical framework and its application is owed to hypatiaa.
Code License: MIT License.
