"""
THE RATIOQUE: TOPOLOGICAL ONTOGENETICS v2.5
================================================================================
Author: hypatiaa -- MIT License Copyright 2025
Classification: Corporate Integrity Monitor / Civic Phronesis
Philosophical Basis: "Naturae species ratioque" (Lucretius) + "Eichinvarianz" (Weyl)

CORE INSIGHT (v2.5):
The Atom is not a Point. The signal is a vibrating string (High/Low). 
We measure the "Thermal Shear" (Brownian Motion) to detect micro-fractures
before the core breaks.

AXIOMS:
  A1: Market = signal on factorial lattice (N! = 720)
  A2: Grid breathes with volatility, maintains stiffness (k=2.0)
  A3: The "Wobble" (High-Low) is the Clinamen (Thermal Energy)
  A4: Signal amplified (x5.0) for Vernier wrapping
  A5: Risk defined by trailing 20-day memory
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import sys
import os

# --- SYSTEM KERNEL CONFIGURATION (v2.5) ---
class KernelConfig:
    N_BASE = 720.0          # The S6 Invariant
    GAUGE_STIFFNESS = 2.0   # Rigid Law
    SIGNAL_SCALAR = 5.0     # High Amplification
    ROLLING_WINDOW = 20     # Fast Memory
    VOL_WINDOW = 14         # Volatility Lookback

def analyze_ratioque(filepath):
    # --- [1] LOAD & STANDARDIZE ---
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Normalize Column Names
    cols_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols_map)
    
    # Identify Price Columns (OHLC)
    # We prioritize finding High/Low for the Brownian calc
    col_close = next((c for c in df.columns if c in ['close', 'price', 'adj close']), None)
    col_high  = next((c for c in df.columns if c in ['high']), col_close) # Fallback to close
    col_low   = next((c for c in df.columns if c in ['low']), col_close)  # Fallback to close
    
    if not col_close:
        print("Error: No price/close column found.")
        return

    # Handle Dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

    print(f"Analyzing {len(df)} periods for {filepath} (Brownian Mode)...")

    # --- [2] THE WEYL-MILLER ENGINE ---
    
    # A. Volatility (Temperature of the Core)
    returns = df[col_close].pct_change()
    volatility = returns.rolling(window=KernelConfig.VOL_WINDOW).std().fillna(0)
    
    # B. The Stiff Gauge (Law)
    gauge_potential = np.log1p(volatility * KernelConfig.GAUGE_STIFFNESS) 
    n_local = KernelConfig.N_BASE * (1 + gauge_potential)
    df['grid_strength'] = n_local

    # C. Signal Scaling (Interference)
    # We apply the scalar to Close, High, and Low
    scalar_factor = (KernelConfig.N_BASE * KernelConfig.SIGNAL_SCALAR) / df[col_close].mean()
    
    s_close = df[col_close] * scalar_factor
    s_high  = df[col_high]  * scalar_factor
    s_low   = df[col_low]   * scalar_factor
    
    # D. Covariant Shear Calculation (The Vernier Function)
    def calc_shear(signal, n_loc):
        grid_f = n_loc - 1
        grid_p = n_loc + 1
        
        # Modulo Wrapping
        phase_f = signal % grid_f
        phase_p = signal % grid_p
        
        norm_f = phase_f / grid_f
        norm_p = phase_p / grid_p
        
        return norm_f - norm_p

    # We calculate Shear for the Center (Close) and the Extremes (High/Low)
    df['shear_core'] = calc_shear(s_close, n_local)
    df['shear_high'] = calc_shear(s_high, n_local)
    df['shear_low']  = calc_shear(s_low, n_local)
    
    # Determine the "Thermal Halo" (Max deviation of the wobble)
    # The Halo is the range between Shear_High and Shear_Low
    # Note: Because of modulo, High price doesn't always mean High shear. 
    # We take the max absolute deviation from the core.
    
    # --- [3] ROLLING PHRONESIS (Adaptive Thresholds) ---
    shear_abs = df['shear_core'].abs()
    
    # Rolling Percentiles based on the CORE signal
    df['p90'] = shear_abs.rolling(window=KernelConfig.ROLLING_WINDOW, min_periods=10).quantile(0.90)
    df['p75'] = shear_abs.rolling(window=KernelConfig.ROLLING_WINDOW, min_periods=10).quantile(0.75)
    df['p50'] = shear_abs.rolling(window=KernelConfig.ROLLING_WINDOW, min_periods=10).quantile(0.50)
    
    # Fill gaps
    df['p90'] = df['p90'].ffill().bfill()
    df['p75'] = df['p75'].ffill().bfill()
    df['p50'] = df['p50'].ffill().bfill()

    # Integrity State Logic (v2.5 Brownian Update)
    def get_state(row):
        core = abs(row['shear_core'])
        # Check Thermal Extremes (Did the wobble breach the limit?)
        therm_max = max(abs(row['shear_high']), abs(row['shear_low']))
        
        thresh_red = row['p90']
        
        if core >= thresh_red: return '#FF0000'    # RED (Core Breach)
        if therm_max >= thresh_red: return '#FF8800' # ORANGE (Thermal Breach / Tremor)
        
        if core >= row['p75']: return '#FF8800'    # ORANGE (Core Stress)
        if core >= row['p50']: return '#FFFF00'    # YELLOW (Tension)
        return '#00FF00'                           # GREEN (Build)
        
    df['civic_color'] = df.apply(get_state, axis=1)
    
    # --- [4] VISUALIZATION ---
    plot_ratioque(df, col_close, filepath)
    
    # Export
    out_csv = filepath.replace('.csv', '_ratioque_data.csv')
    df.to_csv(out_csv, index=False)
    print(f"Structural data saved to: {out_csv}")

def plot_ratioque(df, price_col, filename):
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 2, 0.4]})
    
    # PANEL 1: THE SPECIES
    ax1.plot(df['date'], df[price_col], color='cyan', linewidth=1.5)
    ax1.set_title(f"1. THE SPECIES ({price_col})", fontsize=10, color='white', loc='left')
    ax1.grid(True, color='#333333', alpha=0.5)
    
    # PANEL 2: THE RATIOQUE (Thermal Edition)
    # The Core Line
    ax2.plot(df['date'], df['shear_core'], color='white', linewidth=0.8, alpha=0.8, label='Core Shear')
    
    # The Thermal Halo (Wobble)
    # We construct a fill between the min and max shear of the day
    shear_min = df[['shear_high', 'shear_low', 'shear_core']].min(axis=1)
    shear_max = df[['shear_high', 'shear_low', 'shear_core']].max(axis=1)
    
    ax2.fill_between(df['date'], shear_min, shear_max, color='white', alpha=0.15, label='Thermal Halo (Brownian)')
    
    # Core Fill Logic
    ax2.fill_between(df['date'], df['shear_core'], 0, where=(df['shear_core'] >= 0), color='#aaff00', alpha=0.3)
    ax2.fill_between(df['date'], df['shear_core'], 0, where=(df['shear_core'] < 0), color='#ff00aa', alpha=0.3)
    
    # Red Zones (Core Breach)
    stress_mask = df['shear_core'].abs() >= df['p90']
    ax2.fill_between(df['date'], df['shear_core'], 0, where=stress_mask, color='red', alpha=1.0)
    
    # Gauge Ghost
    ax2_twin = ax2.twinx()
    normalized_grid = (df['grid_strength'] - KernelConfig.N_BASE) / KernelConfig.N_BASE
    ax2_twin.fill_between(df['date'], normalized_grid, 0, color='white', alpha=0.05)
    ax2_twin.set_ylim(0, 0.5)
    ax2_twin.axis('off')
    
    ax2.axhline(0, color='yellow', linestyle='--', linewidth=0.8)
    ax2.set_title("2. THE RATIOQUE (Thermal Shear & Brownian Halo)", fontsize=10, color='white', loc='left')
    
    # PANEL 3: THE STATE
    colors = df['civic_color'].tolist()
    dates_num = mdates.date2num(df['date'])
    
    for i, (d, c) in enumerate(zip(dates_num, colors)):
        rect = Rectangle((d, 0), 1, 1, color=c, ec=None, alpha=1.0)
        ax3.add_patch(rect)
            
    ax3.set_xlim(dates_num[0], dates_num[-1])
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.text(0.01, 0.5, "INTEGRITY STATE:", transform=ax3.transAxes, color='black', fontsize=9, fontweight='bold', va='center')
    ax3.set_title("3. CIVIC DASHBOARD STATE (Sensitive to Tremors)", fontsize=10, color='white', loc='left')
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    save_name = filename.replace('.csv', '_ratioque_v2_5.png')
    plt.savefig(save_name, dpi=150)
    print(f"Visual analysis saved to: {save_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_ratioque(sys.argv[1])
    else:
        print("Usage: python ratioque.py <data.csv>")
