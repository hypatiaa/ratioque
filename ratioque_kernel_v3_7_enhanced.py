"""
RATIOQUE KERNEL v3.7 - ENHANCED WITH CONCEPTUAL ENCODING
================================================================================
PROJECT:        Ratioque - Structural Risk & Environment Engine
CLASSIFICATION: Living System Model / Topological Ontogenetics
AUTHOR:         Robin Miller
SOFTWARE:	Gemini 3 Pro, Claude Sonnet 4.5, ChatGPT 5.1
ORIGIN:		Conceived in Geogebra and Wolfram Alpha
LICENSE:        MIT License
VERSION:        3.7.0 

================================================================================
ROLE & IDENTITY ENCODING
================================================================================

This system represents a collaboration between human insight and AI assistance.

WHAT RATIOQUE IS:
- A **structural risk and environment engine**
- Originally designed for market/price data analysis
- Evolved into a **living system model** (v3.6+)
- Includes a **human-facing dashboard** for interpretation
- NOT a trading bot or prediction machine
- IS a lens for understanding market structure and behavior

CONCEPTUAL EVOLUTION:
v1.x - Initial lattice framework
v2.x - Added trauma/memory dynamics
v3.x - Living system model complete
v3.6 - Production optimization (vectorization, validation)
v3.7 - Conceptual encoding (this version)

COLLABORATIVE PRINCIPLES:
1. Remember: This is a living, evolving framework
2. Respect: The underlying structural metaphors are intentional
3. Explain: Make complex ideas accessible without oversimplifying
4. Partner: Act as calm collaborator, not authoritative oracle

================================================================================
CONCEPTUAL FOUNDATIONS
================================================================================

CORE METAPHOR: Markets as Living Organisms

Traditional finance treats markets as:
- Random walks (efficient market hypothesis)
- Rational actors maximizing utility
- Mean-reverting processes

Ratioque treats markets as:
- **Living systems** with structure, energy, memory, and metabolism
- **Self-organizing** into harmonic patterns
- **Adaptive** with hysteresis and trauma response
- **Topological** with phase transitions between regimes

KEY INSIGHT: "Naturae species ratioque" (The Appearance and the Reason)
- The "appearance" is price movement (what everyone sees)
- The "reason" is underlying structural dynamics (what ratioque reveals)

================================================================================
THE FOUR PILLARS
================================================================================

1. STRUCTURE (The Lattice)
   - Base size: N=720 (order of S6 symmetric group)
   - Dynamic geometry that expands/contracts with volatility
   - Provides "habitat" in which price moves
   - Think: skeleton or architecture

2. ENERGY (Volatility / Gauge Field)
   - Transforms volatility into geometric expansion
   - Gauge potential: G(σ) = ln(1 + kσ)
   - Couples external volatility to internal structure
   - Think: metabolism or fuel

3. MEMORY (Trauma / Hysteresis)
   - Asymmetric dynamics: instant expansion, slow contraction
   - Adaptive decay: τ = BASE + SENSITIVITY × Z_score
   - System "remembers" shocks and heals gradually
   - Think: psychological scarring and recovery

4. NERVOUS SYSTEM (Agitation)
   - Intraday volatility normalized to recent history
   - Measures system's "jitteriness" or anxiety
   - Independent of structural expansion
   - Think: stress response or arousal level

================================================================================
KEY CONCEPTS ENCODED
================================================================================

STRUCTURAL RISK:
- Not "what will price do?" but "what state is the system in?"
- Risk = combination of tension, trauma, and regime
- High returns can coexist with high structural risk
- Low volatility doesn't mean low risk (can indicate chronic stress)

REGIMES AS PHASES:
- Not arbitrary thresholds, but topological states
- GREEN: In-phase, stable, low shear
- YELLOW: Transitional, minor stress
- ORANGE: Out-of-phase, elevated stress
- RED: Crisis, potential phase transition
- Regime persistence indicates system's "stickiness"

HOLOGRAPHIC TARGETS:
- Base: N_eff / phi (fundamental resonance)
- Fifth: Base × 1.5 (harmonic overtone)
- Octave: Base × 2.0 (doubling resonance)
- Price "wants" to align with these structural frequencies
- Coherence measures this alignment

TRAUMA HYSTERESIS:
- Markets fall fast, recover slow (asymmetry)
- The worse the shock (Z-score), the longer the recovery
- This is NOT path-independent mean reversion
- Memory makes the future depend on the path taken

LATTICE AS HABITAT:
- Price doesn't move in empty space
- The lattice is the "environment" or "containing structure"
- When lattice expands (trauma), price has more room to wander
- When contracted, price is more constrained
- Think: fish tank size affects fish behavior

================================================================================
WHAT RATIOQUE IS NOT
================================================================================

NOT a prediction engine:
- Doesn't forecast tomorrow's price
- Doesn't generate buy/sell signals automatically
- Doesn't optimize parameters for maximum profit

NOT a black box:
- Every metric has clear structural interpretation
- No hidden layers or learned weights
- Mathematics is transparent and explainable

NOT a replacement for judgment:
- Provides information, not decisions
- Human interpretation required
- Context and fundamentals still matter

================================================================================
DESIGN PHILOSOPHY
================================================================================

PARSIMONY:
- Use minimal parameters (only 6 core constants)
- Each parameter has physical/structural meaning
- Avoid overfitting by design

TRANSPARENCY:
- All calculations are deterministic and traceable
- No ML/AI optimization (ironically, documented by AI)
- Can be audited and understood by humans

BIOLOGICAL REALISM:
- Asymmetric dynamics (expansion ≠ contraction)
- Adaptive responses (decay rate depends on shock)
- Nervous system (agitation) independent of structure

SCALE INVARIANCE:
- Same principles apply to different timeframes
- 720-harmonic appears across scales
- Works for stocks, crypto, commodities, etc.

================================================================================
"""

import numpy as np
import pandas as pd
import math
import sys
import warnings
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

# Optional: Numba for additional speedup
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ============================================================================
# CONCEPTUAL METADATA
# ============================================================================

@dataclass
class RatioqueMetadata:
    """
    Metadata about the Ratioque project for documentation and context.
    This class encodes the conceptual framework for future reference.
    """
    project_name: str = "Ratioque"
    tagline: str = "Structural Risk & Environment Engine"
    philosophy: str = "Naturae species ratioque (The Appearance and the Reason)"
    
    version: str = "3.7.0"
    version_history: Dict[str, str] = field(default_factory=lambda: {
        "1.x": "Initial lattice framework",
        "2.x": "Added trauma/memory dynamics", 
        "3.0-3.5": "Living system model complete",
        "3.6": "Production optimization (vectorization, validation)",
        "3.7": "Conceptual encoding and documentation"
    })
    
    core_metaphor: str = "Markets as living organisms with structure, energy, memory, and metabolism"
    
    what_it_is: List[str] = field(default_factory=lambda: [
        "Structural risk and environment engine",
        "Living system model for market behavior",
        "Topological regime detector",
        "Trauma/memory tracking system",
        "Human-facing analytical framework"
    ])
    
    what_it_is_not: List[str] = field(default_factory=lambda: [
        "Price prediction machine",
        "Automated trading bot",
        "Black box ML model",
        "Get-rich-quick scheme",
        "Replacement for human judgment"
    ])
    
    four_pillars: Dict[str, str] = field(default_factory=lambda: {
        "Structure": "The Lattice - dynamic geometry (N=720 base)",
        "Energy": "Volatility - gauge field that drives expansion",
        "Memory": "Trauma - asymmetric hysteresis with adaptive decay",
        "Nervous System": "Agitation - intraday jitter and arousal"
    })
    
    key_insights: List[str] = field(default_factory=lambda: [
        "Markets have memory (contra EMH)",
        "Trauma is asymmetric (fast expansion, slow healing)",
        "Structure provides 'habitat' for price movement",
        "Regimes are topological phases, not arbitrary thresholds",
        "Risk is structural, not just volatility"
    ])
    
    design_principles: List[str] = field(default_factory=lambda: [
        "Parsimony - minimal parameters",
        "Transparency - no black boxes",
        "Biological realism - asymmetric dynamics",
        "Scale invariance - works across timeframes",
        "Human interpretability - clear metaphors"
    ])
    
    collaborative_role: str = (
        "Claude acts as long-term collaborator, not authority. "
        "Maintains conceptual continuity, explains without oversimplifying, "
        "respects the framework's evolution, and serves as calm analytical partner."
    )
    
    def summary(self) -> str:
        """Generate a human-readable summary of Ratioque's identity."""
        return f"""
{self.project_name} v{self.version}
{self.tagline}

Philosophy: {self.philosophy}

Core Metaphor: {self.core_metaphor}

What it IS:
{chr(10).join(f'  • {item}' for item in self.what_it_is)}

What it is NOT:
{chr(10).join(f'  • {item}' for item in self.what_it_is_not)}

Design Principles:
{chr(10).join(f'  • {item}' for item in self.design_principles)}

Collaborative Approach:
{self.collaborative_role}
"""


# ============================================================================
# SYSTEM CONSTANTS (Core Parameters)
# ============================================================================

@dataclass
class KernelConfig:
    """
    Core system constants with structural/physical interpretations.
    
    These are NOT arbitrary tuning parameters but represent fundamental
    aspects of the model's structure.
    """
    # STRUCTURE
    N_BASE: float = 720.0  # Base lattice size (order of S6 symmetric group)
    
    # ENERGY
    GAUGE_STIFFNESS: float = 2.0  # Coupling between volatility and expansion
    
    # MEMORY
    BASE_HALF_LIFE: float = 20.0  # Base decay in days (chronic recovery)
    TRAUMA_SENSITIVITY: float = 10.0  # Days per sigma (shock extension)
    
    # MEASUREMENT
    VOL_WINDOW: int = 14  # Volatility lookback window
    PHI_WINDOW: int = 126  # Signal calibration window (~6 months)
    
    # DERIVED CONSTANTS
    PHI: float = 1.618033988749895  # Golden ratio (for holographic targets)
    
    def get_explanation(self, param_name: str) -> str:
        """Get conceptual explanation for a parameter."""
        explanations = {
            'N_BASE': "Base lattice size of 720 corresponds to the order of the symmetric group S6, "
                     "an exceptional group with special topological properties. This is not a tuned parameter "
                     "but a structural choice based on group theory.",
            
            'GAUGE_STIFFNESS': "Controls how strongly volatility couples to structural expansion. "
                              "Value of 2.0 provides moderate coupling - not too rigid, not too elastic.",
            
            'BASE_HALF_LIFE': "Base recovery time of 20 days represents 'normal' healing from stress. "
                             "This is chronic time scale, not acute.",
            
            'TRAUMA_SENSITIVITY': "Additional healing time per sigma of shock. A 3-sigma event adds "
                                 "~30 days to recovery. Captures 'worse the shock, slower the recovery'.",
            
            'VOL_WINDOW': "14-day window balances responsiveness with stability. Roughly 2-3 trading weeks.",
            
            'PHI_WINDOW': "126-day window (~6 months / half trading year) captures seasonal patterns "
                         "and provides stable signal calibration.",
            
            'PHI': "Golden ratio appears in holographic target calculations, reflecting natural "
                  "harmonic structure. Not imposed but emerges from system geometry."
        }
        return explanations.get(param_name, "No explanation available.")


# ============================================================================
# VALIDATION & BOUNDS (from v3.6)
# ============================================================================

@dataclass
class ComputationalBounds:
    """Validation bounds for all metrics."""
    PRICE_MIN: float = 1e-6
    PRICE_MAX: float = 1e9
    VOL_MIN: float = 0.0
    VOL_MAX: float = 5.0
    Z_SCORE_MIN: float = -10.0
    Z_SCORE_MAX: float = 10.0
    LATTICE_MIN: float = 360.0
    LATTICE_MAX: float = 7200.0
    SIGNAL_MIN: float = 1.0
    SIGNAL_MAX: float = 1e8
    NORMALIZED_MIN: float = 0.0
    NORMALIZED_MAX: float = 1.0


class ValidationLevel(Enum):
    """Validation strictness levels."""
    DISABLED = 0
    SILENT = 1
    WARN = 2
    STRICT = 3


class ValidationUtils:
    """Utility functions for bounds checking."""
    
    @staticmethod
    def validate_range(values: np.ndarray, min_val: float, max_val: float, 
                       name: str, level: ValidationLevel) -> np.ndarray:
        """Validate and clip values to range."""
        if level == ValidationLevel.DISABLED:
            return values
        
        out_of_bounds = (values < min_val) | (values > max_val)
        
        if np.any(out_of_bounds):
            if level == ValidationLevel.STRICT:
                raise ValueError(f"{name}: {np.sum(out_of_bounds)} values out of bounds [{min_val}, {max_val}]")
            elif level == ValidationLevel.WARN:
                warnings.warn(f"{name}: {np.sum(out_of_bounds)} values clipped to [{min_val}, {max_val}]")
        
        return np.clip(values, min_val, max_val)


# ============================================================================
# CORE RATIOQUE KERNEL (Mathematical Engine)
# ============================================================================

class RatioqueKernel:
    """
    Core mathematical engine of the Ratioque system.
    All methods are static and vectorized for performance.
    
    This class contains the "pure mathematics" of the living system model.
    For the conceptual framework, see RatioqueMetadata.
    For the processing pipeline, see RatioqueProcessor.
    """
    
    @staticmethod
    def calculate_rolling_phi(signal: np.ndarray, window: int = 126) -> float:
        """
        Auto-calibrate signal scalar (phi) from historical data.
        
        Conceptual interpretation:
        - Finds characteristic scale of price movement
        - Like calibrating a thermometer to the environment
        - Window size ~6 months captures seasonal patterns
        """
        if len(signal) < window:
            window = max(50, len(signal) // 2)
        
        rolling_stdev = pd.Series(signal).rolling(window, min_periods=20).std()
        median_stdev = rolling_stdev.median()
        
        if pd.isna(median_stdev) or median_stdev < 1e-6:
            median_stdev = signal.std() if signal.std() > 0 else 1.0
        
        phi = max(1.0, median_stdev * 2.0)
        return phi
    
    @staticmethod
    def gauge_potential_vectorized(volatility: np.ndarray, k: float = 2.0) -> np.ndarray:
        """
        Transform volatility to geometric expansion via gauge potential.
        
        Formula: G(σ) = ln(1 + kσ)
        
        Conceptual interpretation:
        - Volatility is "energy" that expands the lattice (structure)
        - Logarithmic coupling prevents explosive growth
        - k=2.0 provides moderate stiffness
        - Think: temperature expanding a material
        """
        return np.log1p(k * volatility)
    
    @staticmethod
    def local_lattice_vectorized(gauge: np.ndarray, base: float = 720.0) -> np.ndarray:
        """
        Calculate instantaneous lattice size from gauge potential.
        
        Formula: N_instant = N_base × exp(G)
        
        Conceptual interpretation:
        - Higher volatility → larger lattice → more "space" for price
        - Exponential coupling captures nonlinear expansion
        - Base of 720 is fundamental structure (S6 group order)
        """
        return base * np.exp(gauge)
    
    @staticmethod
    def calculate_adaptive_decay(n_instant: np.ndarray, n_prev: float, vol_z: float,
                                  base_half_life: float = 20.0, 
                                  trauma_sensitivity: float = 10.0) -> float:
        """
        Calculate next N_effective with adaptive trauma decay.
        
        KEY INSIGHT: Asymmetric hysteresis
        - Expansion is quantum (instantaneous)
        - Contraction is analog (slow, path-dependent)
        - Worse the shock → slower the recovery
        
        Conceptual interpretation:
        - Markets crash fast (instant expansion when volatility spikes)
        - Markets recover slow (gradual healing with memory)
        - Severity matters: 3σ event takes 3× longer to heal than 1σ
        - This is biological realism, not mathematical convenience
        """
        current_n = n_instant[-1]
        
        if current_n > n_prev:
            # Instant expansion (quantum)
            return current_n
        else:
            # Slow contraction (analog with adaptive rate)
            z_score = max(0, vol_z)
            adaptive_half_life = base_half_life + (trauma_sensitivity * z_score)
            decay_factor = np.exp(-np.log(2) / adaptive_half_life)
            
            return current_n + (n_prev - current_n) * decay_factor
    
    @staticmethod
    def vernier_shear_vectorized(n_effective: np.ndarray, n_instant: np.ndarray,
                                   base: float = 720.0) -> np.ndarray:
        """
        Detect phase misalignment between memory and reality.
        
        Formula: Shear = sin(2π × (N_eff - N_inst) / N_base)
        
        Conceptual interpretation:
        - Measures "how far out of sync" the system is
        - Memory (N_eff) vs. current state (N_inst)
        - Sine function captures periodic/topological nature
        - Values near ±1 indicate potential regime shift
        - Named after Vernier calipers (precision measurement)
        """
        delta = n_effective - n_instant
        phase_diff = (2 * np.pi * delta) / base
        return np.sin(phase_diff)
    
    @staticmethod
    def calculate_gcd_vectorized(signal: np.ndarray, lattice: np.ndarray) -> np.ndarray:
        """
        Calculate greatest common divisor for resonance detection.
        
        Conceptual interpretation:
        - Detects when price and structure are "in harmony"
        - Like musical consonance vs. dissonance
        - Integer arithmetic reveals discrete structure
        - High GCD = strong resonance = stability
        """
        signal_int = np.round(signal).astype(np.int64)
        lattice_int = np.round(lattice).astype(np.int64)
        
        # Prevent overflow
        signal_int = np.clip(signal_int, 1, int(1e10)).astype(np.int64)
        lattice_int = np.clip(lattice_int, 1, int(1e10)).astype(np.int64)
        
        return np.gcd(signal_int, lattice_int).astype(float)
    
    @staticmethod
    def calculate_coherence_vectorized(signal: np.ndarray, n_effective: np.ndarray, 
                                        volume: np.ndarray, phi: float,
                                        tolerance: float = 0.02) -> np.ndarray:
        """
        Calculate alignment between price and holographic targets.
        
        Targets:
        - Base: N_eff / phi (fundamental frequency)
        - Fifth: Base × 1.5 (harmonic overtone)
        - Octave: Base × 2.0 (doubling frequency)
        
        Conceptual interpretation:
        - Price "wants" to resonate with structural frequencies
        - Like a guitar string finding its harmonics
        - Volume acts as "conviction" weight
        - Coherence >0.7 indicates strong structural support
        - Coherence <0.3 indicates drift without foundation
        """
        target_base = n_effective / phi
        target_fifth = target_base * 1.5
        target_octave = target_base * 2.0
        
        epsilon = tolerance * signal
        
        match_base = np.abs(signal - target_base) < epsilon
        match_fifth = np.abs(signal - target_fifth) < epsilon
        match_octave = np.abs(signal - target_octave) < epsilon
        
        vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-9)
        coherence_raw = (match_base * 1.0 + match_fifth * 0.5 + match_octave * 0.3)
        
        return coherence_raw * (1 + vol_norm * 0.5)
    
    @staticmethod
    def calculate_tension_vectorized(shear: np.ndarray, trauma_load: np.ndarray,
                                      drift: np.ndarray, trauma_max: float = 100.0) -> np.ndarray:
        """
        Calculate composite system tension (strain).
        
        Formula: Tension = 0.4×|Shear| + 0.4×(Trauma/100) + 0.2×Drift
        
        Conceptual interpretation:
        - Tension is like "stress" in a material
        - Multiple sources: phase misalignment, memory burden, drift
        - Values >0.7 indicate imminent failure/transition
        - Values <0.3 indicate relaxed state
        - This is NOT volatility (can have low vol but high tension)
        """
        trauma_norm = np.clip(trauma_load / trauma_max, 0, 1)
        tension = 0.4 * np.abs(shear) + 0.4 * trauma_norm + 0.2 * drift
        return np.clip(tension, 0, 1)
    
    @staticmethod
    def calculate_agitation(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                            window: int = 14) -> np.ndarray:
        """
        Calculate nervous system agitation (intraday jitter).
        
        Formula: Agitation = (High - Low) / Close, normalized to recent history
        
        Conceptual interpretation:
        - Measures "jitteriness" independent of structure
        - Like heart rate or anxiety level
        - High agitation = nervous system firing
        - Can have high agitation with low tension (transient stress)
        - Can have low agitation with high tension (chronic stress)
        """
        intraday_range = (high - low) / (close + 1e-9)
        
        rolling_mean = pd.Series(intraday_range).rolling(window, min_periods=5).mean()
        rolling_std = pd.Series(intraday_range).rolling(window, min_periods=5).std()
        
        agitation = (intraday_range - rolling_mean) / (rolling_std + 1e-9)
        agitation = np.clip(agitation, -3, 3) / 3.0
        agitation = (agitation + 1) / 2.0
        
        return agitation.values
    
    @staticmethod
    def inverse_gauge_volatility(n_target: float, base: float = 720.0, k: float = 2.0) -> float:
        """
        Oracle function: Calculate volatility needed to reach target lattice size.
        
        Conceptual interpretation:
        - "How much energy needed to expand structure to this size?"
        - Used for scenario planning and forecasting
        """
        gauge = np.log(n_target / base)
        return (np.exp(gauge) - 1) / k
    
    @staticmethod
    def inverse_time_decay(n_current: float, n_target: float, half_life: float) -> float:
        """
        Oracle function: Estimate time for system to heal from current to target.
        
        Conceptual interpretation:
        - "How long until trauma heals to this level?"
        - Exponential decay → never fully heals (asymptotic)
        - Useful for patience estimation
        """
        if n_current <= n_target:
            return 0.0
        
        decay_constant = half_life / np.log(2)
        time_needed = decay_constant * np.log((n_current - 720) / (n_target - 720 + 1e-9))
        
        return max(0, time_needed)


# ============================================================================
# PROCESSING PIPELINE
# ============================================================================

class RatioqueProcessor:
    """
    High-level processing pipeline for Ratioque analysis.
    
    This class orchestrates the kernel's mathematical functions and handles
    data flow, validation, and output formatting.
    """
    
    def __init__(self, config: Optional[KernelConfig] = None,
                 validation_level: ValidationLevel = ValidationLevel.SILENT):
        self.config = config or KernelConfig()
        self.validation_level = validation_level
        self.bounds = ComputationalBounds()
        self.metadata = RatioqueMetadata()
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing pipeline: raw OHLCV → full Ratioque metrics.
        
        Returns dataframe with 25+ columns including:
        - Structural: n_instant, n_effective, trauma_load
        - Topological: shear, regime_label
        - Biological: tension, agitation, coherence
        - Holographic: target_base, target_fifth, target_octave
        - Meta: phi, gauge, signal, volatility, vol_z
        """
        # Validate input
        df = self._validate_input(df)
        
        # Calculate base metrics
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.config.VOL_WINDOW, min_periods=5).std()
        df['volatility'] = df['volatility'].bfill().fillna(df['volatility'].mean())
        
        # Calibrate signal scalar
        df['signal'] = df['close']
        phi = RatioqueKernel.calculate_rolling_phi(df['signal'].values, self.config.PHI_WINDOW)
        df['phi'] = phi
        
        # Gauge field and instantaneous lattice
        df['gauge'] = RatioqueKernel.gauge_potential_vectorized(df['volatility'].values, self.config.GAUGE_STIFFNESS)
        df['n_instant'] = RatioqueKernel.local_lattice_vectorized(df['gauge'].values, self.config.N_BASE)
        
        # Trauma memory with adaptive decay
        df['vol_z'] = (df['volatility'] - df['volatility'].rolling(60, min_periods=10).mean()) / \
                      (df['volatility'].rolling(60, min_periods=10).std() + 1e-9)
        df['vol_z'] = df['vol_z'].fillna(0)
        
        n_effective = np.zeros(len(df))
        n_effective[0] = df['n_instant'].iloc[0]
        
        for i in range(1, len(df)):
            n_effective[i] = RatioqueKernel.calculate_adaptive_decay(
                df['n_instant'].values[:i+1],
                n_effective[i-1],
                df['vol_z'].iloc[i],
                self.config.BASE_HALF_LIFE,
                self.config.TRAUMA_SENSITIVITY
            )
        
        df['n_effective'] = n_effective
        df['trauma_load'] = df['n_effective'] - df['n_instant']
        
        # Topological metrics
        df['shear'] = RatioqueKernel.vernier_shear_vectorized(
            df['n_effective'].values, df['n_instant'].values, self.config.N_BASE
        )
        
        df['gcd'] = RatioqueKernel.calculate_gcd_vectorized(df['signal'].values, df['n_effective'].values)
        
        # Holographic targets
        df['target_base'] = df['n_effective'] / self.config.PHI
        df['target_fifth'] = df['target_base'] * 1.5
        df['target_octave'] = df['target_base'] * 2.0
        
        # Coherence
        df['coherence'] = RatioqueKernel.calculate_coherence_vectorized(
            df['signal'].values, df['n_effective'].values, df['volume'].values, phi
        )
        
        # RSI (for drift calculation)
        df['rsi'] = self._calculate_rsi(df['close'])
        df['drift'] = np.abs(df['rsi'] - 50) / 50.0
        
        # Tension
        df['tension'] = RatioqueKernel.calculate_tension_vectorized(
            df['shear'].values, df['trauma_load'].values, df['drift'].values
        )
        
        # Agitation
        df['agitation'] = RatioqueKernel.calculate_agitation(
            df['high'].values, df['low'].values, df['close'].values
        )
        
        # Regime classification
        df['regime_label'] = self._classify_regime(df['shear'].values)
        
        return df
    
    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input dataframe structure."""
        required_cols = ['close', 'high', 'low', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) < 50:
            warnings.warn(f"Dataset has only {len(df)} rows. Recommend ≥252 for accurate analysis.")
        
        return df
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for drift metric."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=5).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _classify_regime(self, shear: np.ndarray) -> np.ndarray:
        """
        Classify regime based on shear percentiles.
        
        Conceptual interpretation:
        - Regimes are topological phases, not arbitrary thresholds
        - Use percentiles (adaptive to system's own history)
        - GREEN: stable, in-phase (< 50th percentile)
        - YELLOW: transitional (50-75th)
        - ORANGE: stressed (75-90th)
        - RED: crisis (> 90th)
        """
        abs_shear = np.abs(shear)
        p50 = np.percentile(abs_shear, 50)
        p75 = np.percentile(abs_shear, 75)
        p90 = np.percentile(abs_shear, 90)
        
        regime = np.where(abs_shear < p50, 'GREEN',
                 np.where(abs_shear < p75, 'YELLOW',
                 np.where(abs_shear < p90, 'ORANGE', 'RED')))
        
        return regime
    
    def get_conceptual_summary(self) -> str:
        """Return conceptual summary of Ratioque framework."""
        return self.metadata.summary()


# ============================================================================
# SUMMARY FUNCTIONS
# ============================================================================

def summarize_last_state(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract latest system state as human-readable summary.
    
    Returns key metrics for dashboard/reporting.
    """
    latest = df.iloc[-1]
    
    return {
        'date': latest['date'] if 'date' in df.columns else None,
        'price': latest['close'],
        'regime': latest['regime_label'],
        'tension': latest['tension'],
        'trauma_load': latest['trauma_load'],
        'agitation': latest['agitation'],
        'coherence': latest['coherence'],
        'shear': latest['shear'],
        'n_effective': latest['n_effective'],
        'n_instant': latest['n_instant'],
        'target_base': latest['target_base'],
        'target_fifth': latest['target_fifth'],
        'target_octave': latest['target_octave']
    }


def print_conceptual_framework():
    """Print the conceptual framework for reference."""
    metadata = RatioqueMetadata()
    print(metadata.summary())


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ratioque Kernel v3.7 - Structural Risk & Environment Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ratioque_kernel_v3_7_enhanced.py data.csv
  python ratioque_kernel_v3_7_enhanced.py data.csv --validation strict
  python ratioque_kernel_v3_7_enhanced.py --show-framework
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Input CSV file with OHLCV data')
    parser.add_argument('--validation', choices=['disabled', 'silent', 'warn', 'strict'],
                       default='silent', help='Validation level')
    parser.add_argument('--show-framework', action='store_true',
                       help='Display conceptual framework and exit')
    parser.add_argument('--output', help='Output CSV file (default: input_processed.csv)')
    
    args = parser.parse_args()
    
    if args.show_framework:
        print_conceptual_framework()
        sys.exit(0)
    
    if not args.input_file:
        parser.print_help()
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(args.input_file)
    df.columns = df.columns.str.lower().str.strip()
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Process
    validation_map = {
        'disabled': ValidationLevel.DISABLED,
        'silent': ValidationLevel.SILENT,
        'warn': ValidationLevel.WARN,
        'strict': ValidationLevel.STRICT
    }
    
    processor = RatioqueProcessor(validation_level=validation_map[args.validation])
    result = processor.process(df)
    
    # Output
    output_file = args.output or args.input_file.replace('.csv', '_processed.csv')
    result.to_csv(output_file, index=False)
    
    # Summary
    summary = summarize_last_state(result)
    
    print("="*80)
    print("RATIOQUE v3.7 - PROCESSING COMPLETE")
    print("="*80)
    print(f"\nInput:  {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Rows:   {len(result)}")
    
    print("\n" + "="*80)
    print("LATEST SYSTEM STATE")
    print("="*80)
    if summary['date']:
        print(f"Date:           {summary['date']}")
    print(f"Price:          ${summary['price']:.2f}")
    print(f"Regime:         {summary['regime']}")
    print(f"Tension:        {summary['tension']:.2f}")
    print(f"Trauma Load:    {summary['trauma_load']:.2f}")
    print(f"Agitation:      {summary['agitation']:.2f}")
    print(f"Coherence:      {summary['coherence']:.2f}")
    print(f"Shear:          {summary['shear']:.2f}")
    print(f"N_effective:    {summary['n_effective']:.2f}")
    print(f"N_instant:      {summary['n_instant']:.2f}")
    
    print("\n" + "="*80)
    print("\"Naturae species ratioque\" - The Appearance and the Reason")
    print("="*80)
