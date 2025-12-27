"""
Analytical solutions for validation of simple heat transfer problems.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import special


class AnalyticalSolutions:
    """
    Analytical solutions for standard heat transfer problems.
    
    Used to validate PINN and FDM solvers on problems with known solutions.
    """
    
    @staticmethod
    def steady_state_1d(
        x: np.ndarray,
        T_left: float,
        T_right: float,
        L: float = 1.0,
    ) -> np.ndarray:
        """
        1D steady-state conduction between two fixed temperatures.
        
        T(x) = T_left + (T_right - T_left) * x / L
        """
        return T_left + (T_right - T_left) * x / L
    
    @staticmethod
    def transient_1d_semi_infinite(
        x: np.ndarray,
        t: float,
        T_initial: float,
        T_surface: float,
        alpha: float,
    ) -> np.ndarray:
        """
        1D transient conduction in semi-infinite solid.
        
        Sudden change in surface temperature.
        
        T(x,t) = T_initial + (T_surface - T_initial) * erfc(x / (2 * sqrt(α * t)))
        """
        if t <= 0:
            return np.full_like(x, T_initial)
        
        eta = x / (2 * np.sqrt(alpha * t))
        return T_initial + (T_surface - T_initial) * special.erfc(eta)
    
    @staticmethod
    def transient_1d_finite_slab(
        x: np.ndarray,
        t: float,
        T_initial: float,
        T_boundary: float,
        alpha: float,
        L: float,
        n_terms: int = 50,
    ) -> np.ndarray:
        """
        1D transient conduction in finite slab with symmetric BCs.
        
        Slab of thickness 2L, symmetric about x=0.
        Both surfaces at T_boundary.
        
        Uses Fourier series solution.
        """
        if t <= 0:
            return np.full_like(x, T_initial)
        
        T = np.full_like(x, T_boundary, dtype=np.float64)
        
        for n in range(n_terms):
            lambda_n = (2*n + 1) * np.pi / (2 * L)
            Fo = alpha * t * lambda_n**2
            
            term = ((-1)**n / (2*n + 1)) * np.cos(lambda_n * x) * np.exp(-Fo)
            T += (4 / np.pi) * (T_initial - T_boundary) * term
        
        return T
    
    @staticmethod
    def steady_state_2d_rectangular(
        X: np.ndarray,
        Y: np.ndarray,
        Lx: float,
        Ly: float,
        T_bottom: float = 0.0,
        T_top: float = 0.0,
        T_left: float = 0.0,
        T_right: float = 100.0,
        n_terms: int = 50,
    ) -> np.ndarray:
        """
        2D steady-state conduction in rectangular domain.
        
        Three sides at T=0, one side at T=T_right.
        Uses separation of variables solution.
        """
        T = np.zeros_like(X)
        
        for n in range(1, n_terms + 1):
            lambda_n = n * np.pi / Ly
            
            # Coefficient for series
            if n % 2 == 1:  # Odd terms only for uniform BC
                An = 4 * T_right / (n * np.pi)
            else:
                continue
            
            term = An * np.sin(lambda_n * Y) * np.sinh(lambda_n * X) / np.sinh(lambda_n * Lx)
            T += term
        
        return T
    
    @staticmethod
    def lumped_capacitance(
        t: np.ndarray,
        T_initial: float,
        T_ambient: float,
        h: float,
        A: float,
        rho: float,
        V: float,
        cp: float,
    ) -> np.ndarray:
        """
        Lumped capacitance model for transient cooling/heating.
        
        Valid when Bi = hL/k << 0.1
        
        T(t) = T_ambient + (T_initial - T_ambient) * exp(-t/tau)
        where tau = rho * V * cp / (h * A)
        """
        tau = rho * V * cp / (h * A)
        return T_ambient + (T_initial - T_ambient) * np.exp(-t / tau)
    
    @staticmethod
    def steady_state_with_generation(
        x: np.ndarray,
        q_dot: float,
        k: float,
        L: float,
        T_surface: float,
    ) -> np.ndarray:
        """
        1D steady-state conduction with uniform heat generation.
        
        Slab with both surfaces at T_surface.
        
        T(x) = T_surface + (q_dot / 2k) * (L² - x²)
        
        Maximum temperature at center.
        """
        return T_surface + (q_dot / (2 * k)) * (L**2 - x**2)
    
    @staticmethod
    def fin_temperature(
        x: np.ndarray,
        T_base: float,
        T_ambient: float,
        h: float,
        k: float,
        P: float,
        Ac: float,
        L: float,
        tip_condition: str = "adiabatic",
    ) -> np.ndarray:
        """
        Temperature distribution in a 1D fin.
        
        Args:
            x: Position along fin (0 = base)
            T_base: Base temperature
            T_ambient: Ambient temperature
            h: Heat transfer coefficient
            k: Thermal conductivity
            P: Fin perimeter
            Ac: Cross-sectional area
            L: Fin length
            tip_condition: "adiabatic", "convective", or "prescribed"
        """
        m = np.sqrt(h * P / (k * Ac))
        theta_b = T_base - T_ambient
        
        if tip_condition == "adiabatic":
            theta = theta_b * np.cosh(m * (L - x)) / np.cosh(m * L)
        elif tip_condition == "convective":
            mL = m * L
            numerator = np.cosh(m * (L - x)) + (h / (m * k)) * np.sinh(m * (L - x))
            denominator = np.cosh(mL) + (h / (m * k)) * np.sinh(mL)
            theta = theta_b * numerator / denominator
        else:
            # Infinite fin approximation
            theta = theta_b * np.exp(-m * x)
        
        return theta + T_ambient
    
    @staticmethod
    def gaussian_heat_pulse(
        X: np.ndarray,
        Y: np.ndarray,
        t: float,
        Q: float,
        alpha: float,
        rho: float,
        cp: float,
        x0: float = 0.5,
        y0: float = 0.5,
        T_initial: float = 0.0,
    ) -> np.ndarray:
        """
        2D transient temperature from instantaneous point heat source.
        
        Green's function solution for infinite medium.
        """
        if t <= 0:
            return np.full_like(X, T_initial)
        
        r2 = (X - x0)**2 + (Y - y0)**2
        
        # 2D point source solution
        T = T_initial + (Q / (4 * np.pi * rho * cp * alpha * t)) * np.exp(-r2 / (4 * alpha * t))
        
        return T
