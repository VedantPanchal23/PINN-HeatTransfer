"""
Interactive thermal analysis dashboard.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from typing import Dict, Any, Optional, Callable


class ThermalDashboard:
    """
    Interactive dashboard for thermal analysis visualization.
    
    Features:
    - Real-time temperature field display
    - Time slider for transient simulations
    - Cross-section views
    - Interactive parameter adjustment
    """
    
    def __init__(self, figsize: tuple = (14, 10)):
        self.figsize = figsize
        self.fig = None
        self.axes = {}
        self.artists = {}
        self.data = {}
        
        self.callbacks = {}
    
    def create_dashboard(
        self,
        temperature_history: np.ndarray,
        times: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        material_info: Dict = None,
        thermal_limits: Any = None,
    ):
        """
        Create interactive dashboard.
        
        Args:
            temperature_history: Shape (n_times, ny, nx) temperature data
            times: Time values
            x_coords: X coordinates
            y_coords: Y coordinates
            material_info: Optional material information
            thermal_limits: Optional thermal limit analysis
        """
        self.data = {
            'T': temperature_history,
            'times': times,
            'x': x_coords,
            'y': y_coords,
            'material': material_info,
            'limits': thermal_limits,
        }
        
        self.fig = plt.figure(figsize=self.figsize)
        
        # Create layout
        gs = self.fig.add_gridspec(
            3, 3, 
            height_ratios=[1, 0.8, 0.15],
            width_ratios=[1, 1, 0.8],
            hspace=0.3, wspace=0.3
        )
        
        # Main heatmap
        self.axes['main'] = self.fig.add_subplot(gs[0, 0:2])
        self._setup_heatmap()
        
        # Info panel
        self.axes['info'] = self.fig.add_subplot(gs[0, 2])
        self._setup_info_panel()
        
        # Time history
        self.axes['history'] = self.fig.add_subplot(gs[1, 0])
        self._setup_history()
        
        # Temperature profile
        self.axes['profile'] = self.fig.add_subplot(gs[1, 1])
        self._setup_profile()
        
        # Statistics
        self.axes['stats'] = self.fig.add_subplot(gs[1, 2])
        self._setup_stats()
        
        # Time slider
        ax_slider = self.fig.add_subplot(gs[2, :])
        self._setup_slider(ax_slider)
        
        self.fig.suptitle("Thermal Analysis Dashboard", fontsize=14, fontweight='bold')
    
    def _setup_heatmap(self):
        """Setup main temperature heatmap."""
        ax = self.axes['main']
        T = self.data['T']
        x, y = self.data['x'], self.data['y']
        
        X, Y = np.meshgrid(x, y)
        
        vmin, vmax = np.nanmin(T), np.nanmax(T)
        
        self.artists['heatmap'] = ax.pcolormesh(
            X, Y, T[0],
            cmap='hot',
            vmin=vmin, vmax=vmax,
            shading='auto'
        )
        
        self.artists['colorbar'] = plt.colorbar(
            self.artists['heatmap'], ax=ax, label='Temperature (°C)'
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Temperature Field')
        ax.set_aspect('equal')
        
        # Time annotation
        self.artists['time_text'] = ax.text(
            0.02, 0.98, f't = {self.data["times"][0]:.4f}s',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=11
        )
        
        # Max temp marker
        T0 = T[0]
        max_idx = np.unravel_index(np.nanargmax(T0), T0.shape)
        self.artists['max_marker'], = ax.plot(
            x[max_idx[1]], y[max_idx[0]], 'b*', markersize=15, label='Max T'
        )
    
    def _setup_info_panel(self):
        """Setup information panel."""
        ax = self.axes['info']
        ax.axis('off')
        
        info_text = []
        
        if self.data['material']:
            mat = self.data['material']
            info_text.extend([
                "Material Properties:",
                f"  k = {mat.get('thermal_conductivity', 'N/A'):.1f} W/(m·K)",
                f"  ρ = {mat.get('density', 'N/A'):.0f} kg/m³",
                f"  cp = {mat.get('specific_heat', 'N/A'):.0f} J/(kg·K)",
                "",
            ])
        
        T = self.data['T']
        info_text.extend([
            "Simulation Info:",
            f"  Grid: {T.shape[2]} × {T.shape[1]}",
            f"  Time steps: {T.shape[0]}",
            f"  t_max = {self.data['times'][-1]:.3f}s",
        ])
        
        if self.data['limits']:
            limits = self.data['limits']
            info_text.append("")
            info_text.append("Thermal Limits:")
            if hasattr(limits, 'is_safe'):
                status = '✅ SAFE' if limits.is_safe else '❌ UNSAFE'
                info_text.append(f"  Status: {status}")
            if hasattr(limits, 'geometry_lifetime') and limits.geometry_lifetime is not None:
                if limits.geometry_lifetime == float('inf'):
                    info_text.append("  Lifetime: Unlimited")
                elif limits.geometry_lifetime < 60:
                    info_text.append(f"  Lifetime: {limits.geometry_lifetime:.1f}s")
                elif limits.geometry_lifetime < 3600:
                    info_text.append(f"  Lifetime: {limits.geometry_lifetime/60:.1f} min")
                else:
                    info_text.append(f"  Lifetime: {limits.geometry_lifetime/3600:.1f} hours")
            if hasattr(limits, 'max_temperature_capacity') and limits.max_temperature_capacity is not None:
                info_text.append(f"  Max Capacity: {limits.max_temperature_capacity:.0f}°C")
            if hasattr(limits, 'thermal_headroom') and limits.thermal_headroom is not None:
                info_text.append(f"  Headroom: {limits.thermal_headroom:.1f}%")
        
        self.artists['info_text'] = ax.text(
            0.05, 0.95, '\n'.join(info_text),
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3)
        )
    
    def _setup_history(self):
        """Setup temperature history plot."""
        ax = self.axes['history']
        T = self.data['T']
        times = self.data['times']
        
        max_temps = [np.nanmax(T[i]) for i in range(len(times))]
        mean_temps = [np.nanmean(T[i]) for i in range(len(times))]
        min_temps = [np.nanmin(T[i]) for i in range(len(times))]
        
        ax.plot(times, max_temps, 'r-', label='Max', linewidth=2)
        ax.plot(times, mean_temps, 'g--', label='Mean', linewidth=2)
        ax.plot(times, min_temps, 'b:', label='Min', linewidth=2)
        
        # Current time marker
        self.artists['time_marker'], = ax.plot(
            [times[0]], [max_temps[0]], 'ko', markersize=10
        )
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature History')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        self.data['max_temps'] = max_temps
        self.data['mean_temps'] = mean_temps
    
    def _setup_profile(self):
        """Setup temperature profile plot."""
        ax = self.axes['profile']
        T = self.data['T']
        x = self.data['x']
        y = self.data['y']
        
        # Horizontal profile through center
        mid_y = len(y) // 2
        self.artists['profile_line'], = ax.plot(
            x, T[0, mid_y, :], 'b-', linewidth=2
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Profile at Y = {y[mid_y]:.3f}m')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(np.nanmin(T) - 5, np.nanmax(T) + 5)
        ax.grid(True, alpha=0.3)
    
    def _setup_stats(self):
        """Setup statistics display."""
        ax = self.axes['stats']
        ax.axis('off')
        
        T0 = self.data['T'][0]
        
        stats_text = [
            "Current Statistics:",
            f"  Max T: {np.nanmax(T0):.1f}°C",
            f"  Min T: {np.nanmin(T0):.1f}°C",
            f"  Mean T: {np.nanmean(T0):.1f}°C",
            f"  Std T: {np.nanstd(T0):.1f}°C",
        ]
        
        self.artists['stats_text'] = ax.text(
            0.05, 0.95, '\n'.join(stats_text),
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=11,
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
        )
    
    def _setup_slider(self, ax):
        """Setup time slider."""
        times = self.data['times']
        
        self.artists['slider'] = Slider(
            ax, 'Time',
            times[0], times[-1],
            valinit=times[0],
            valstep=(times[-1] - times[0]) / (len(times) - 1)
        )
        
        self.artists['slider'].on_changed(self._update_time)
    
    def _update_time(self, val):
        """Update display for new time value."""
        times = self.data['times']
        T = self.data['T']
        x, y = self.data['x'], self.data['y']
        
        # Find nearest time index
        idx = np.argmin(np.abs(times - val))
        
        # Update heatmap
        self.artists['heatmap'].set_array(T[idx].ravel())
        
        # Update time text
        self.artists['time_text'].set_text(f't = {times[idx]:.4f}s')
        
        # Update max temp marker
        max_idx = np.unravel_index(np.nanargmax(T[idx]), T[idx].shape)
        self.artists['max_marker'].set_data([x[max_idx[1]]], [y[max_idx[0]]])
        
        # Update time marker in history
        self.artists['time_marker'].set_data(
            [times[idx]], [self.data['max_temps'][idx]]
        )
        
        # Update profile
        mid_y = len(y) // 2
        self.artists['profile_line'].set_ydata(T[idx, mid_y, :])
        
        # Update stats
        Ti = T[idx]
        stats_text = [
            "Current Statistics:",
            f"  Max T: {np.nanmax(Ti):.1f}°C",
            f"  Min T: {np.nanmin(Ti):.1f}°C",
            f"  Mean T: {np.nanmean(Ti):.1f}°C",
            f"  Std T: {np.nanstd(Ti):.1f}°C",
        ]
        self.artists['stats_text'].set_text('\n'.join(stats_text))
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the dashboard."""
        plt.show()
    
    def save(self, path: str, dpi: int = 150):
        """Save current view to file."""
        self.fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"Dashboard saved to: {path}")
