"""
Temperature field animation generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class AnimationConfig:
    """Configuration for temperature animation."""
    fps: int = 10
    dpi: int = 100
    figsize: Tuple[int, int] = (10, 8)
    colormap: str = "hot"
    
    show_colorbar: bool = True
    show_time: bool = True
    show_max_temp: bool = True
    
    vmin: Optional[float] = None
    vmax: Optional[float] = None


class TemperatureAnimator:
    """
    Generate animations of temperature evolution.
    """
    
    def __init__(self, config: AnimationConfig = None):
        self.config = config or AnimationConfig()
    
    def create_animation(
        self,
        temperature_history: np.ndarray,
        times: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        save_path: str = None,
        domain_mask: np.ndarray = None,
    ) -> animation.FuncAnimation:
        """
        Create animation from temperature history.
        
        Args:
            temperature_history: Shape (n_times, ny, nx) temperature data
            times: Time values for each frame
            x_coords: X coordinates (1D array)
            y_coords: Y coordinates (1D array)
            save_path: Path to save animation (mp4/gif)
            domain_mask: Optional mask for geometry (0 = outside, 1 = inside)
            
        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Determine color limits
        vmin = self.config.vmin or np.nanmin(temperature_history)
        vmax = self.config.vmax or np.nanmax(temperature_history)
        
        # Initial plot
        X, Y = np.meshgrid(x_coords, y_coords)
        
        T0 = temperature_history[0].copy()
        if domain_mask is not None:
            T0 = np.ma.masked_where(domain_mask == 0, T0)
        
        im = ax.pcolormesh(
            X, Y, T0,
            cmap=self.config.colormap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        
        if self.config.show_colorbar:
            cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        
        # Text annotations
        time_text = ax.text(
            0.02, 0.98, '', transform=ax.transAxes,
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        max_temp_text = ax.text(
            0.98, 0.98, '', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        def init():
            return [im, time_text, max_temp_text]
        
        def animate(frame):
            T = temperature_history[frame].copy()
            if domain_mask is not None:
                T = np.ma.masked_where(domain_mask == 0, T)
            
            im.set_array(T.ravel())
            
            if self.config.show_time:
                time_text.set_text(f't = {times[frame]:.3f} s')
            
            if self.config.show_max_temp:
                max_T = np.nanmax(T)
                max_temp_text.set_text(f'Max: {max_T:.1f}°C')
            
            return [im, time_text, max_temp_text]
        
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(times),
            interval=1000 // self.config.fps,
            blit=True
        )
        
        if save_path:
            self.save_animation(anim, save_path)
        
        return anim
    
    def save_animation(
        self,
        anim: animation.FuncAnimation,
        path: str
    ):
        """Save animation to file."""
        if path.endswith('.gif'):
            writer = animation.PillowWriter(fps=self.config.fps)
        else:
            writer = animation.FFMpegWriter(
                fps=self.config.fps,
                metadata={'title': 'Temperature Evolution'},
                bitrate=2000
            )
        
        anim.save(path, writer=writer, dpi=self.config.dpi)
        print(f"Animation saved to: {path}")
    
    def create_multi_view_animation(
        self,
        temperature_history: np.ndarray,
        times: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        probe_locations: List[Tuple[float, float]] = None,
        save_path: str = None,
    ) -> animation.FuncAnimation:
        """
        Create animation with multiple synchronized views.
        
        Views:
        - Main heatmap
        - Temperature profiles (cross-sections)
        - Time history at probe points
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax_main = fig.add_subplot(gs[0, 0])
        ax_profile_x = fig.add_subplot(gs[0, 1])
        ax_profile_y = fig.add_subplot(gs[1, 0])
        ax_history = fig.add_subplot(gs[1, 1])
        
        # Main heatmap
        X, Y = np.meshgrid(x_coords, y_coords)
        vmin = np.nanmin(temperature_history)
        vmax = np.nanmax(temperature_history)
        
        im = ax_main.pcolormesh(
            X, Y, temperature_history[0],
            cmap='hot', vmin=vmin, vmax=vmax, shading='auto'
        )
        plt.colorbar(im, ax=ax_main, label='T (°C)')
        ax_main.set_title('Temperature Field')
        ax_main.set_xlabel('X (m)')
        ax_main.set_ylabel('Y (m)')
        
        # X profile (horizontal slice through center)
        mid_y = len(y_coords) // 2
        line_x, = ax_profile_x.plot(x_coords, temperature_history[0, mid_y, :])
        ax_profile_x.set_xlim(x_coords.min(), x_coords.max())
        ax_profile_x.set_ylim(vmin - 5, vmax + 5)
        ax_profile_x.set_title('X Profile (Y=center)')
        ax_profile_x.set_xlabel('X (m)')
        ax_profile_x.set_ylabel('T (°C)')
        ax_profile_x.grid(True, alpha=0.3)
        
        # Y profile (vertical slice through center)
        mid_x = len(x_coords) // 2
        line_y, = ax_profile_y.plot(y_coords, temperature_history[0, :, mid_x])
        ax_profile_y.set_xlim(y_coords.min(), y_coords.max())
        ax_profile_y.set_ylim(vmin - 5, vmax + 5)
        ax_profile_y.set_title('Y Profile (X=center)')
        ax_profile_y.set_xlabel('Y (m)')
        ax_profile_y.set_ylabel('T (°C)')
        ax_profile_y.grid(True, alpha=0.3)
        
        # Time history (max and mean temperature)
        max_temps = [np.nanmax(T) for T in temperature_history]
        mean_temps = [np.nanmean(T) for T in temperature_history]
        
        line_max, = ax_history.plot([], [], 'r-', label='Max T')
        line_mean, = ax_history.plot([], [], 'b--', label='Mean T')
        ax_history.set_xlim(times.min(), times.max())
        ax_history.set_ylim(vmin - 5, vmax + 5)
        ax_history.set_title('Temperature History')
        ax_history.set_xlabel('Time (s)')
        ax_history.set_ylabel('T (°C)')
        ax_history.legend()
        ax_history.grid(True, alpha=0.3)
        
        def animate(frame):
            # Update heatmap
            im.set_array(temperature_history[frame].ravel())
            
            # Update profiles
            line_x.set_ydata(temperature_history[frame, mid_y, :])
            line_y.set_ydata(temperature_history[frame, :, mid_x])
            
            # Update history
            line_max.set_data(times[:frame+1], max_temps[:frame+1])
            line_mean.set_data(times[:frame+1], mean_temps[:frame+1])
            
            return [im, line_x, line_y, line_max, line_mean]
        
        anim = animation.FuncAnimation(
            fig, animate,
            frames=len(times),
            interval=1000 // self.config.fps,
            blit=True
        )
        
        if save_path:
            self.save_animation(anim, save_path)
        
        return anim
