import os
import matplotlib
# Headless-friendly: only grab an interactive backend if the caller asks for one.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import pandas as pd
import logging
import warnings
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

logger = logging.getLogger(__name__)

# Color palette for cycling through distinct colors for different files
_COLOR_PALETTE = plt.get_cmap("tab20").colors
_FILE_COLORS_CACHE = {}


# Function to get color per file
def get_color_for_file(file_name):
    if file_name not in _FILE_COLORS_CACHE:
        file_count = len(_FILE_COLORS_CACHE)
        _FILE_COLORS_CACHE[file_name] = _COLOR_PALETTE[file_count % len(_COLOR_PALETTE)]
    return _FILE_COLORS_CACHE[file_name]


def get_display_name(file_name, special_labels, is_primary=False, extract_display_name=None):
    """Retrieve display name based on the file name key, defaulting to base name if no label exists."""
    if special_labels is None:
        special_labels = {}

    if is_primary:
        return special_labels.get(file_name, extract_display_name(file_name))
    else:
        return special_labels.get(file_name, os.path.splitext(file_name)[0].split('_')[0])


class PlotGenerator:
    def __init__(self, plot_params, downsampling_rate, rolling_window_params, segment_type):
        """
        Initializes the PlotGenerator with plot parameters and downsampling rate.

        Parameters:
        plot_params (dict): Dictionary containing plot settings such as font sizes, title, and color settings.
        downsampling_rate (int): The rate at which data points are downsampled for plotting.
        rolling_window_params (dict): Parameters for rolling window filtering.
        """
        self.ax2 = None
        self.plot_params = plot_params
        self.downsampling_rate = downsampling_rate
        self.rolling_window_params = rolling_window_params
        self.segment_type = segment_type
        self.show_degradation_rate = False
        self.pol_curve_settings = None  # Will be set externally
        self.sigma_band_type = "1σ only"  # NEW: Default sigma band type
        self.axis_length_x = plot_params.get('axis_length_x', 1280)  # Default width in pixels
        self.axis_length_y = plot_params.get('axis_length_y', 720)  # Default height in pixels
        self.axis_range_x_min = plot_params.get('axis_range_x_min', None)
        self.axis_range_x_max = plot_params.get('axis_range_x_max', None)
        self.axis_range_y_min = plot_params.get('axis_range_y_min', None)
        self.axis_range_y_max = plot_params.get('axis_range_y_max', None)
        self.show_mean_voltage = plot_params.get('show_mean_voltage', True)
        self._data_processor = None
        self._column_cache = {}  # Cache for column existence checks
        self._slice_cache = {}  # Cache for DataFrame slices

    def set_pol_curve_settings(self, pol_curve_settings):
        """Set polarization curve settings for multiple curve plotting."""
        self.pol_curve_settings = pol_curve_settings

    def set_sigma_band_type(self, sigma_band_type):
        """Set the sigma band type for error band display."""
        self.sigma_band_type = sigma_band_type

    def get_data_processor(self):
        """Lazy load DataProcessor instance to avoid repeated instantiation."""
        if self._data_processor is None:
            try:
                from data_processor import DataProcessor
            except ImportError:
                from src.data_processing.data_processor import DataProcessor
            self._data_processor = DataProcessor()
        return self._data_processor

    def generate_color_gradient(self, base_color, num_curves):
        """
        Generate a gradient of colors from light to dark based on a base color.

        Parameters:
        base_color (str): Base color in hex format
        num_curves (int): Number of curves to generate colors for

        Returns:
        list: List of hex colors from light to dark
        """
        try:
            # Convert hex to RGB
            rgb = mcolors.hex2color(base_color)

            if num_curves == 1:
                return [base_color]

            # Create gradient from lighter to darker
            colors = []
            for i in range(num_curves):
                # Calculate darkness factor (0.3 to 1.0, where 1.0 is original color)
                darkness_factor = 0.3 + (0.7 * i / (num_curves - 1))

                # Apply darkness factor to RGB values
                dark_rgb = tuple(c * darkness_factor for c in rgb)

                # Convert back to hex
                dark_hex = mcolors.to_hex(dark_rgb)
                colors.append(dark_hex)

            return colors

        except Exception as e:
            logger.error(f"Error generating color gradient: {e}")
            # Fallback to original color
            return [base_color] * num_curves

    def create_colormap_legend(self, num_curves, base_colors):
        """
        Create a colormap legend showing the relationship between curve number and color intensity.

        Parameters:
        num_curves (int): Maximum number of curves across all files
        base_colors (dict): Dictionary mapping file names to their base colors
        """
        try:
            if num_curves <= 1:
                return  # No need for colormap legend

            # Create a small subplot for the colormap
            fig = plt.gcf()  # Get current figure

            # Add text annotation explaining the color scheme
            colormap_text = f"Color Intensity: Curve #1 (Light) → Curve #{num_curves} (Dark)"

            # Position the text at the top of the plot
            fig.text(0.5, 0.95, colormap_text, ha='center', va='top',
                     fontsize=self.plot_params.get('fontsize_legends', 8),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        except Exception as e:
            logger.error(f"Error creating colormap legend: {e}")

    def rolling_filter(self, data, column, window_size, threshold):
        """
        Filters data points that deviate significantly from the rolling median.
        """
        if column not in data.columns:
            logger.warning(f"Column '{column}' not found in data. Skipping rolling filter.")
            return data

        # Vectorized operation - avoid intermediate Series objects
        col_data = data[column].values
        rolling_median = pd.Series(col_data).rolling(window=window_size, center=True).median().values
        deviation = np.abs(col_data - rolling_median)
        max_deviation = threshold * np.abs(rolling_median)
        filtered_data = data[deviation <= max_deviation]

        return filtered_data

    def generate_plot(self, primary_data_list, additional_data_list, special_labels=None, color_selections=None):
        """
        Main function to generate the plot with support for multiple polarization curves.
        """
        try:
            if additional_data_list is None:
                additional_data_list = []

            # Step 1: Setup
            # Create a single figure with custom axis lengths
            fig_width = self.axis_length_x / 100  # Convert pixels to inches (assuming 100 DPI)
            fig_height = self.axis_length_y / 100  # Convert pixels to inches (assuming 100 DPI)
            self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
            self.ax2 = self.ax.twinx() if self.show_degradation_rate else None
            labeled_files = set()

            self.total_experiment_duration = get_total_experiment_duration(primary_data_list, additional_data_list)

            if self.segment_type == "polarization":
                # Handle polarization curve plotting
                max_curves = 0
                base_colors = {}

                # Only create colormap legend if multiple curves mode is enabled
                if self.pol_curve_settings and self.pol_curve_settings.get("multiple_curves", False):
                    all_datasets = []
                    if primary_data_list:
                        all_datasets.extend([(pd, True) for pd in primary_data_list])
                    if additional_data_list:
                        all_datasets.extend([(ad, False) for ad in additional_data_list])

                    for dataset, is_primary in all_datasets:
                        file_name = dataset.get('file_name')
                        file_color = color_selections.get(file_name, {}).get('hex_code', get_color_for_file(file_name))
                        base_colors[file_name] = file_color

                        pol_curves = self.get_selected_polarization_curves(dataset)
                        max_curves = max(max_curves, len(pol_curves))

                    if max_curves > 1:
                        self.create_colormap_legend(max_curves, base_colors)

                # Plot curves
                if primary_data_list:
                    for primary_data in primary_data_list:
                        self.plot_multiple_polarization_curves(primary_data, labeled_files, special_labels,
                                                              color_selections, is_primary=True)

                if additional_data_list:
                    for additional_data in additional_data_list:
                        self.plot_multiple_polarization_curves(additional_data, labeled_files, special_labels,
                                                              color_selections, is_primary=False)

                self.set_plot_limits(primary_data_list, additional_data_list, segment_type="polarization")
                self.finalize_plot(primary_data_list, additional_data_list, segment_type="polarization")


            elif self.segment_type == "constant":
                if primary_data_list:
                    for primary_data in primary_data_list:
                        self.plot_primary_data(primary_data, labeled_files, special_labels, color_selections)

                if additional_data_list:
                    for additional_data in additional_data_list:
                        self.plot_additional_data(additional_data, labeled_files, special_labels, color_selections)

                self.set_plot_limits(primary_data_list, additional_data_list, segment_type="constant")
                self.finalize_plot(primary_data_list, additional_data_list, segment_type="constant")


            elif self.segment_type == "both":
                if primary_data_list:
                    for primary_data in primary_data_list:
                        self.plot_primary_data(primary_data, labeled_files, special_labels, color_selections)

                if additional_data_list:
                    for additional_data in additional_data_list:
                        self.plot_additional_data(additional_data, labeled_files, special_labels, color_selections)

                self.set_plot_limits(primary_data_list, additional_data_list, segment_type="both")
                self.finalize_plot(primary_data_list, additional_data_list, segment_type="both")

            # Update the legend dynamically
            self.update_legend(color_selections)

        except Exception as e:
            logger.error(f"Error generating plot: {e}")
            raise Exception(f"Error generating plot: {e}")

    def get_selected_polarization_curves(self, dataset):
        """
        Get selected polarization curves based on settings.
        Now works with complete polarization curves (groups of segments).

        Parameters:
        dataset (dict): Dataset containing segments

        Returns:
        list: Selected complete polarization curves (each curve is a list of segments)
        """
        try:
            if not self.pol_curve_settings or not self.pol_curve_settings.get("multiple_curves", False):
                # Single curve mode - use existing logic to find last complete curve
                segments = self.filter_segments(dataset.get('segments', []))
                last_curve = self.find_last_complete_polarization_curve(segments)
                return [last_curve] if last_curve else []

            # Multiple curve mode - use new selection logic
            curve_selection = self.pol_curve_settings.get("curve_selection", "last")
            range_start = self.pol_curve_settings.get("range_start")
            range_end = self.pol_curve_settings.get("range_end")

            # Get all segments
            all_segments = dataset.get('segments', [])

            dp = self.get_data_processor()
            selected_complete_curves = dp.get_polarization_curves_by_selection(
                all_segments, curve_selection, range_start, range_end
            )

            print(f"Selected {len(selected_complete_curves)} complete polarization curves")
            return selected_complete_curves

        except Exception as e:
            logger.error(f"Error getting selected polarization curves: {e}")
            return []

    def find_last_complete_polarization_curve(self, segments):
        """
        Find the last complete polarization curve (group of segments).

        Parameters:
        segments (list): List of segments

        Returns:
        list: List of segments forming the last complete polarization curve
        """
        try:
            # Group segments by curve number
            complete_curves = {}

            for segment in segments:
                if (segment["type"] == "polarization" and
                        segment.get("pol_curve_number") is not None):

                    curve_num = segment["pol_curve_number"]
                    if curve_num not in complete_curves:
                        complete_curves[curve_num] = []
                    complete_curves[curve_num].append(segment)

            if not complete_curves:
                return None

            # Get the last (highest numbered) curve
            last_curve_num = max(complete_curves.keys())
            last_curve_segments = complete_curves[last_curve_num]

            # Sort segments by start time
            last_curve_segments.sort(key=lambda x: x["start_time"])

            return last_curve_segments

        except Exception as e:
            logger.error(f"Error finding last complete polarization curve: {e}")
            return None

    def plot_multiple_polarization_curves(self, data, labeled_files, special_labels=None,
                                          color_selections=None, is_primary=False):
        """
        Plot multiple complete polarization curves with gradient coloring.
        """
        try:
            file_name = data.get('file_name')
            short_display_name = get_display_name(file_name, special_labels, is_primary, self.extract_display_name)
            base_color = color_selections.get(file_name, {}).get('hex_code', get_color_for_file(file_name))

            # Check if multiple curves is enabled
            if not self.pol_curve_settings or not self.pol_curve_settings.get("multiple_curves", False):
                # Single curve mode - use original logic
                self.plot_single_polarization_curve(data, labeled_files, special_labels, color_selections, is_primary)
                return

            # Multiple curves mode - get complete curves
            selected_complete_curves = self.get_selected_polarization_curves(data)

            if not selected_complete_curves:
                logger.warning(f"No complete polarization curves found for {file_name}")
                return

            num_curves = len(selected_complete_curves)
            print(f"Plotting {num_curves} complete curves for {file_name}")

            # Generate color gradient
            curve_colors = self.generate_color_gradient(base_color, num_curves)

            # Plot each complete curve
            for i, complete_curve_segments in enumerate(selected_complete_curves):
                if not complete_curve_segments:
                    continue

                # Process this complete curve (made of multiple segments)
                current_density, avg_voltage = self.process_complete_polarization_curve(complete_curve_segments)

                if not current_density or not avg_voltage:
                    print(f"No data for complete curve {i + 1}")
                    continue

                # Sort by current density for proper curve connection
                sorted_data = sorted(zip(current_density, avg_voltage))
                current_density, avg_voltage = zip(*sorted_data)
                current_density, avg_voltage = list(current_density), list(avg_voltage)

                curve_color = curve_colors[i]
                # Get curve number from any segment in this complete curve
                curve_number = complete_curve_segments[0].get('pol_curve_number', i + 1)

                # Create label for first curve only
                if short_display_name not in labeled_files:
                    if self.pol_curve_settings.get("curve_selection") == "all":
                        label = f"{short_display_name} (Curves 1-{num_curves})"
                    elif num_curves == 1:
                        label = f"{short_display_name} (Curve #{curve_number})"
                    else:
                        label = f"{short_display_name} (Multiple Curves)"
                    labeled_files.add(short_display_name)
                else:
                    label = "_nolegend_"

                # Plot curve as connected line with markers
                self.ax.plot(current_density, avg_voltage, linestyle='-', color=curve_color,
                             linewidth=2, marker='o', markersize=4, label=label)

                print(
                    f"Plotted complete curve #{curve_number} with {len(current_density)} points from {len(complete_curve_segments)} segments")

                # Extrapolate
                if len(current_density) >= 3:
                    self.extrapolate_polarization_curve(current_density, avg_voltage, curve_color)

                # Add curve number annotation at highest current density point
                if len(current_density) > 0 and len(avg_voltage) > 0:
                    max_current_idx = np.argmax(current_density)
                    self.ax.annotate(f'#{curve_number}',
                                     (current_density[max_current_idx], avg_voltage[max_current_idx]),
                                     xytext=(5, 5), textcoords='offset points',
                                     fontsize=8, color=curve_color, fontweight='bold')

            # Add error bands for primary data (use first curve)
            if is_primary and selected_complete_curves:
                first_complete_curve = selected_complete_curves[0]
                current_density, avg_voltage = self.process_complete_polarization_curve(first_complete_curve)
                if current_density and avg_voltage:
                    self.add_complete_curve_error_bands(first_complete_curve, current_density, avg_voltage, base_color)

        except Exception as e:
            logger.error(f"Error plotting multiple polarization curves for {file_name}: {e}")

    def add_complete_curve_error_bands(self, complete_curve_segments, current_density, avg_voltage, color):
        """Add error bands for complete polarization curves."""
        try:
            if not complete_curve_segments:
                return

            # Skip if no error bands should be displayed
            if self.sigma_band_type == "No error band":
                return

            std_devs = []
            sigma_2_devs = []

            # Define slice indices once
            slice_idx = slice(-12, -2)

            for segment in complete_curve_segments:
                seg_data = segment.get('data')
                if seg_data is None:
                    std_devs.append(0)
                    sigma_2_devs.append(0)
                    continue

                # Check columns once and extract sliced data
                has_std_dev = 'std dev voltage (v)' in seg_data.columns
                has_2sigma = '2σ voltage (v)' in seg_data.columns

                if has_std_dev:
                    # Vectorized extraction and mean calculation
                    std_dev_vals = seg_data.iloc[slice_idx]['std dev voltage (v)'].values
                    std_dev_val = np.mean(std_dev_vals)
                else:
                    std_dev_val = 0
                std_devs.append(std_dev_val)

                # Get 2σ values
                if has_2sigma:
                    sigma_2_vals = seg_data.iloc[slice_idx]['2σ voltage (v)'].values
                    sigma_2_devs.append(np.mean(sigma_2_vals))
                else:
                    sigma_2_devs.append(std_dev_val * 2)

            std_devs = np.array(std_devs)
            sigma_2_devs = np.array(sigma_2_devs)
            avg_voltage = np.array(avg_voltage)

            if self.sigma_band_type == "1σ only":
                if len(std_devs) == len(avg_voltage):
                    self.ax.fill_between(current_density, avg_voltage - std_devs, avg_voltage + std_devs,
                                         color=color, alpha=0.3)
            elif self.sigma_band_type == "2σ only":
                if len(sigma_2_devs) == len(avg_voltage):
                    self.ax.fill_between(current_density, avg_voltage - sigma_2_devs, avg_voltage + sigma_2_devs,
                                         color=color, alpha=0.3)
            elif self.sigma_band_type == "QC bands (1σ green + 2σ yellow)":
                if len(std_devs) == len(avg_voltage) and len(sigma_2_devs) == len(avg_voltage):
                    self.ax.fill_between(current_density, avg_voltage - sigma_2_devs, avg_voltage + sigma_2_devs,
                                         color='yellow', alpha=0.2)
                    self.ax.fill_between(current_density, avg_voltage - std_devs, avg_voltage + std_devs,
                                         color='green', alpha=0.3)

        except Exception as e:
            logger.error(f"Error adding complete curve error bands: {e}")

    def process_complete_polarization_curve(self, complete_curve_segments):
        """
        Process a complete polarization curve (made of multiple segments) to extract current density and voltage points.

        Parameters:
        complete_curve_segments (list): List of segments forming one complete polarization curve

        Returns:
        tuple: (current_density, avg_voltage) lists
        """
        try:
            if not complete_curve_segments:
                return [], []

            current_density = []
            avg_voltage = []

            # Define slices once
            current_slice = slice(-10, -5)
            voltage_slice = slice(-12, -2)

            # Process each segment in the complete curve
            for segment in complete_curve_segments:
                data = segment.get('data')
                if data is None or data.empty:
                    continue

                # Check columns once
                if 'current(a)' not in data.columns or 'voltage(v)' not in data.columns:
                    continue

                # Vectorized extraction using numpy
                current_values = data['current(a)'].iloc[current_slice].values.mean()
                density = (current_values / 4) * 1000  # Convert to mA/cm²

                avg_volt = data['voltage(v)'].iloc[voltage_slice].values.mean()

                current_density.append(density)
                avg_voltage.append(avg_volt)

            return current_density, avg_voltage

        except Exception as e:
            logger.error(f"Error processing complete polarization curve: {e}")
            return [], []

    def add_single_segment_error_bands(self, segments, current_density, avg_voltage, color):
        """Add error bands for primary data polarization curves from single segments."""
        try:
            if not segments or not current_density or not avg_voltage:
                return

            # Skip if no error bands should be displayed
            if self.sigma_band_type == "No error band":
                return

            # For single segment error bands, we need to sample std dev along the curve
            segment = segments[0]
            seg_data = segment.get('data')

            if seg_data is None or 'std dev voltage (v)' not in seg_data.columns:
                return

            total_points = len(seg_data)
            if total_points < 10:
                indices = np.arange(total_points)
            else:
                # Sample points more efficiently using modulo arithmetic
                sample_rate = max(1, total_points // len(avg_voltage))
                indices = np.arange(0, min(total_points, len(avg_voltage) * sample_rate),
                                   max(1, total_points // max(10, len(avg_voltage))))
                indices = np.clip(indices, 0, total_points - 1)

            # Extract std dev values at these indices
            std_dev_col = seg_data['std dev voltage (v)'].values
            std_devs = np.take(std_dev_col, indices, mode='clip')[:len(avg_voltage)]

            if len(std_devs) == len(avg_voltage) == len(current_density):
                self.ax.fill_between(current_density,
                                     np.array(avg_voltage) - std_devs,
                                     np.array(avg_voltage) + std_devs,
                                     color=color, alpha=0.2)

        except Exception as e:
            logger.error(f"Error adding single segment error bands: {e}")

    def add_polarization_error_bands(self, curve_segments, current_density, avg_voltage, color):
        """Add error bands for primary data polarization curves."""
        try:
            if not curve_segments:
                return

            # Skip if no error bands should be displayed
            if self.sigma_band_type == "No error band":
                return

            std_devs = []
            sigma_2_devs = []

            # Define slice once
            slice_idx = slice(-12, -2)

            for segment in curve_segments:
                seg_data = segment.get('data')
                if seg_data is not None:
                    if 'std dev voltage (v)' in seg_data.columns:
                        std_dev_vals = seg_data.iloc[slice_idx]['std dev voltage (v)'].values
                        std_devs.append(np.mean(std_dev_vals))
                    else:
                        std_devs.append(0)

                    # Get 2σ values
                    if '2σ voltage (v)' in seg_data.columns:
                        sigma_2_vals = seg_data.iloc[slice_idx]['2σ voltage (v)'].values
                        sigma_2_devs.append(np.mean(sigma_2_vals))
                    else:
                        sigma_2_devs.append(std_devs[-1] * 2)  # Calculate from 1σ
                else:
                    std_devs.append(0)
                    sigma_2_devs.append(0)

            std_devs = np.array(std_devs)
            sigma_2_devs = np.array(sigma_2_devs)
            avg_voltage = np.array(avg_voltage)

            # Plot based on sigma band type
            if self.sigma_band_type == "1σ only":
                if len(std_devs) == len(avg_voltage):
                    self.ax.fill_between(current_density,
                                         avg_voltage - std_devs,
                                         avg_voltage + std_devs,
                                         color=color, alpha=0.3)

            elif self.sigma_band_type == "2σ only":
                if len(sigma_2_devs) == len(avg_voltage):
                    self.ax.fill_between(current_density,
                                         avg_voltage - sigma_2_devs,
                                         avg_voltage + sigma_2_devs,
                                         color=color, alpha=0.3)

            elif self.sigma_band_type == "QC bands (1σ green + 2σ yellow)":
                if len(std_devs) == len(avg_voltage) and len(sigma_2_devs) == len(avg_voltage):
                    # Plot 2σ band (yellow) first
                    self.ax.fill_between(current_density,
                                         avg_voltage - sigma_2_devs,
                                         avg_voltage + sigma_2_devs,
                                         color='yellow', alpha=0.2)

                    # Plot 1σ band (green) on top
                    self.ax.fill_between(current_density,
                                         avg_voltage - std_devs,
                                         avg_voltage + std_devs,
                                         color='green', alpha=0.3)

        except Exception as e:
            logger.error(f"Error adding polarization error bands: {e}")


    def process_single_polarization_segment(self, segment):
        """
        Process a single polarization segment to extract current density and voltage points.
        This creates multiple points along the segment to form a curve.
        """
        try:
            data = segment.get('data')
            if not data or 'current(a)' not in data.columns or 'voltage(v)' not in data.columns:
                return [], []

            # Sample points along the segment to create the curve
            # Take points at regular intervals to show the full curve shape
            total_points = len(data)
            if total_points < 10:
                # If segment is short, use all points
                indices = range(total_points)
            else:
                # Sample points along the curve (start, middle points, end)
                indices = []
                indices.extend(range(0, min(10, total_points)))  # First 10 points
                if total_points > 20:
                    # Add some middle points
                    middle_start = total_points // 3
                    middle_end = 2 * total_points // 3
                    indices.extend(range(middle_start, middle_end, max(1, (middle_end - middle_start) // 10)))
                indices.extend(range(max(0, total_points - 10), total_points))  # Last 10 points

                # Remove duplicates and sort
                indices = sorted(list(set(indices)))

            current_density = []
            avg_voltage = []

            for idx in indices:
                current_val = data['current(a)'].iloc[idx]
                voltage_val = data['voltage(v)'].iloc[idx]

                # Convert current to current density (A to mA/cm²)
                density = (current_val / 4) * 1000  # Assuming 4 cm² electrode area

                current_density.append(density)
                avg_voltage.append(voltage_val)

            return current_density, avg_voltage

        except Exception as e:
            logger.error(f"Error processing single polarization segment: {e}")
            return [], []


    def plot_single_polarization_curve(self, data, labeled_files, special_labels=None,
                                       color_selections=None, is_primary=False):
        """
        Plot single polarization curve - restored original functionality.
        """
        try:
            if 'segments' not in data:
                return

            segments = self.filter_segments(data['segments'])
            last_pol_curve_segments = self.find_last_polarization_curve(segments)

            if not last_pol_curve_segments:
                logger.warning("No valid polarization curve found.")
                return

            current_density, avg_voltage = self.process_polarization_curve(last_pol_curve_segments)

            if not current_density or not avg_voltage:
                return

            # Sort by current density for proper curve connection
            sorted_data = sorted(zip(current_density, avg_voltage))
            current_density, avg_voltage = zip(*sorted_data)
            current_density, avg_voltage = list(current_density), list(avg_voltage)

            file_name = data.get('file_name')
            short_display_name = get_display_name(file_name, special_labels, is_primary, self.extract_display_name)
            file_color = color_selections.get(file_name, {}).get('hex_code', get_color_for_file(file_name))

            # Plot as connected curve with markers
            self.ax.plot(current_density, avg_voltage, linestyle='-', color=file_color,
                         linewidth=2, marker='o', markersize=4, label=f"{short_display_name}")

            # Extrapolate based on the last 3 points
            if len(current_density) >= 3:
                self.extrapolate_polarization_curve(current_density, avg_voltage, file_color)

            # Add error bands for primary data
            if is_primary:
                self.add_polarization_error_bands(last_pol_curve_segments, current_density, avg_voltage, file_color)

        except Exception as e:
            logger.error(f"Error plotting single polarization curve: {e}")


    def filter_segments(self, segments):
        """Filters segments based on user-selected segment type."""
        if self.segment_type == "constant":
            return [s for s in segments if s["type"] == "constant"]
        elif self.segment_type == "polarization":
            return [s for s in segments if s["type"] == "polarization"]
        return segments


    def extract_display_name(self, file_name):
        """
        Extracts a readable display name from the file name.
        """
        if not hasattr(self, '_display_name_cache'):
            self._display_name_cache = {}

        if file_name in self._display_name_cache:
            return self._display_name_cache[file_name]

        name_without_ext = os.path.splitext(file_name)[0]
        if " I " in name_without_ext:
            parts = name_without_ext.split(" I ", 1)  # Split only once
            display_name = f"{parts[0]} I {parts[1].split()[0]}"
        else:
            display_name = name_without_ext

        self._display_name_cache[file_name] = display_name
        return display_name

    def plot_primary_data(self, primary_data, labeled_files, special_labels=None, color_selections=None):
        """Plots primary dataset segments."""
        primary_file_name = primary_data['file_name']
        primary_display_name = get_display_name(primary_file_name, special_labels, is_primary=True,
                                                extract_display_name=self.extract_display_name)
        file_color = color_selections.get(primary_file_name, {}).get('hex_code', get_color_for_file(primary_file_name))

        # NEW: Pass sigma_band_type to get correct std dev for label
        mean_voltage, mean_std_dev = get_longest_segment_stats(primary_data['segments'], is_primary=True,
                                                               sigma_band_type=self.sigma_band_type)

        # NEW: Update label based on sigma band type and show_mean_voltage setting
        if self.show_mean_voltage and mean_voltage is not None and mean_std_dev is not None:
            if self.sigma_band_type == "2σ only":
                label = f"{primary_display_name} (mean = {mean_voltage:.3f} V ± {mean_std_dev:.1f} mV, 2σ)"
            elif self.sigma_band_type == "QC bands (1σ green + 2σ yellow)":
                label = f"{primary_display_name} (mean = {mean_voltage:.3f} V ± {mean_std_dev:.1f} mV, QC bands)"
            else:  # "1σ only"
                label = f"{primary_display_name} (mean = {mean_voltage:.3f} V ± {mean_std_dev:.1f} mV)"
        else:
            label = primary_display_name

        segments = (primary_data['segments'] if self.segment_type == "both"
                    else self.filter_segments(primary_data['segments']))

        for segment in segments:
            self.plot_segment(segment, file_color, label, labeled_files, primary_display_name, is_primary=True)


    def plot_additional_data(self, additional_data, labeled_files, special_labels=None, color_selections=None):
        """Plots additional datasets."""
        file_name = additional_data.get('file_name')
        short_display_name = get_display_name(file_name, special_labels, is_primary=False)
        file_color = color_selections.get(file_name, {}).get('hex_code', get_color_for_file(file_name))
        mean_voltage, _ = get_longest_segment_stats(additional_data['segments'], is_primary=False)
        
        # Create label based on show_mean_voltage setting
        if self.show_mean_voltage and mean_voltage is not None:
            label = (f"{short_display_name} (mean = {mean_voltage:.3f} V)"
                     if short_display_name not in labeled_files else "_nolegend_")
        else:
            label = short_display_name if short_display_name not in labeled_files else "_nolegend_"

        if 'segments' in additional_data:
            segments = (additional_data['segments'] if self.segment_type == "both"
                        else self.filter_segments(additional_data['segments']))

            for segment in segments:
                self.plot_segment(segment, file_color, label, labeled_files, short_display_name, is_primary=False)


    def plot_segment(self, segment, file_color, label, labeled_files, display_name, is_primary):
        """Plots a single segment."""
        data = segment['data']
        if 'adjusted_time' not in data or 'voltage(v)' not in data:
            logger.error(f"Skipping segment: Missing Voltage or adjusted time columns.")
            return

        use_filtering = not (self.segment_type == "both" and segment["type"] == "polarization")

        filtered_data = (
            self.rolling_filter(data, "voltage(v)", self.rolling_window_params.get("window_size", 30),
                                self.rolling_window_params.get("threshold", 0.05)).copy()
            if use_filtering else data.copy()
        )

        if self.segment_type == "both" and segment["type"] == "polarization":
            filtered_data = data.copy()
        else:
            filtered_data = filtered_data.iloc[::self.downsampling_rate, :]

        time, voltage = filtered_data['adjusted_time'].values, filtered_data["voltage(v)"].values

        linestyle = '-' if segment["type"] in ['polarization', 'constant'] else '--'

        # Check label tracking efficiently using set membership
        is_labeled = display_name in labeled_files
        self.ax.plot(time, voltage, linestyle=linestyle,
                     linewidth=1.5, label=label if not is_labeled else "_nolegend_",
                     color=file_color)

        if self.show_degradation_rate:
            self.plot_degradation_rate(segment, time, voltage, file_color, labeled_files, display_name)

        if is_primary:
            self.plot_error_bands(filtered_data, time, voltage, file_color)

        labeled_files.add(display_name)

    def plot_error_bands(self, filtered_data, time, voltage, file_color):
        """Plot error bands based on sigma band type selection."""
        try:
            # Skip if no error bands should be displayed
            if self.sigma_band_type == "No error band":
                return

            # Cache column existence checks
            has_std_dev_1sigma = 'std dev voltage (v)' in filtered_data.columns
            has_2sigma = '2σ voltage (v)' in filtered_data.columns

            if self.sigma_band_type == "1σ only":
                # Original behavior - plot 1σ bands
                if has_std_dev_1sigma:
                    std_dev = filtered_data['std dev voltage (v)'].values
                    self.ax.fill_between(time, voltage - std_dev, voltage + std_dev,
                                         color=file_color, alpha=0.3)

            elif self.sigma_band_type == "2σ only":
                # Plot 2σ bands only
                if has_2sigma:
                    sigma_2_dev = filtered_data['2σ voltage (v)'].values
                    self.ax.fill_between(time, voltage - sigma_2_dev, voltage + sigma_2_dev,
                                         color=file_color, alpha=0.3)
                elif has_std_dev_1sigma:
                    # Fallback: calculate 2σ from 1σ
                    std_dev_2 = filtered_data['std dev voltage (v)'].values * 2
                    self.ax.fill_between(time, voltage - std_dev_2, voltage + std_dev_2,
                                         color=file_color, alpha=0.3)

            elif self.sigma_band_type == "QC bands (1σ green + 2σ yellow)":
                # Plot both 1σ (green) and 2σ (yellow) bands
                if has_std_dev_1sigma:
                    std_dev_1 = filtered_data['std dev voltage (v)'].values
                    # Get or calculate 2σ
                    if has_2sigma:
                        sigma_2_dev = filtered_data['2σ voltage (v)'].values
                    else:
                        sigma_2_dev = std_dev_1 * 2

                    # Plot 2σ band (yellow) first (background)
                    self.ax.fill_between(time, voltage - sigma_2_dev, voltage + sigma_2_dev,
                                         color='yellow', alpha=0.2,
                                         label='2σ Warning Band' if not hasattr(self, '_sigma_legend_added') else '')
                    # Plot 1σ band (green) on top (foreground)
                    self.ax.fill_between(time, voltage - std_dev_1, voltage + std_dev_1,
                                         color='green', alpha=0.3,
                                         label='1σ Acceptance Band' if not hasattr(self, '_sigma_legend_added') else '')
                    # Mark that we've added sigma legends (to avoid duplicates)
                    self._sigma_legend_added = True


        except Exception as e:
            logger.error(f"Error plotting error bands: {e}")

    def plot_degradation_rate(self, segment, time_arr, voltage_arr, file_color, labeled_files, display_name):
        """Plots the degradation rate for a given segment."""
        try:
            if len(time_arr) == 0 or len(voltage_arr) == 0:
                logger.error("Error: Time or voltage array is empty!")
                return

            if segment["type"] != "constant":
                return

            if len(time_arr) < 6 or len(voltage_arr) < 6:
                return

            valid_mask = time_arr >= 5
            time_arr = time_arr[valid_mask]
            voltage_arr = voltage_arr[valid_mask]

            if len(time_arr) > 5:
                sampling_interval = np.median(np.diff(time_arr))
                window_hours = 10
                window_size = int(window_hours / sampling_interval)

                if len(voltage_arr) < window_size:
                    return

                voltage_smooth_sg = savgol_filter(voltage_arr, window_size, polyorder=3)
                voltage_smooth = gaussian_filter1d(voltage_smooth_sg, sigma=8)
                self.ax.plot(time_arr, voltage_smooth_sg,
                             linestyle="--",
                             linewidth=1.2,
                             color=file_color,
                             alpha=0.6)

                dv_dt = np.gradient(voltage_smooth, time_arr)
                shift_amount = window_size // 2
                time_shifted = time_arr[shift_amount:]
                dv_dt_mV_day = dv_dt * 24 * 1000
                dv_dt_mV_day = gaussian_filter1d(dv_dt_mV_day, sigma=4)
                dv_dt_mV_day = dv_dt_mV_day[shift_amount:]

                if self.ax2 is None:
                    self.ax2 = self.ax.twinx()

                if self.ax2:
                    try:
                        self.ax2.plot(time_shifted, dv_dt_mV_day, linestyle=':', linewidth=1,
                                      color=file_color,
                                      label=f'{display_name} dV/dt' if display_name not in labeled_files else "_nolegend_")
                        self.ax2.set_ylabel('Degradation Rate (mV/day)',
                                            fontsize=self.plot_params['fontsize_axes_labels'])

                        total_duration = getattr(self, "total_experiment_duration", segment["duration"])

                        if total_duration > 300:
                            interval_hours = segment["duration"]
                        elif total_duration > 168:
                            interval_hours = 48
                        elif total_duration > 72:
                            interval_hours = 24
                        else:
                            interval_hours = 12

                        if total_duration > 300:
                            center_time = time_shifted[len(time_shifted) // 2]
                            mean_rate = np.mean(dv_dt_mV_day)

                            self.ax2.text(center_time, mean_rate, f"{mean_rate:.2f} mV/day",
                                          color=file_color, fontsize=10, ha='center')
                        else:
                            current_time = time_shifted[0] + interval_hours
                            end_time = time_shifted[-1]
                            while current_time < end_time:
                                mask = (time_shifted >= current_time - interval_hours / 2) & (
                                        time_shifted < current_time + interval_hours / 2)
                                if mask.sum() > 0:
                                    avg = np.mean(dv_dt_mV_day[mask])
                                    self.ax2.text(current_time, avg, f"{avg:.2f} mV/day",
                                                  color=file_color, fontsize=10, ha='center')
                                current_time += interval_hours

                    except Exception as plot_error:
                        logger.error(f"Error while plotting annotation degradation rate: {plot_error}")

        except Exception as e:
            logger.error(f"Error during degradation rate plotting: {e}")


    def find_last_polarization_curve(self, segments):
        """Finds the last polarization curve by detecting the last ramp-down section."""
        ramp_down_segments = []
        previous_current = None
        for segment in reversed(segments):
            if segment["type"] == "polarization":
                current = segment["data"]["current(a)"].iloc[-1]
                if previous_current is not None and current < previous_current:
                    break
                ramp_down_segments.insert(0, segment)
                previous_current = current
        return ramp_down_segments if ramp_down_segments else None

    def process_polarization_curve(self, segments):
        """Processes polarization curve data to extract avg voltage and current density."""
        current_density = []
        avg_voltage = []

        # Define slices once
        current_slice = slice(-10, -5)
        voltage_slice = slice(-12, -2)

        for segment in segments:
            data = segment['data']
            if 'current(a)' not in data or 'voltage(v)' not in data or 'adjusted_time' not in data:
                logger.error("Missing required data columns in polarization segment.")
                continue

            # Use vectorized numpy operations
            current_values = data['current(a)'].iloc[current_slice].values.mean()
            density = (current_values / 4) * 1000

            avg_volt = data['voltage(v)'].iloc[voltage_slice].values.mean()

            current_density.append(density)
            avg_voltage.append(avg_volt)

        return current_density, avg_voltage


    def extrapolate_polarization_curve(self, current_density, avg_voltage, color):
        """Extrapolates the polarization curve based on the last 3 highest current density points."""
        if len(current_density) < 3:
            logger.warning("Not enough points to extrapolate.")
            return

        # OPTIMIZATION: Convert to arrays once for consistent operations
        current_density = np.asarray(current_density)
        avg_voltage = np.asarray(avg_voltage)

        sorted_indices = np.argsort(current_density)
        current_density_sorted = current_density[sorted_indices]
        avg_voltage_sorted = avg_voltage[sorted_indices]

        x_last = current_density_sorted[-4:]
        y_last = avg_voltage_sorted[-4:]

        coeffs = np.polyfit(x_last, y_last, 1)
        poly_func = np.poly1d(coeffs)

        x_extrapolated = np.linspace(x_last[0], x_last[-1] * 1.2, 100)
        y_extrapolated = poly_func(x_extrapolated)

        self.ax.plot(x_extrapolated, y_extrapolated, linestyle='dashed', color=color, alpha=0.5)

    def set_plot_limits(self, primary_data, additional_data_list, segment_type):
        """Sets the limits for the plot axes dynamically."""
        # Check if custom axis ranges are provided
        if self.axis_range_x_min is not None or self.axis_range_x_max is not None or \
           self.axis_range_y_min is not None or self.axis_range_y_max is not None:
            # Use custom ranges if provided
            if self.axis_range_x_min is not None and self.axis_range_x_max is not None:
                self.ax.set_xlim(self.axis_range_x_min, self.axis_range_x_max)
            elif self.axis_range_x_min is not None:
                current_xlim = self.ax.get_xlim()
                self.ax.set_xlim(self.axis_range_x_min, current_xlim[1])
            elif self.axis_range_x_max is not None:
                current_xlim = self.ax.get_xlim()
                self.ax.set_xlim(current_xlim[0], self.axis_range_x_max)

            if self.axis_range_y_min is not None and self.axis_range_y_max is not None:
                self.ax.set_ylim(self.axis_range_y_min, self.axis_range_y_max)
            elif self.axis_range_y_min is not None:
                current_ylim = self.ax.get_ylim()
                self.ax.set_ylim(self.axis_range_y_min, current_ylim[1])
            elif self.axis_range_y_max is not None:
                current_ylim = self.ax.get_ylim()
                self.ax.set_ylim(current_ylim[0], self.axis_range_y_max)
            
            # If any range was set manually, we're done
            if self.axis_range_x_min is not None or self.axis_range_x_max is not None or \
               self.axis_range_y_min is not None or self.axis_range_y_max is not None:
                return

        # ...existing code...
        x_values = []
        y_values = []

        if segment_type == "polarization":
            datasets = []
            if isinstance(primary_data, list):
                datasets.extend(primary_data)
            elif primary_data and 'segments' in primary_data:
                datasets.append(primary_data)

            datasets.extend(additional_data_list)

            for dataset in datasets:
                # Get selected curves instead of just last curve
                selected_curves = self.get_selected_polarization_curves(dataset)
                if selected_curves:
                    # Handle both single curves and multiple curves
                    for curve_segments in selected_curves:
                        if isinstance(curve_segments, list):
                            # Complete curve - process it
                            current_density, avg_voltage = self.process_complete_polarization_curve(curve_segments)
                        else:
                            # Single segment - shouldn't happen with new logic, but handle it
                            current_density, avg_voltage = self.process_polarization_curve([curve_segments])

                        if current_density and avg_voltage:
                            x_values.extend(current_density)
                            y_values.extend(avg_voltage)

            if x_values and y_values:
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
                x_margin = (x_max - x_min) * 0.02 if x_max > x_min else 0.1
                y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
                self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
                self.ax.set_ylim(y_min - y_margin, y_max + y_margin)

        else:
            datasets = []
            if isinstance(primary_data, list):
                datasets.extend(primary_data)
            elif primary_data and 'segments' in primary_data:
                datasets.append(primary_data)
            datasets.extend(additional_data_list)

            window_size = self.rolling_window_params.get("window_size", 30)
            threshold = self.rolling_window_params.get("threshold", 0.05)

            for dataset in datasets:
                for segment in dataset.get('segments', []):
                    if segment_type != "both" and segment['type'] != segment_type:
                        continue

                    data = segment['data']
                    filtered_data = self.rolling_filter(data, "voltage(v)", window_size, threshold).copy()
                    if not filtered_data.empty and 'adjusted_time' in filtered_data and "voltage(v)" in filtered_data:
                        x_values.extend(filtered_data['adjusted_time'].values)
                        y_values.extend(filtered_data["voltage(v)"].values)
                    else:
                        logger.error(f"Missing 'voltage(v)' data in segment.")

            # Ensure limits are properly set before applying margins
            if x_values and y_values:
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
                x_margin = (x_max - x_min) * 0.02 if x_max > x_min else 0.1
                y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
                self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
                self.ax.set_ylim(y_min - y_margin, y_max + y_margin)
            else:
                logger.warning("No valid data found for axis limits.")

    def finalize_plot(self, primary_data, additional_data_list, segment_type):
        """
        Finalizes the plot by setting axis labels, title, and grid.
        """
        # Calculate current density ONLY for constant segment types
        current_density = None
        longest_segment = None

        # Calculate current density only if segment_type is constant
        if segment_type == "constant":
            if primary_data:
                if isinstance(primary_data, list):
                    # If it's a list, check each item for segments
                    for item in primary_data:
                        if isinstance(item, dict) and 'segments' in item:
                            current_longest = max(item['segments'], key=lambda seg: seg['duration'], default=None)
                            if current_longest and (
                                    longest_segment is None or current_longest['duration'] > longest_segment[
                                'duration']):
                                longest_segment = current_longest
                else:
                    if 'segments' in primary_data:
                        longest_segment = max(primary_data['segments'], key=lambda seg: seg['duration'], default=None)

            if longest_segment is None and additional_data_list:
                for additional_data in additional_data_list:
                    if 'segments' in additional_data:
                        possible_longest = max(additional_data['segments'], key=lambda seg: seg['duration'],
                                               default=None)
                        if possible_longest and (
                                longest_segment is None or possible_longest['duration'] > longest_segment['duration']):
                            longest_segment = possible_longest

            if longest_segment and (
                    'current(a)' in longest_segment['data'].columns or 'Average Current (A)' in longest_segment[
                'data'].columns):
                # Use the appropriate current column name
                current_col = 'current(a)' if 'current(a)' in longest_segment['data'].columns else 'Average Current (A)'
                mean_current = longest_segment['data'][current_col].mean()
                current_density = (mean_current / 4) * 1000

        # Set labels based on segment type
        if segment_type == "polarization":
            self.ax.set_xlabel("Current Density (mA/cm²)", fontsize=self.plot_params['fontsize_axes_labels'])
            if current_density is not None:
                self.ax.set_ylabel(f'Voltage at {current_density:.0f} mA/cm²',
                                   fontsize=self.plot_params['fontsize_axes_labels'])
            else:
                self.ax.set_ylabel("Voltage (V)", fontsize=self.plot_params['fontsize_axes_labels'])
        else:
            self.ax.set_xlabel('Time (hours)', fontsize=self.plot_params['fontsize_axes_labels'])
            if current_density is not None:
                self.ax.set_ylabel(f'Voltage at {current_density:.0f} mA/cm²',
                                   fontsize=self.plot_params['fontsize_axes_labels'])
            else:
                self.ax.set_ylabel('Voltage (V)', fontsize=self.plot_params['fontsize_axes_labels'])
        self.ax.set_title(self.plot_params['plot_title'], fontsize=self.plot_params['fontsize_title'])
        self.ax.tick_params(axis='both', labelsize=self.plot_params.get('fontsize_ticks', 10))
        if self.plot_params['use_grid']:
            self.ax.grid(visible=True, linestyle='--', alpha=0.7)
        else:
            self.ax.grid(visible=False)


    def update_legend(self, color_selections=None):
        """
        Dynamically updates the legend to only include datasets with valid labels.
        """
        handles, labels = self.ax.get_legend_handles_labels()

        # Filter out "_nolegend_" (used for segments without unique labels)
        valid_handles_labels = [(h, l) for h, l in zip(handles, labels) if l != "_nolegend_"]

        if valid_handles_labels:
            valid_handles, valid_labels = zip(*valid_handles_labels)
            custom_legend = self.ax.legend(
                valid_handles, valid_labels, 
                fontsize=self.plot_params['fontsize_legends'],
                loc='best',
                framealpha=0.95,  # Slightly transparent but readable
                edgecolor='black',  # Add border for better visibility
                fancybox=True,  # Rounded corners
                shadow=True  # Add subtle shadow
            )

            # Match legend text color with line color, considering user-defined colors
            for legend_text, handle, label in zip(custom_legend.get_texts(), valid_handles, valid_labels):
                if color_selections and label in color_selections:
                    hex_color = color_selections[label]['hex_code']
                elif isinstance(handle, plt.Line2D):
                    hex_color = handle.get_color()
                elif isinstance(handle, PathCollection):
                    hex_color = handle.get_facecolor()[0]

                legend_text.set_color(hex_color)
        else:
            logger.warning("No valid labels found for legend.")


    def save_plot(self, folder, filename, file_format="PNG", show=True):
        """
        Saves the generated plot to the specified folder, optionally opening it.

        Parameters:
        folder (str): Folder path where the plot will be saved
        filename (str): Filename without extension
        file_format (str): File format (PNG, SVG, PDF, JPG) - default is PNG
        show (bool): Open an interactive window after saving (off for batch runs)
        """
        try:
            if not folder:
                raise Exception("No folder specified for saving the plot.")
            if not filename:
                raise Exception("No filename specified for saving the plot.")

            if hasattr(self, 'fig') and self.fig:
                # Use os.path.join for cross-platform compatibility
                file_format_lower = file_format.lower()
                file_path = os.path.join(folder, f"{filename}.{file_format_lower}")
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight', format=file_format_lower)
                print(f"Plot saved to: {file_path}")
                logger.info(f"Plot saved to: {file_path}")

                # Open the plot in an interactive matplotlib window
                if show:
                    try:
                        plt.show()
                        logger.info("Interactive matplotlib window opened")
                    except Exception as show_error:
                        logger.warning(f"Could not open interactive plot window: {show_error}")
            else:
                raise Exception("No plot has been generated to save.")


        except Exception as e:
            raise Exception(f"Error saving plot: {e}")


    def show_plot(self):
        """
        Displays the generated plot.
        """
        try:
            if hasattr(self, 'fig') and self.fig:
                plt.show()
            else:
                raise Exception("No plot has been generated to display.")
        except Exception as e:
            raise Exception(f"Error displaying plot: {e}")


def get_longest_segment_stats(segments, is_primary=False, sigma_band_type="1σ only"):
    """
    Finds the longest segment based on duration and calculates:
    - Mean voltage
    - Mean standard deviation of voltage (1σ or 2σ based on selection)
    """
    if not segments:
        return None, None, 0

    longest_segment = segments[0]
    for segment in segments[1:]:
        if segment['duration'] > longest_segment['duration']:
            longest_segment = segment

    data = longest_segment['data']

    if 'voltage(v)' not in data.columns:
        return None, None

    mean_voltage = data['voltage(v)'].mean()

    if is_primary:
        if sigma_band_type == "2σ only":
            # Use 2σ values for the label
            if '2σ voltage (v)' in data.columns:
                mean_std_dev = data['2σ voltage (v)'].mean() * 1000  # Convert to mV
                return mean_voltage, mean_std_dev
            elif 'std dev voltage (v)' in data.columns:
                # Fallback: calculate 2σ from 1σ
                mean_std_dev = data['std dev voltage (v)'].mean() * 2 * 1000  # 2σ in mV
                return mean_voltage, mean_std_dev
        elif sigma_band_type == "QC bands (1σ green + 2σ yellow)":
            # For QC bands, show 1σ in label (the inner green zone)
            if 'std dev voltage (v)' in data.columns:
                mean_std_dev = data['std dev voltage (v)'].mean() * 1000  # 1σ in mV
                return mean_voltage, mean_std_dev
        else:  # "1σ only" - original behavior
            if 'std dev voltage (v)' in data.columns:
                mean_std_dev = data['std dev voltage (v)'].mean() * 1000  # 1σ in mV
                return mean_voltage, mean_std_dev

    return mean_voltage, None


def get_total_experiment_duration(primary_data_list, additional_data_list):
    """Compute total experiment length across all files and segments."""
    max_time = 0

    for dataset in (primary_data_list or []) + (additional_data_list or []):
        for segment in dataset.get("segments", []):
            if "data" in segment and "adjusted_time" in segment["data"].columns:
                segment_max_time = segment["data"]["adjusted_time"].max()
                if pd.notnull(segment_max_time):
                    max_time = max(max_time, segment_max_time)

    return max_time


def sliding_window_slope(time_arr, voltage_arr, window_size=15):
    slopes = np.full_like(voltage_arr, fill_value=np.nan, dtype=np.float64)

    for i in range(window_size, len(time_arr)):
        x = time_arr[i - window_size:i]
        y = voltage_arr[i - window_size:i]

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator != 0:
            slopes[i] = numerator / denominator

    return slopes

