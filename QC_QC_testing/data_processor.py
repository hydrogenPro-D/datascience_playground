import pandas as pd
import numpy as np
import os
import logging
import warnings
import time
from python_calamine import CalamineWorkbook

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

logger = logging.getLogger(__name__)

MIN_SEGMENT_DURATION = 3 / 60  # 3 minutes in hours


def _calamine_sheet_to_df(sheet):
    """
    Convert calamine sheet → pandas DataFrame
    """
    data = sheet.to_python()
    # first row = headers (typical Excel structure)
    headers = data[0]
    rows = data[1:]
    return pd.DataFrame(rows, columns=headers)


class DataProcessor:
    def __init__(self, primary_files=None, additional_files=None, rolling_window_params=None):
        """
        Initializes the DataProcessor with the primary and additional file paths.

        Parameters:
        primary_file (str): Path to the primary data file.
        additional_files (list of str): Paths to additional data files.
        rolling_window_params (dict): Parameters for rolling window filtering.
        """
        self.primary_files = primary_files or []
        self.additional_files = additional_files or []
        self.rolling_window_params = rolling_window_params or {"window_size": 20, "threshold": 0.05}

    def _load_excel_file(self, file_path, sheet_name):
        """
        Common helper function to load and preprocess an Excel file.
        """
        try:

            wb = CalamineWorkbook.from_path(file_path)
            sheet = wb.get_sheet_by_name(sheet_name)
            df = _calamine_sheet_to_df(sheet)
            df.columns = df.columns.str.strip()

            return df
        except Exception as e:
            raise Exception(f"Error loading Excel file {file_path} (sheet: {sheet_name}): {e}")

    def load_primary_data(self):
        """
        Loads and processes one or more primary data files.
        Returns a list of dictionaries with 'loaded' DataFrame and 'file_name'.
        Returns None if no files are provided.
        """
        if not self.primary_files:
            return None  # No primary files provided

        # Define column mappings locally for clarity
        old_format_cols = ["Time (hours)", "Average Voltage (V)", "Std Dev Voltage (V)", 'Average Current (A)']
        new_format_cols = old_format_cols + ["2σ Voltage (V)"]

        column_rename_map = {
            "Average Voltage (V)": "voltage(v)",
            "Average Current (A)": "current(a)",
            "Std Dev Voltage (V)": "std dev voltage (v)",
            "2σ Voltage (V)": "2σ voltage (v)"
        }

        all_primary_data = []
        for file_path in self.primary_files:
            try:
                df = self._load_excel_file(file_path, 'Summary')

                # Check for new format columns
                if not all(col in df.columns for col in new_format_cols):
                    # Check if it's an old format (without 2σ column)
                    if all(col in df.columns for col in old_format_cols):
                        # Add 2σ column by calculating 2 * std dev
                        df["2σ Voltage (V)"] = df["Std Dev Voltage (V)"] * 2
                        logger.info(f"Added 2σ column to old format file: {file_path}")
                    else:
                        raise ValueError(f"Missing required columns in file: {file_path}")

                # Single rename operation instead of two separate ones
                df.rename(columns=column_rename_map, inplace=True)
                all_primary_data.append({"loaded": df, "file_name": os.path.basename(file_path)})

            except Exception as e:
                raise Exception(f"Error loading primary file: {e}, File: {file_path}")

        return all_primary_data

    def load_additional_data(self, file_path):
        """
        Loads and processes an additional data file.

        Parameters:
        file_path (str): Path to the additional data file.

        Returns:
        pd.DataFrame: The processed additional dataset.
        """
        try:
            additional_df = self._load_excel_file(file_path, 'record')


            # Define required columns
            time_col = 'Total Time'
            time_hours_col = 'time hours'
            required_columns = [time_hours_col, 'Voltage(V)', 'Current(A)']

            if time_col in additional_df.columns:
                additional_df[time_hours_col] = self.convert_time_to_hours(additional_df, time_col)

            else:
                raise ValueError(f"Missing '{time_col}' column in {file_path}. Cannot calculate '{time_hours_col}'.")

            if not all(col in additional_df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in {file_path}. Required: {required_columns}")

            return additional_df
        except Exception as e:
            raise Exception(f"Error loading additional file {file_path}: {e}")




    def _detect_columns(self, df, current_column, time_column):
        """
        Detects and validates required columns in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataset to check.
        current_column (str): The column representing current values.
        time_column (str): The column representing time in hours.

        Returns:
        tuple: (time_column, current_column) - validated column names

        Raises:
        ValueError: If required columns are not found.
        """
        df.columns = df.columns.str.lower()

        if time_column is None:
            if 'time (hours)' in df.columns:
                time_column = 'time (hours)'
            elif 'total time' in df.columns:
                df['time hours'] = self.convert_time_to_hours(df, 'total time')
                time_column = 'time hours'
            else:
                raise ValueError("Time column not found in the dataset.")

        if current_column is None:
            possible_current_columns = ['current(a)', 'average current (a)', 'current']
            current_column = next((col for col in possible_current_columns if col in df.columns), None)
            if not current_column:
                raise ValueError("Current column not found in the dataset.")

        required_columns = [time_column, current_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the input data: {missing_columns}")

        return time_column, current_column

    def _iterate_rows(self, df, current_column, time_column, min_duration_constant, steady_current_tolerance):
        """
        Iterates through rows and identifies segments based on current stability.

        Parameters:
        df (pd.DataFrame): The dataset to segment.
        current_column (str): The column representing current values.
        time_column (str): The column representing time in hours.
        min_duration_constant (float): Minimum duration to classify a segment as constant current.
        steady_current_tolerance (float): Tolerance to consider current as steady.

        Returns:
        list: List of segments with indices (not yet converted to DataFrames).
        """
        segments = []
        segment_indices = []
        segment_start_time = None
        steady_state = None
        last_current = None

        # Convert to numpy arrays for faster iteration
        current_values = df[current_column].values
        time_values = df[time_column].values

        # Pre-filter valid indices (non-null currents)
        valid_mask = ~pd.isnull(current_values)
        valid_indices = np.where(valid_mask)[0]

        for idx in valid_indices:
            current = current_values[idx]
            current_time = time_values[idx]

            if last_current is None:
                last_current = current
                segment_start_time = current_time
                steady_state = current
                segment_indices.append(idx)
                continue

            is_steady = abs(current - steady_state) < steady_current_tolerance

            if not is_steady:
                duration = current_time - segment_start_time
                segment_type = "constant" if duration >= min_duration_constant else "polarization"

                segments.append({
                    "type": segment_type,
                    "start_time": segment_start_time,
                    "duration": duration,
                    "indices": segment_indices.copy()
                })

                segment_indices = []
                segment_start_time = current_time
                steady_state = current
            elif current_time - segment_start_time >= min_duration_constant:
                steady_state = (steady_state + current) / 2

            segment_indices.append(idx)
            last_current = current

        return segments, segment_indices, segment_start_time, time_values, len(valid_indices)

    def _finalize_segments(self, df, segments, segment_indices, segment_start_time, time_column, min_duration_constant):
        """
        Handles the final segment and converts all indices to DataFrames.

        Parameters:
        df (pd.DataFrame): The original dataset.
        segments (list): List of segments with indices.
        segment_indices (list): Indices for the final segment being built.
        segment_start_time (float): Start time of the final segment.
        time_column (str): The column representing time in hours.
        min_duration_constant (float): Minimum duration to classify a segment as constant current.
        """
        # Handle final segment
        if segment_indices and segment_start_time is not None:
            duration = df[time_column].iloc[-1] - segment_start_time
            segment_type = "constant" if duration >= min_duration_constant else "polarization"

            segments.append({
                "type": segment_type,
                "start_time": segment_start_time,
                "duration": duration,
                "indices": segment_indices
            })

        for segment in segments:
            segment["data"] = df.take(segment.pop("indices"))

    def _adjust_segments(self, segments, current_column, time_column):
        """
        Filters and adjusts segments, removing invalid or too-short segments.

        Parameters:
        segments (list): List of segments with data.
        current_column (str): The column representing current values.
        time_column (str): The column representing time in hours.

        Returns:
        list: List of adjusted and filtered segments.
        """
        adjusted_segments = []
        running_time_offset = 0

        for segment in segments:
            try:
                if "start_time" not in segment or segment["start_time"] is None:
                    logger.error(f"Segment missing 'start_time': {segment}")
                    continue

                if "voltage(v)" not in segment["data"].columns:
                    logger.error(f"Skipping segment {segment['start_time']} due to missing 'voltage(v)' data.")
                    continue

                if segment["type"] == "constant" and segment["data"][current_column].mean() <= 0.05:
                    continue

                if segment["duration"] < MIN_SEGMENT_DURATION:
                    logger.info(f"Skipping short segment ({segment['duration']} hours) at {segment['start_time']}.")
                    continue

                segment["data"]["adjusted_time"] = segment["data"][time_column] - segment["start_time"] + running_time_offset
                running_time_offset += segment["duration"]

                adjusted_segments.append(segment)
            except Exception as e:
                logger.error(f"Error processing segment: {e}. Segment: {segment}")
                continue

        return adjusted_segments

    def segment_experiment(self, df, current_column=None, time_column=None,
                           min_duration_constant=1.0, polarization_duration_threshold=0.5,
                           steady_current_tolerance=0.05):
        """
        Dynamically segments the dataset into experiment parts based on current stability and durations.

        Parameters:
        df (pd.DataFrame): The dataset to segment.
        current_column (str): The column representing current values. If None, it will be detected dynamically.
        time_column (str): The column representing time in hours. If None, it will be detected dynamically.
        min_duration_constant (float): Minimum duration in hours to classify a segment as constant current.
        polarization_duration_threshold (float): Maximum duration in hours for polarization steps.
        steady_current_tolerance (float): Tolerance to consider current as steady.

        Returns:
        list: A list of dictionaries, each representing a segment with type ('constant' or 'polarization') and data.
        """
        try:
            # Column detection
            time_column, current_column = self._detect_columns(df, current_column, time_column)

            # Iterate through rows using numpy arrays for speed
            segments, segment_indices, segment_start_time, time_values, valid_count = self._iterate_rows(
                df, current_column, time_column, min_duration_constant, steady_current_tolerance
            )

            # Handle final segment and convert indices to DataFrames
            self._finalize_segments(df, segments, segment_indices, segment_start_time, time_column, min_duration_constant)

            # Adjust segments
            adjusted_segments = self._adjust_segments(segments, current_column, time_column)

            # Polarization curve numbering
            adjusted_segments = self.add_polarization_curve_numbering(adjusted_segments)

            return adjusted_segments
        except Exception as e:
            logger.error(f"Error segmenting experiment: {e}")
            raise

    def add_polarization_curve_numbering(self, segments):
        """
        Adds polarization curve numbering to segments using current direction detection.
        Groups consecutive polarization segments into complete polarization curves.
        Detects curves by identifying current increase→decrease patterns.
        """
        try:
            polarization_segments = [seg for seg in segments if seg["type"] == "polarization"]

            if not polarization_segments:
                return segments

            complete_curves = []
            current_curve_segments = []
            curve_counter = 0

            for i, seg in enumerate(polarization_segments):
                current_avg = seg["data"]["current(a)" if "current(a)" in seg["data"].columns else "current"].mean()

                if i == 0:
                    current_curve_segments.append(seg)
                    continue

                prev_seg = polarization_segments[i - 1]
                prev_avg = prev_seg["data"][
                    "current(a)" if "current(a)" in prev_seg["data"].columns else "current"].mean()

                if current_avg > prev_avg and len(current_curve_segments) >= 3:
                    if self.has_decreasing_pattern(current_curve_segments):
                        curve_counter += 1
                        ramp_down_segments = self.extract_ramp_down_only(current_curve_segments)

                        for curve_seg in ramp_down_segments:
                            curve_seg["pol_curve_number"] = curve_counter
                            curve_seg["pol_curve_group"] = curve_counter
                            curve_seg["is_activation"] = False
                            curve_seg["segments_in_curve"] = len(ramp_down_segments)

                        complete_curves.append(ramp_down_segments.copy())

                    current_curve_segments = [seg]
                else:
                    current_curve_segments.append(seg)

            if current_curve_segments and len(current_curve_segments) >= 3:
                if self.has_decreasing_pattern(current_curve_segments):
                    curve_counter += 1
                    ramp_down_segments = self.extract_ramp_down_only(current_curve_segments)

                    for curve_seg in ramp_down_segments:
                        curve_seg["pol_curve_number"] = curve_counter
                        curve_seg["pol_curve_group"] = curve_counter
                        curve_seg["is_activation"] = False
                        curve_seg["segments_in_curve"] = len(ramp_down_segments)

                    complete_curves.append(ramp_down_segments.copy())

            for segment in segments:
                if segment["type"] != "polarization":
                    segment["pol_curve_number"] = None
                    segment["pol_curve_group"] = None
                    segment["is_activation"] = False
                    segment["segments_in_curve"] = 0

            return segments

        except Exception as e:
            logger.error(f"Error adding polarization curve numbering: {e}")
            return segments

    def extract_ramp_down_only(self, curve_segments):
        """Extract only the ramp-down part of a polarization curve."""
        try:
            # Find peak current segment
            peak_current = 0
            peak_index = 0

            for idx, seg in enumerate(curve_segments):
                current_avg = seg["data"]["current(a)" if "current(a)" in seg["data"].columns else "current"].mean()
                if current_avg > peak_current:
                    peak_current = current_avg
                    peak_index = idx

            # Return segments from peak onwards (ramp-down only)
            ramp_down_segments = curve_segments[peak_index:]
            # print(
            #    f"DEBUG: Extracted ramp-down: peak at index {peak_index}, {len(ramp_down_segments)} segments from {len(curve_segments)} total")
            return ramp_down_segments

        except Exception as e:
            logger.error(f"Error extracting ramp-down segments: {e}")
            return curve_segments  # Fallback to original

    def has_decreasing_pattern(self, curve_segments):
        """
        Check if a group of segments shows a decreasing current pattern.

        Parameters:
        curve_segments (list): List of segments forming a potential curve

        Returns:
        bool: True if the curve shows decreasing current pattern
        """
        try:
            if len(curve_segments) < 3:
                return False

            # Get current from first and last segments
            first_current = curve_segments[0]["data"][
                "current(a)" if "current(a)" in curve_segments[0]["data"].columns else "current"].mean()
            last_current = curve_segments[-1]["data"][
                "current(a)" if "current(a)" in curve_segments[-1]["data"].columns else "current"].mean()

            # Should have significant current decrease (at least 10%)
            current_decrease = first_current - last_current
            decrease_percentage = current_decrease / first_current if first_current > 0 else 0

            # print(
            #    f"DEBUG: Curve validation - first: {first_current:.3f}A, last: {last_current:.3f}A, decrease: {decrease_percentage:.1%}")

            return decrease_percentage > 0.1  # At least 10% decrease

        except Exception as e:
            logger.error(f"Error validating curve pattern: {e}")
            return False

    def is_valid_polarization_curve(self, curve_segments):
        """
        Check if a group of segments forms a valid polarization curve.
        A valid curve should show a decreasing current pattern from start to end.

        Parameters:
        curve_segments (list): List of consecutive polarization segments

        Returns:
        bool: True if this forms a valid polarization curve
        """
        try:
            if len(curve_segments) < 3:  # Need at least 3 segments for a valid curve
                return False

            # Get current values from first and last segments
            first_segment = curve_segments[0]
            last_segment = curve_segments[-1]

            first_current = first_segment["data"][
                "current(a)" if "current(a)" in first_segment["data"].columns else "current"].mean()
            last_current = last_segment["data"][
                "current(a)" if "current(a)" in last_segment["data"].columns else "current"].mean()

            # Valid curve should have decreasing current (high to low)
            current_decrease = first_current - last_current

            # Should have at least 10% current decrease to be considered a valid curve
            if current_decrease > 0.1 * first_current:
                return True

            return False

        except Exception as e:
            logger.error(f"Error validating polarization curve: {e}")
            return False

    def get_polarization_curves_by_selection(self, segments, curve_selection, range_start=None, range_end=None):
        """
        Extracts specific polarization curves based on user selection.
        Now works with complete polarization curves (groups of segments).

        Parameters:
        segments (list): List of all segments.
        curve_selection (str): "last", "first", "all", or "specific range"
        range_start (int): Start of range for "specific range" selection
        range_end (int): End of range for "specific range" selection

        Returns:
        list: List of complete polarization curves (each curve is a list of segments).
        """
        try:
            # Group segments by curve number to get complete curves
            complete_curves = {}

            for segment in segments:
                if (segment["type"] == "polarization" and
                        segment.get("pol_curve_number") is not None):

                    curve_num = segment["pol_curve_number"]
                    if curve_num not in complete_curves:
                        complete_curves[curve_num] = []
                    complete_curves[curve_num].append(segment)

            if not complete_curves:
                print("No complete polarization curves found.")
                return []

            # Sort curves by number and ensure segments within each curve are sorted by time
            sorted_curve_numbers = sorted(complete_curves.keys())
            sorted_complete_curves = []

            for curve_num in sorted_curve_numbers:
                # Sort segments within each curve by start time
                curve_segments = sorted(complete_curves[curve_num], key=lambda x: x["start_time"])
                sorted_complete_curves.append(curve_segments)

            total_curves = len(sorted_complete_curves)
            print(f"Found {total_curves} complete polarization curves")

            # Select curves based on user choice
            if curve_selection == "last":
                return [sorted_complete_curves[-1]]

            elif curve_selection == "first":
                return [sorted_complete_curves[0]]

            elif curve_selection == "all":
                return sorted_complete_curves

            elif curve_selection == "specific range":
                if range_start is None or range_end is None:
                    logger.error("Range start and end must be specified for specific range selection")
                    return []

                # Validate range
                range_start = max(1, min(range_start, total_curves))
                range_end = max(range_start, min(range_end, total_curves))

                # Convert to 0-based indexing
                start_idx = range_start - 1
                end_idx = range_end

                selected_curves = sorted_complete_curves[start_idx:end_idx]
                print(
                    f"Selected polarization curves {range_start} to {range_end} ({len(selected_curves)} complete curves)")
                return selected_curves

            else:
                logger.error(f"Unknown curve selection: {curve_selection}")
                return []

        except Exception as e:
            logger.error(f"Error selecting polarization curves: {e}")
            return []

    def get_polarization_curve_info(self, segments):
        """
        Returns information about available complete polarization curves.

        Parameters:
        segments (list): List of all segments.

        Returns:
        dict: Information about complete polarization curves.
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

            curve_numbers = sorted(complete_curves.keys())
            curve_start_times = []

            for curve_num in curve_numbers:
                curve_segments = sorted(complete_curves[curve_num], key=lambda x: x["start_time"])
                curve_start_times.append(curve_segments[0]["start_time"])

            return {
                "total_count": len(curve_numbers),
                "curve_numbers": curve_numbers,
                "curve_start_times": curve_start_times,
                "has_curves": len(curve_numbers) > 0
            }

        except Exception as e:
            logger.error(f"Error getting polarization curve info: {e}")
            return {
                "total_count": 0,
                "curve_numbers": [],
                "curve_start_times": [],
                "has_curves": False
            }

    def convert_time_to_hours(self, df, time_column):
        """
        Converts a time column in hh:mm:ss format to hours as a float.
        Optimized using pd.to_timedelta() for better performance.
        """
        try:
            result = pd.to_timedelta(df[time_column]).dt.total_seconds() / 3600
            return result
        except Exception as e:
            logger.error(f"Error converting time to hours: {e}")