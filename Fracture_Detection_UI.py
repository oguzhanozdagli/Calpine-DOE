import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFrame, QMessageBox, QPushButton
from PyQt5.QtCore import QTimer, Qt
from matplotlib.dates import DateFormatter
from datetime import timedelta


class FractureDetectionApp(QMainWindow):
    def __init__(self, edr_data):
        super().__init__()

        # Filter the data for Hole Depth between 4000 and 6000 feet
        self.data = edr_data[(edr_data['Hole Depth (feet)'] >= 4000) & (edr_data['Hole Depth (feet)'] <= 6000)].reset_index(drop=True)

        # Convert HH:MM:SS to Time (sec)
        self.data = self.convert_time_to_seconds(self.data)

        # Process the data for fracture detection categories
        self.data = self.calculate_rop_derivative(self.data)

        # Initialize PyQt window
        self.initUI()

        # Timer for real-time data simulation
        self.timer = QTimer()
        self.timer.setInterval(1000)  # 1 second interval
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        # Current index to simulate real-time data feed
        self.current_index = 0
        self.red_alert_start_time = None  # To track red alert durations

        # Time range options for the graph view
        self.time_range = None  # None means entire time view
        self.time_ranges = [None, timedelta(minutes=5), timedelta(minutes=10), timedelta(minutes=30)]

    def initUI(self):
        """Initialize the GUI layout and components."""
        self.setWindowTitle("Real-time Fracture Detection")

        # Create a main widget to hold everything
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # Create a vertical layout
        main_layout = QVBoxLayout(main_widget)

        # Create a horizontal layout for graphs
        graph_layout = QHBoxLayout()
        main_layout.addLayout(graph_layout)

        # Initialize matplotlib figures and canvases for each graph
        self.figure_rop, self.ax_rop = plt.subplots(figsize=(3, 15))
        self.figure_rop_derivative, self.ax_rop_derivative = plt.subplots(figsize=(3, 15))
        self.figure_wob, self.ax_wob = plt.subplots(figsize=(3, 15))
        self.figure_rpm, self.ax_rpm = plt.subplots(figsize=(3, 15))
        self.figure_depth_time, self.ax_depth_time = plt.subplots(figsize=(3, 15))

        # Set bigger fonts and thicker lines for all graphs
        plt.rcParams.update({'font.size': 24, 'lines.linewidth': 3})

        # Create FigureCanvas for each graph
        self.canvas_rop = FigureCanvas(self.figure_rop)
        self.canvas_rop_derivative = FigureCanvas(self.figure_rop_derivative)
        self.canvas_wob = FigureCanvas(self.figure_wob)
        self.canvas_rpm = FigureCanvas(self.figure_rpm)
        self.canvas_depth_time = FigureCanvas(self.figure_depth_time)

        # Add the canvases to the graph layout
        graph_layout.addWidget(self.canvas_rop)
        graph_layout.addWidget(self.canvas_rop_derivative)
        graph_layout.addWidget(self.canvas_wob)
        graph_layout.addWidget(self.canvas_rpm)
        graph_layout.addWidget(self.canvas_depth_time)

        # Create a color box to show the current fracture detection category for the ROP Derivative graph
        self.color_box = QFrame()
        self.color_box.setFrameShape(QFrame.Box)
        self.color_box.setMinimumHeight(50)
        self.color_box.setStyleSheet("background-color: green")  # Default to green

        # Add the color box below the ROP Derivative graph
        main_layout.addWidget(self.color_box)

        # Create a button to toggle between different time views
        self.toggle_button = QPushButton("Toggle Time View")
        self.toggle_button.clicked.connect(self.toggle_time_view)
        main_layout.addWidget(self.toggle_button)

    def convert_time_to_seconds(self, df):
        """Convert HH:MM:SS to seconds and then format it as DateTime."""
        df['DateTime'] = pd.to_datetime(df['YYYY/MM/DD'] + ' ' + df['HH:MM:SS'])
        df['Time (sec)'] = pd.to_timedelta(df['HH:MM:SS']).dt.total_seconds()
        return df

    def calculate_rop_derivative(self, filtered_data):
        red, orange, yellow = 4, 3.5, 3
        """
        Calculates ROP derivatives using np.gradient and assigns color categories based on thresholds.
        """
        filtered_data = filtered_data.copy()  # Make a copy to avoid changing the original data
        filtered_data['ROP Derivative'] = np.nan  # Create an empty column

        # Calculate ROP derivative using np.gradient for each group
        rop_values = filtered_data['Rate Of Penetration (ft_per_hr)'].values
        time_values = filtered_data['Time (sec)'].values
        if len(rop_values) > 1:
            rop_derivative = np.gradient(rop_values, time_values)
            filtered_data['ROP Derivative'] = rop_derivative

        # Determine color categories based on thresholds
        conditions = [
            (filtered_data['ROP Derivative'] > red),
            (filtered_data['ROP Derivative'] > orange) & (filtered_data['ROP Derivative'] <= red),
            (filtered_data['ROP Derivative'] > yellow) & (filtered_data['ROP Derivative'] <= orange)
        ]
        colors = ['Red', 'Orange', 'Yellow']  # Matching the number of conditions

        # Assign colors based on conditions
        filtered_data['ROP Derivative Color'] = np.select(conditions, colors, default='Green')

        return filtered_data

    def update_plot(self):
        red, orange, yellow = 4, 3.5, 3
        """Update the plot with real-time data."""
        if self.current_index >= len(self.data):
            self.timer.stop()
            return

        # Get the current slice of data up to current_index
        latest_data = self.data.iloc[:self.current_index]

        # Apply time filtering for last 5, 10, or 30 minutes view
        if self.time_range:
            min_time = latest_data['DateTime'].max() - self.time_range
            latest_data = latest_data[latest_data['DateTime'] >= min_time]

        # Clear previous plots
        self.ax_rop.clear()
        self.ax_rop_derivative.clear()
        self.ax_wob.clear()
        self.ax_rpm.clear()
        self.ax_depth_time.clear()

        # Plot ROP (Rate of Penetration) vs Time
        self.ax_rop.plot(latest_data['Rate Of Penetration (ft_per_hr)'], latest_data['DateTime'], color='blue')
        self.ax_rop.set_xlabel('Rate of Penetration (ft/hr)')
        self.ax_rop.set_ylabel('Time')
        self.ax_rop.set_title('ROP vs Time')
        self.ax_rop.invert_yaxis()  # Most recent data at the bottom

        # Plot ROP Derivative vs Time with vertical threshold lines
        self.ax_rop_derivative.plot(latest_data['ROP Derivative'], latest_data['DateTime'], color='green')
        self.ax_rop_derivative.axvline(x=red, color='red', linestyle='--', label='Red Threshold')
        self.ax_rop_derivative.axvline(x=orange, color='orange', linestyle='--', label='Red Threshold')
        self.ax_rop_derivative.axvline(x=yellow, color='yellow', linestyle='--', label='Red Threshold')
        self.ax_rop_derivative.set_xlabel('ROP Derivative')
        self.ax_rop_derivative.set_title('ROP Derivative vs Time')
        self.ax_rop_derivative.invert_yaxis()  # Most recent data at the bottom

        # Plot Weight on Bit (klbs) vs Time
        self.ax_wob.plot(latest_data['Weight on Bit (klbs)'], latest_data['DateTime'], color='purple')
        self.ax_wob.set_xlabel('WOB (klbs)')
        self.ax_wob.set_title('Weight on Bit vs Time')
        self.ax_wob.invert_yaxis()  # Most recent data at the bottom

        # Plot Rotary RPM vs Time
        self.ax_rpm.plot(latest_data['Rotary RPM (RPM)'], latest_data['DateTime'], color='orange')
        self.ax_rpm.set_xlabel('RPM')
        self.ax_rpm.set_title('Rotary RPM vs Time')
        self.ax_rpm.invert_yaxis()  # Most recent data at the bottom

        # Plot Hole Depth (feet) vs Time
        self.ax_depth_time.plot(latest_data['Hole Depth (feet)'], latest_data['DateTime'], color='brown')
        self.ax_depth_time.set_xlabel('Hole Depth (feet)')
        self.ax_depth_time.set_title('Hole Depth vs Time')
        self.ax_depth_time.invert_yaxis()  # Most recent data at the bottom

        # Format time axis as DateTime on all graphs
        date_format = DateFormatter("%H:%M:%S")
        self.ax_rop.yaxis.set_major_formatter(date_format)
        self.ax_rop_derivative.yaxis.set_major_formatter(date_format)
        self.ax_wob.yaxis.set_major_formatter(date_format)
        self.ax_rpm.yaxis.set_major_formatter(date_format)
        self.ax_depth_time.yaxis.set_major_formatter(date_format)

        # Redraw the canvases to show the updated plots
        self.canvas_rop.draw()
        self.canvas_rop_derivative.draw()
        self.canvas_wob.draw()
        self.canvas_rpm.draw()
        self.canvas_depth_time.draw()

        # Check for "Red" alerts (Fracture Detected)
        if not latest_data.empty:
            current_color = latest_data['ROP Derivative Color'].iloc[-1]  # Get the last detected color
            self.update_color_box(current_color)

            # Handle red alert pop-up if condition is met for more than 2 seconds
            if current_color == 'Red':
                if self.red_alert_start_time is None:
                    self.red_alert_start_time = latest_data['DateTime'].iloc[-1]
                elif (latest_data['DateTime'].iloc[-1] - self.red_alert_start_time).total_seconds() > 2:
                    self.trigger_red_alert(latest_data['ROP Derivative'].iloc[-1], 2)
            else:
                self.red_alert_start_time = None

        # Increment the index for the next update
        self.current_index += 1

    def update_color_box(self, color):
        """Update the color box based on current fracture detection level."""
        color_map = {
            'Green': 'green',
            'Yellow': 'yellow',
            'Orange': 'orange',
            'Red': 'red'
        }
        self.color_box.setStyleSheet(f"background-color: {color_map[color]};")

    def trigger_red_alert(self, rop_value, duration):
        """Trigger a pop-up alert for a red alert."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Fracture Detected")
        msg.setText(f"ROP DERIVATIVE > {rop_value:.2f} for {duration} seconds.\nDrilling with ROP recommended due to fractures.")
        msg.exec_()

    def toggle_time_view(self):
        """Toggle between the entire time view and the selected time views."""
        # Cycle through the available time ranges (None for all, 5, 10, 30 min)
        current_idx = self.time_ranges.index(self.time_range)
        self.time_range = self.time_ranges[(current_idx + 1) % len(self.time_ranges)]


# Main application entry point
def main():
    # Load real CSV data from file path
    file_path = r"C:\Users\trsch\Downloads\29301709.csv"
    edr_data = pd.read_csv(file_path)

    # Create the application and main window
    app = QApplication(sys.argv)
    window = FractureDetectionApp(edr_data)
    window.show()

    # Execute the application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
