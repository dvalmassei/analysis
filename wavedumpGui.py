#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:07:52 2024

@author: danielvalmassei
    assisted by ChatGPT 3.5
"""
import os
import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QGridLayout, QWidget, QTextEdit, QCheckBox

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wavedump Controller")
        self.setGeometry(100, 100, 600, 400)

        self.label_run_name = QLabel("Run Name:", self)
        self.label_run_name.setGeometry(50, 50, 100, 30)
        self.input_run_name = QLineEdit(self)
        self.input_run_name.setGeometry(150, 50, 200, 30)

        self.label_run_length = QLabel("Run Length (min):", self)
        self.label_run_length.setGeometry(50, 90, 100, 30)
        self.input_run_length = QLineEdit(self)
        self.input_run_length.setGeometry(150, 90, 200, 30)

        self.channel_checkboxes = []
        for i in range(8):
            checkbox = QCheckBox(f"Ch. {i}", self)
            self.channel_checkboxes.append(checkbox)

        self.start_button = QPushButton("Start Run", self)
        self.start_button.setGeometry(50, 170, 100, 50)
        self.start_button.clicked.connect(self.start_run)

        self.cancel_button = QPushButton("Cancel Run", self)
        self.cancel_button.setGeometry(200, 170, 100, 50)
        self.cancel_button.clicked.connect(self.cancel_run)
        self.cancel_button.setEnabled(False)

        self.output_text = QTextEdit(self)
        self.output_text.setGeometry(50, 230, 500, 150)
        self.output_text.setReadOnly(True)

        self.setup_layout()

        self.proc = None
        self.output_file = None

    def setup_layout(self):
        layout = QGridLayout()
        layout.addWidget(self.label_run_name, 0, 0)
        layout.addWidget(self.input_run_name, 0, 1)
        layout.addWidget(self.label_run_length, 1, 0)
        layout.addWidget(self.input_run_length, 1, 1)

        for i, checkbox in enumerate(self.channel_checkboxes):
            row = i % 4
            col = i // 4
            layout.addWidget(checkbox, row, col + 2)

        layout.addWidget(self.start_button, 4, 0, 1, 2)
        layout.addWidget(self.cancel_button, 4, 2, 1, 2)
        layout.addWidget(self.output_text, 5, 0, 1, 4)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def start_run(self):
        run_name = self.input_run_name.text()
        run_length = int(self.input_run_length.text())
        selected_channels = [i for i, checkbox in enumerate(self.channel_checkboxes) if checkbox.isChecked()]

        wavedump_path = os.path.expanduser("~/CAEN/wavedump-3.10.6/")
        wavedump_cmd = os.path.join(wavedump_path, "wavedump")
        config_file = os.path.join(wavedump_path, " WaveDumpConfig_X742.txt")
        output_folder = os.path.expanduser(f"~/data/{run_name}/")
        output_file_path = os.path.join(output_folder, "output.log")

        try:
            os.makedirs(output_folder, exist_ok=True)
            self.output_file = open(output_file_path, "w")
            self.output_text.append("Starting Wavedump...\n")
            self.proc = subprocess.Popen([wavedump_cmd, config_file], stdout=self.output_file, stderr=subprocess.STDOUT, text=True)
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.input_run_name.setEnabled(False)
            self.input_run_length.setEnabled(False)
            for checkbox in self.channel_checkboxes:
                checkbox.setEnabled(False)

            subprocess.run(["sleep 1"])

            # Start acquisition
            self.output_text.append("Starting acquisition...\n")
            subprocess.run(["echo", "s"], cwd=wavedump_path)
            subprocess.run(["sleep 1"])


            # Continuously write output to files
            self.output_text.append("Continuously writing output...\n")
            subprocess.run(["echo", "W"], cwd=wavedump_path)

            # Wait for the specified run length
            self.output_text.append(f"Waiting for {run_length} minutes...\n")
            subprocess.run(["sleep", str(run_length * 60)])

            # Stop acquisition
            self.output_text.append("Stopping acquisition...\n")
            subprocess.run(["echo", "q"], cwd=wavedump_path)
            subprocess.run(["sleep 1"])
            subprocess.run(["echo", "w"], cwd=wavedump_path)
            
            # Copy output files for selected channels to the specified folder
            self.output_text.append("Copying output files...\n")
            for channel in selected_channels:
                subprocess.run(["cp", os.path.join(wavedump_path, f"wave_{channel}.txt"), output_folder])

            self.output_text.append("Run completed.\n")
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.input_run_name.setEnabled(True)
            self.input_run_length.setEnabled(True)
            for checkbox in self.channel_checkboxes:
                checkbox.setEnabled(True)

        except Exception as e:
            self.output_text.append(f"Error: {e}\n")
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.input_run_name.setEnabled(True)
            self.input_run_length.setEnabled(True)
            for checkbox in self.channel_checkboxes:
                checkbox.setEnabled(True)
        finally:
            if self.output_file:
                self.output_file.close()

    def cancel_run(self):
        if self.proc is not None and self.proc.poll() is None:
            self.output_text.append("Cancelling run...\n")
            subprocess.run(["echo", "s"], cwd=os.path.expanduser("~/CAEN/wavedump-3.10.6/"))
            subprocess.run(["echo", "q"], cwd=os.path.expanduser("~/CAEN/wavedump-3.10.6/"))
            self.proc.terminate()
            self.output_text.append("Run cancelled.\n")
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.input_run_name.setEnabled(True)
            self.input_run_length.setEnabled(True)
            for checkbox in self.channel_checkboxes:
                checkbox.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


