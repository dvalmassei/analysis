#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:07:52 2024

@author: danielvalmassei
"""

import os
import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wavedump Controller")
        self.setGeometry(100, 100, 400, 200)

        self.label_run_name = QLabel("Run Name:", self)
        self.label_run_name.setGeometry(50, 50, 100, 30)
        self.input_run_name = QLineEdit(self)
        self.input_run_name.setGeometry(150, 50, 200, 30)

        self.label_run_length = QLabel("Run Length (min):", self)
        self.label_run_length.setGeometry(50, 95, 100, 30)
        self.input_run_length = QLineEdit(self)
        self.input_run_length.setGeometry(150, 95, 200, 30)

        self.start_button = QPushButton("Start Run", self)
        self.start_button.setGeometry(50, 130, 300, 50)
        self.start_button.clicked.connect(self.start_run)

    def start_run(self):
        run_name = self.input_run_name.text()
        run_length = int(self.input_run_length.text())
        
        wavedump_path = os.path.expanduser("~/CAEN/wavedump-3.10.6/")
        wavedump_cmd = os.path.join(wavedump_path, "wavedump")
        config_file = os.path.join(wavedump_path, "WaveDumpConfig_X742.txt")
        output_folder = os.path.expanduser(f"~/data/{run_name}/")
        
        try:
            os.makedirs(output_folder, exist_ok=True)
            subprocess.Popen([wavedump_cmd, config_file])

            # Start acquisition
            subprocess.run(["echo", "s"], cwd=wavedump_path)

            # Continuously write output to files
            subprocess.run(["echo", "W"], cwd=wavedump_path)

            # Wait for the specified run length
            subprocess.run(["sleep", str(run_length * 60)])

            # Stop acquisition
            subprocess.run(["echo", "q"], cwd=wavedump_path)
            subprocess.run(["echo", "w"], cwd=wavedump_path)
            
            # Move output files to the specified folder
            subprocess.run(["cp", os.path.join(wavedump_path, "TR_0_0.txt"), output_folder])
            subprocess.run(["cp", os.path.join(wavedump_path, "wave_0.txt"), output_folder])
            subprocess.run(["cp", os.path.join(wavedump_path, "wave_1.txt"), output_folder])

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
