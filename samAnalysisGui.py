#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:48:23 2024

@author: danielvalmassei
"""

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)
import samAnalysis


class SamAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('SamAnalysis App')

        self.runNums_label = QLabel('runNums (comma-separated list):')
        self.runNums_edit = QLineEdit()

        self.events_label = QLabel('events:')
        self.events_edit = QLineEdit()
        self.events_edit.setText('10000')

        self.atten_label = QLabel('atten:')
        self.atten_edit = QLineEdit()
        self.atten_edit.setText('1.0')

        self.triggerThreshold_label = QLabel('triggerThreshold:')
        self.triggerThreshold_edit = QLineEdit()
        self.triggerThreshold_edit.setText('500')

        self.secondaryThreshold_label = QLabel('secondaryThreshold:')
        self.secondaryThreshold_edit = QLineEdit()
        self.secondaryThreshold_edit.setText('0')

        self.nBins_label = QLabel('nBins:')
        self.nBins_edit = QLineEdit()
        self.nBins_edit.setText('128')

        self.histEndpoint_label = QLabel('histEndpoint:')
        self.histEndpoint_edit = QLineEdit()
        self.histEndpoint_edit.setText('20.0')

        self.run_button = QPushButton('Run samAnalysis')
        self.run_button.clicked.connect(self.runAnalysis)

        layout = QVBoxLayout()
        layout.addWidget(self.runNums_label)
        layout.addWidget(self.runNums_edit)
        layout.addWidget(self.events_label)
        layout.addWidget(self.events_edit)
        layout.addWidget(self.atten_label)
        layout.addWidget(self.atten_edit)
        layout.addWidget(self.triggerThreshold_label)
        layout.addWidget(self.triggerThreshold_edit)
        layout.addWidget(self.secondaryThreshold_label)
        layout.addWidget(self.secondaryThreshold_edit)
        layout.addWidget(self.nBins_label)
        layout.addWidget(self.nBins_edit)
        layout.addWidget(self.histEndpoint_label)
        layout.addWidget(self.histEndpoint_edit)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def runAnalysis(self):
        runNums = self.runNums_edit.text().split(',')
        events = int(self.events_edit.text())
        atten = float(self.atten_edit.text())
        triggerThreshold = int(self.triggerThreshold_edit.text())
        secondaryThreshold = int(self.secondaryThreshold_edit.text())
        nBins = int(self.nBins_edit.text())
        histEndpoint = float(self.histEndpoint_edit.text())

        try:
            samAnalysis.main(runNums, events, atten, triggerThreshold, secondaryThreshold, nBins, histEndpoint)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    sam_app = SamAnalysisApp()
    sam_app.show()
    sys.exit(app.exec())
