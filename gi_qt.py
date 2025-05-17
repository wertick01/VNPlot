import sys
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QCheckBox, QComboBox, QColorDialog, QTableView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

class DataMergerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Data Merger App")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.select_folder_button = QPushButton("Select Folder")
        self.select_folder_button.clicked.connect(self.select_folder)
        layout.addWidget(self.select_folder_button)

        self.merge_button = QPushButton("Merge Data")
        self.merge_button.clicked.connect(self.merge_data)
        layout.addWidget(self.merge_button)

        self.table_view = QTableView()
        layout.addWidget(self.table_view)

        self.virus_serum_layout = QVBoxLayout()
        layout.addLayout(self.virus_serum_layout)

        self.plot_button = QPushButton("Plot Data")
        self.plot_button.clicked.connect(self.plot_data)
        layout.addWidget(self.plot_button)

        self.setLayout(layout)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path = folder_path

    def merge_data(self):
        if hasattr(self, 'folder_path'):
            files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.xlsx')]
            merged_data, virus_serum_combinations = self.mergeData(files)
            self.display_data(merged_data)
            self.display_virus_serum_combinations(virus_serum_combinations)

    def mergeData(self, files):
        # Пример метода mergeData
        dataframes = [pd.read_excel(file) for file in files]
        merged_data = pd.concat(dataframes, ignore_index=True)
        virus_serum_combinations = merged_data[['virus', 'serum']].drop_duplicates()
        return merged_data, virus_serum_combinations

    def display_data(self, data):
        model = QStandardItemModel(data.shape[0], data.shape[1])
        model.setHorizontalHeaderLabels(data.columns)

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                item = QStandardItem(str(data.iat[row, col]))
                model.setItem(row, col, item)

        self.table_view.setModel(model)

    def display_virus_serum_combinations(self, combinations):
        self.virus_serum_config = {}
        for i, row in combinations.iterrows():
            layout = QVBoxLayout()
            checkbox = QCheckBox(f"{row['virus']} - {row['serum']}")
            layout.addWidget(checkbox)

            marker_combo = QComboBox()
            marker_combo.addItems(['o', 'P', 'D', '^', 'X', 'v', '*', 'h'])
            layout.addWidget(marker_combo)

            color_button = QPushButton("Select Color")
            color_button.clicked.connect(lambda _, cb=checkbox, mc=marker_combo: self.select_color(cb, mc))
            layout.addWidget(color_button)

            self.virus_serum_layout.addLayout(layout)
            self.virus_serum_config[f"{row['virus']} - {row['serum']}"] = (checkbox, marker_combo, color_button)

    def select_color(self, checkbox, marker_combo):
        color = QColorDialog.getColor()
        if color.isValid():
            checkbox.setStyleSheet(f"background-color: {color.name()};")
            self.virus_serum_config[checkbox.text()][2].setStyleSheet(f"background-color: {color.name()};")

    def plot_data(self):
        markers_colors_viruses = {}
        for key, (checkbox, marker_combo, color_button) in self.virus_serum_config.items():
            if checkbox.isChecked():
                virus, serum = key.split(' - ')
                markers_colors_viruses[virus] = (color_button.palette().button().color().name(), marker_combo.currentText())

        # Пример метода plotData
        self.plotData2(self.merged_data, markers_colors_viruses)

    def plotData2(self, parsed_dataframe, markers_colors_dict):
        plt.figure(figsize=(18, 36))
        for virus, (color, marker) in markers_colors_dict.items():
            data = parsed_dataframe[parsed_dataframe['virus'] == virus]
            plt.plot(data['concentration'], data['fraction infectivity'], color=color, marker=marker, label=virus)
        plt.xlabel('[Ab] нг/мл')
        plt.ylabel('Количество вируса, %')
        plt.title('BsAb neutrolizing values')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataMergerApp()
    ex.show()
    sys.exit(app.exec_())
