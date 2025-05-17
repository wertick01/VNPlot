import os
import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QSizePolicy, QDoubleSpinBox, QVBoxLayout, QTableView, QCheckBox, QComboBox, QGroupBox, QColorDialog, QHBoxLayout, QFormLayout, QLineEdit, QScrollArea, QDialog, QDialogButtonBox, QSlider, QSpinBox, QGridLayout, QDesktopWidget
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt5.QtCore import Qt, QSortFilterProxyModel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from openData import MergeData
from plotData import PlotData

class PlotWindow(QDialog):
    def __init__(self, fig, plot_data_func, virus_config, serum_virus_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot")

        # Получаем размеры экрана
        screen_geometry = QDesktopWidget().screenGeometry()
        width = screen_geometry.width()
        height = screen_geometry.height()

        # Устанавливаем размеры окна на 90% от размеров экрана
        self.setGeometry(int(width * 0.05), int(height * 0.05), int(width * 0.9), int(height * 0.9))

        self.plot_data_func = plot_data_func
        self.fig = fig
        self.serum_virus_dict = serum_virus_dict
        self.virus_config = virus_config
        self.display = {serum: {virus: True for virus in viruses} for serum, viruses in serum_virus_dict.items()}  # Инициализация словаря display

        main_layout = QVBoxLayout()

        # Верхняя часть окна
        upper_layout = QHBoxLayout()

        # Левая часть верхней области для статических параметров
        self.params_scroll_area = QScrollArea()
        self.params_scroll_area.setWidgetResizable(True)
        self.params_scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.params_scroll_area.setFixedWidth(int(width * 0.15))  # 10% ширины окна

        self.params_container = QWidget()
        self.params_layout = QGridLayout()
        self.params_container.setLayout(self.params_layout)
        self.params_scroll_area.setWidget(self.params_container)
        upper_layout.addWidget(self.params_scroll_area)

        self.marker_size_px_slider = self.create_slider(0, 40, 10)
        self.params_layout.addWidget(QLabel("Marker Size (px):"), 0, 0)
        self.params_layout.addWidget(self.marker_size_px_slider, 0, 1)

        self.marker_line_px_slider = self.create_slider(0, 20, 5)
        self.params_layout.addWidget(QLabel("Marker Line (px):"), 1, 0)
        self.params_layout.addWidget(self.marker_line_px_slider, 1, 1)

        self.xlabel_title_fontsize_spinbox = self.create_spinbox(10, 50, 20)
        self.params_layout.addWidget(QLabel("X Label Title Fontsize:"), 2, 0)
        self.params_layout.addWidget(self.xlabel_title_fontsize_spinbox, 2, 1)

        self.ylabel_title_fontsize_spinbox = self.create_spinbox(10, 50, 20)
        self.params_layout.addWidget(QLabel("Y Label Title Fontsize:"), 3, 0)
        self.params_layout.addWidget(self.ylabel_title_fontsize_spinbox, 3, 1)

        self.xlabel_fontsize_spinbox = self.create_spinbox(10, 50, 20)
        self.params_layout.addWidget(QLabel("X Label Fontsize:"), 4, 0)
        self.params_layout.addWidget(self.xlabel_fontsize_spinbox, 4, 1)

        self.ylabel_fontsize_spinbox = self.create_spinbox(10, 50, 20)
        self.params_layout.addWidget(QLabel("Y Label Fontsize:"), 5, 0)
        self.params_layout.addWidget(self.ylabel_fontsize_spinbox, 5, 1)

        self.subplot_title_fontsize_spinbox = self.create_spinbox(10, 50, 30)
        self.params_layout.addWidget(QLabel("Subplot Title Fontsize:"), 6, 0)
        self.params_layout.addWidget(self.subplot_title_fontsize_spinbox, 6, 1)

        self.title_fontsize_spinbox = self.create_spinbox(10, 50, 35)
        self.params_layout.addWidget(QLabel("Title Fontsize:"), 7, 0)
        self.params_layout.addWidget(self.title_fontsize_spinbox, 7, 1)

        self.legend_fontsize_spinbox = self.create_spinbox(10, 50, 25)
        self.params_layout.addWidget(QLabel("Legend Fontsize:"), 8, 0)
        self.params_layout.addWidget(self.legend_fontsize_spinbox, 8, 1)

        self.legend_ncols_spinbox = self.create_spinbox(1, 10, 5)
        self.params_layout.addWidget(QLabel("Legend Columns:"), 9, 0)
        self.params_layout.addWidget(self.legend_ncols_spinbox, 9, 1)

        self.ncol_spinbox = self.create_spinbox(1, 10, 4)
        self.params_layout.addWidget(QLabel("Subplot Columns:"), 10, 0)
        self.params_layout.addWidget(self.ncol_spinbox, 10, 1)

        self.average_only_checkbox = self.create_checkbox(True)
        self.params_layout.addWidget(QLabel("Average Only:"), 11, 0)
        self.params_layout.addWidget(self.average_only_checkbox, 11, 1)

        self.no_average_checkbox = self.create_checkbox(False)
        self.params_layout.addWidget(QLabel("No Average:"), 12, 0)
        self.params_layout.addWidget(self.no_average_checkbox, 12, 1)

        self.scale_checkbox = self.create_checkbox(False)
        self.params_layout.addWidget(QLabel("Scale"), 13, 0)
        self.params_layout.addWidget(self.scale_checkbox, 13, 1)

        self.reverse_checkbox = self.create_checkbox(False)
        self.params_layout.addWidget(QLabel("Reverse"), 14, 0)
        self.params_layout.addWidget(self.reverse_checkbox, 14, 1)

        self.font_lineedit = QLineEdit("Arial")
        self.params_layout.addWidget(QLabel("Font:"), 15, 0)
        self.params_layout.addWidget(self.font_lineedit, 15, 1)

        self.fheight_spinbox = self.create_spinbox(1, 100, 36)
        self.params_layout.addWidget(QLabel("Figure Height (in):"), 16, 0)
        self.params_layout.addWidget(self.fheight_spinbox, 16, 1)

        self.fwidth_spinbox = self.create_spinbox(1, 100, 18)
        self.params_layout.addWidget(QLabel("Figure Width (in):"), 17, 0)
        self.params_layout.addWidget(self.fwidth_spinbox, 17, 1)

        self.marker_line_width_spinbox = self.create_spinbox(1, 50, 1)
        self.params_layout.addWidget(QLabel("Marker Line Width:"), 18, 0)
        self.params_layout.addWidget(self.marker_line_width_spinbox, 18, 1)

        self.xaxis_coords_x_spinbox = self.create_spinbox(-100, 100, 0.5, is_float=True)
        self.params_layout.addWidget(QLabel("X Axis Coords X:"), 19, 0)
        self.params_layout.addWidget(self.xaxis_coords_x_spinbox, 19, 1)

        self.xaxis_coords_y_spinbox = self.create_spinbox(-100, 100, -0.13, is_float=True)
        self.params_layout.addWidget(QLabel("X Axis Coords Y:"), 20, 0)
        self.params_layout.addWidget(self.xaxis_coords_y_spinbox, 20, 1)

        self.yaxis_coords_x_spinbox = self.create_spinbox(-100, 100, -0.09, is_float=True)
        self.params_layout.addWidget(QLabel("Y Axis Coords X:"), 21, 0)
        self.params_layout.addWidget(self.yaxis_coords_x_spinbox, 21, 1)

        self.yaxis_coords_y_spinbox = self.create_spinbox(-100, 100, 0.5, is_float=True)
        self.params_layout.addWidget(QLabel("Y Axis Coords Y:"), 22, 0)
        self.params_layout.addWidget(self.yaxis_coords_y_spinbox, 22, 1)

        self.xlabel_pad_spinbox = self.create_spinbox(0, 100, 35)
        self.params_layout.addWidget(QLabel("X Label Pad:"), 23, 0)
        self.params_layout.addWidget(self.xlabel_pad_spinbox, 23, 1)

        self.ylabel_pad_spinbox = self.create_spinbox(0, 100, 35)
        self.params_layout.addWidget(QLabel("Y Label Pad:"), 24, 0)
        self.params_layout.addWidget(self.ylabel_pad_spinbox, 24, 1)

        self.title_padding_spinbox = self.create_spinbox(0, 100, 0.92, is_float=True)
        self.params_layout.addWidget(QLabel("Title Padding:"), 25, 0)
        self.params_layout.addWidget(self.title_padding_spinbox, 25, 1)

        self.bottom_padding_spinbox = self.create_spinbox(0, 100, 0.16, is_float=True)
        self.params_layout.addWidget(QLabel("Bottom Padding:"), 26, 0)
        self.params_layout.addWidget(self.bottom_padding_spinbox, 26, 1)

        self.legend_borderaxespad_spinbox = self.create_spinbox(0, 100, 0, is_float=True)
        self.params_layout.addWidget(QLabel("Legend Borderaxespad:"), 27, 0)
        self.params_layout.addWidget(self.legend_borderaxespad_spinbox, 27, 1)

        self.wspace_spinbox = self.create_spinbox(0, 100, 0.17, is_float=True)
        self.params_layout.addWidget(QLabel("Wspace:"), 28, 0)
        self.params_layout.addWidget(self.wspace_spinbox, 28, 1)

        self.hspace_spinbox = self.create_spinbox(0, 100, 0.39, is_float=True)
        self.params_layout.addWidget(QLabel("Hspace:"), 29, 0)
        self.params_layout.addWidget(self.hspace_spinbox, 29, 1)

        self.xtitle_lineedit = QLineEdit("[Ab] нг/мл")
        self.params_layout.addWidget(QLabel("X Title:"), 30, 0)
        self.params_layout.addWidget(self.xtitle_lineedit, 30, 1)

        self.ytitle_lineedit = QLineEdit("Уровень нейтрализации, %")
        self.params_layout.addWidget(QLabel("Y Title:"), 31, 0)
        self.params_layout.addWidget(self.ytitle_lineedit, 31, 1)

        self.infectivity_or_neutralized_combo = QComboBox()
        self.infectivity_or_neutralized_combo.addItems(["infectivity", "neutralized"])
        self.params_layout.addWidget(QLabel("Infectivity or neutralized"), 32, 0)
        self.params_layout.addWidget(self.infectivity_or_neutralized_combo, 32, 1)

        # Правая часть верхней области для отображения графика
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Создаем контейнер для QLabel
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        upper_layout.addWidget(self.scroll_area)

        main_layout.addLayout(upper_layout)

        # Нижняя часть окна
        lower_layout = QVBoxLayout()

        self.dynamic_params_scroll_area = QScrollArea()
        self.dynamic_params_scroll_area.setWidgetResizable(True)
        self.dynamic_params_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dynamic_params_scroll_area.setFixedHeight(int(height * 0.15))  # 10% высоты окна

        self.dynamic_params_container = QWidget()
        self.dynamic_params_layout = QVBoxLayout()
        self.dynamic_params_container.setLayout(self.dynamic_params_layout)
        self.dynamic_params_scroll_area.setWidget(self.dynamic_params_container)

        self.display_params_groupbox = QGroupBox("Display params")
        self.display_params_layout = QGridLayout()

        # Создаем матрицу галочек
        max_viruses = max(len(viruses) for viruses in self.serum_virus_dict.values())
        for col, (serum, viruses) in enumerate(self.serum_virus_dict.items()):
            self.display_params_layout.addWidget(QLabel(serum), 0, col * 2 + 1, 1, 2)
            for row, virus in enumerate(viruses.keys()):
                self.display_params_layout.addWidget(QLabel(virus), row + 1, col * 2)
                checkbox = self.create_checkbox(True)
                checkbox.stateChanged.connect(lambda state, serum=serum, virus=virus: self.update_checkbox_state(serum, virus, state))
                self.display_params_layout.addWidget(checkbox, row + 1, col * 2 + 1)

        self.display_params_groupbox.setLayout(self.display_params_layout)
        self.dynamic_params_layout.addWidget(self.display_params_groupbox)

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_plot)
        self.dynamic_params_layout.addWidget(self.update_button)

        self.save_button = QPushButton("Сохранить")
        self.save_button.clicked.connect(self.save_plot)
        self.dynamic_params_layout.addWidget(self.save_button)

        lower_layout.addWidget(self.dynamic_params_scroll_area)

        main_layout.addLayout(lower_layout)

        self.setLayout(main_layout)

        self.adjust_font_sizes()
        self.update_plot()

    def adjust_font_sizes(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        width = screen_geometry.width()
        height = screen_geometry.height()

        base_font_size = 12
        scale_factor = min(width / 1920, height / 1080) // 2

        font_size = int(base_font_size * scale_factor)

        font = self.font()
        font.setPointSize(font_size)
        self.setFont(font)

        for widget in self.findChildren(QWidget):
            widget.setFont(font)

    def create_slider(self, min_val, max_val, default_val):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        return slider

    def create_spinbox(self, min_val, max_val, default_val, is_float=False):
        if is_float:
            spinbox = QDoubleSpinBox()
            spinbox.setDecimals(2)  # Установите количество десятичных знаков
        else:
            spinbox = QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default_val)
        return spinbox

    def create_checkbox(self, default_val):
        checkbox = QCheckBox()
        checkbox.setChecked(default_val)
        return checkbox

    def update_checkbox_state(self, serum, virus, state):
        self.display[serum][virus] = state == Qt.Checked
        print(f"{serum} - {virus}: {'Checked' if state == Qt.Checked else 'Unchecked'}")
        self.update_plot()

    def update_plot(self):
        params = {
            'marker_size_px': self.marker_size_px_slider.value(),
            'marker_line_px': self.marker_line_px_slider.value(),
            'xlabel_title_fontsize': self.xlabel_title_fontsize_spinbox.value(),
            'ylabel_title_fontsize': self.ylabel_title_fontsize_spinbox.value(),
            'xlabel_fontsize': self.xlabel_fontsize_spinbox.value(),
            'ylabel_fontsize': self.ylabel_fontsize_spinbox.value(),
            'subplot_title_fontsize': self.subplot_title_fontsize_spinbox.value(),
            'title_fontsize': self.title_fontsize_spinbox.value(),
            'legend_fontsize': self.legend_fontsize_spinbox.value(),
            'legend_ncols': self.legend_ncols_spinbox.value(),
            'ncol': self.ncol_spinbox.value(),
            'average_only': self.average_only_checkbox.isChecked(),
            'no_average': self.no_average_checkbox.isChecked(),
            'scale': self.scale_checkbox.isChecked(),
            'reverse': self.reverse_checkbox.isChecked(),
            'display': self.display,
            'font': self.font_lineedit.text(),
            'fheight': self.fheight_spinbox.value(),
            'fwidth': self.fwidth_spinbox.value(),
            'marker_line_width': self.marker_line_width_spinbox.value(),
            'xaxis_coords': (self.xaxis_coords_x_spinbox.value(), self.xaxis_coords_y_spinbox.value()),
            'yaxis_coords': (self.yaxis_coords_x_spinbox.value(), self.yaxis_coords_y_spinbox.value()),
            'xlabel_pad': self.xlabel_pad_spinbox.value(),
            'ylabel_pad': self.ylabel_pad_spinbox.value(),
            'title_padding': self.title_padding_spinbox.value(),
            'bottom_padding': self.bottom_padding_spinbox.value(),
            'legend_borderaxespad': self.legend_borderaxespad_spinbox.value(),
            'wspace': self.wspace_spinbox.value(),
            'hspace': self.hspace_spinbox.value(),
            'xtitle': self.xtitle_lineedit.text(),
            'ytitle': self.ytitle_lineedit.text(),
            'infectivity_or_neutralized': self.infectivity_or_neutralized_combo.currentText(),
            'serum_virus_dict': self.display,
        }
        self.fig, _, _ = self.plot_data_func(params)
        self.fig.tight_layout()  # Автоматическое подбирание размеров графика

        # Сохраняем график в файл PNG
        self.fig.savefig("plot.png")

        # Загружаем изображение в QLabel
        pixmap = QPixmap("plot.png")
        self.image_label.setPixmap(pixmap.scaled(
            int(self.scroll_area.width() * 0.95),
            self.scroll_area.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def save_plot(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "Images (*.png *.jpg *.tif);;All Files (*)", options=options)
        if fileName:
            self.fig.savefig(fileName)

    def update_zoom(self, value):
        scale_factor = value / 100.0
        pixmap = self.image_label.pixmap()
        if pixmap:
            new_width = int(pixmap.width() * scale_factor)
            new_height = int(pixmap.height() * scale_factor)
            self.image_label.setPixmap(pixmap.scaled(
                new_width,
                new_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

class FileSelectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("File Selector App")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()

        # Создаем горизонтальный layout для кнопок "Select Files" и "Add More Files"
        buttons_layout = QHBoxLayout()

        self.select_files_button = QPushButton("Select Files")
        self.select_files_button.clicked.connect(self.select_files)
        buttons_layout.addWidget(self.select_files_button)

        self.add_files_button = QPushButton("Add More Files")
        self.add_files_button.clicked.connect(self.add_files)
        buttons_layout.addWidget(self.add_files_button)

        layout.addLayout(buttons_layout)

        self.process_files_button = QPushButton("Process Files")
        self.process_files_button.clicked.connect(self.process_files)
        layout.addWidget(self.process_files_button)

        self.table_view = QTableView()
        layout.addWidget(self.table_view)

        self.filter_lineedit = QLineEdit()
        self.filter_lineedit.setPlaceholderText("Filter...")
        self.filter_lineedit.textChanged.connect(self.filter_data)
        layout.addWidget(self.filter_lineedit)

        self.virus_config_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setLayout(self.virus_config_layout)
        scroll_area.setWidget(scroll_content)
        scroll_area.setFixedHeight(200)
        layout.addWidget(scroll_area)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.save_config)
        layout.addWidget(self.ok_button)

        self.serum_config_layout = QVBoxLayout()
        self.serum_scroll_area = QScrollArea()
        self.serum_scroll_area.setWidgetResizable(True)
        self.serum_scroll_content = QWidget()
        self.serum_scroll_content.setLayout(self.serum_config_layout)
        self.serum_scroll_area.setWidget(self.serum_scroll_content)
        self.serum_scroll_area.setFixedHeight(200)
        self.serum_scroll_area.setVisible(False)
        layout.addWidget(self.serum_scroll_area)

        self.plot_button = QPushButton("Построение графиков")
        self.plot_button.clicked.connect(self.plot_data)
        layout.addWidget(self.plot_button)

        self.setLayout(layout)

        self.adjust_font_sizes()

    def adjust_font_sizes(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        width = screen_geometry.width()
        height = screen_geometry.height()

        base_font_size = 12
        scale_factor = min(width / 1920, height / 1080) // 2

        font_size = int(base_font_size * scale_factor)

        font = self.font()
        font.setPointSize(font_size)
        self.setFont(font)

        for widget in self.findChildren(QWidget):
            widget.setFont(font)

    def select_files(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if files:
            for file in files:
                self.add_file_to_list(file)

    def add_files(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Add More Files", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if files:
            for file in files:
                self.add_file_to_list(file)

    def add_file_to_list(self, file):
        if not hasattr(self, 'files'):
            self.files = {}
        self.files[os.path.basename(file)] = file

    def process_files(self):
        if hasattr(self, 'files') and self.files:
            dt = MergeData(files=self.files)
            dt.openDataFrames(files=self.files)
            dt.parseDataColumns()
            dt.parseFileNames()
            new_virus_names = dt.process_dict(data=dt.names)
            merged_data = dt.parseMultipleDfs(dt.dataframes)
            df = dt.processViruses(merged_data, new_virus_names)
            dt.generate_process_combinations(df)
            dt.process_replicates(df)
            self.serum_virus_dict = dt.create_serum_virus_dict(df)
            self.data = df  # Сохраняем данные в атрибуте data
            self.display_data(df)
            self.display_virus_config(df)
        else:
            print("No files selected.")

    def display_data(self, data):
        model = QStandardItemModel(data.shape[0], data.shape[1])
        model.setHorizontalHeaderLabels(data.columns)

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                item = QStandardItem(str(data.iat[row, col]))
                model.setItem(row, col, item)

        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(model)
        self.table_view.setModel(self.proxy_model)
        self.table_view.setSortingEnabled(True)

    def filter_data(self, text):
        self.proxy_model.setFilterRegExp(text)
        self.proxy_model.setFilterKeyColumn(-1)

    def display_virus_config(self, data):
        unique_viruses = data['virus'].unique()
        self.virus_config = {}

        markers = ['o', 'P', 'D', '^', 'X', 'v', '*', 'h']

        for virus in unique_viruses:
            layout = QHBoxLayout()

            label = QLabel(virus)
            layout.addWidget(label)

            checkbox = QCheckBox()
            layout.addWidget(checkbox)

            marker_combo = QComboBox()
            marker_combo.addItems(markers)
            layout.addWidget(marker_combo)

            color_label = QLabel()
            color_label.setFixedSize(20, 20)
            color_label.setStyleSheet("background-color: white; border: 1px solid black;")
            color_label.mousePressEvent = lambda event, virus=virus: self.select_color(virus)
            layout.addWidget(color_label)

            color_line_edit = QLineEdit()
            color_line_edit.setFixedWidth(100)
            color_line_edit.textChanged.connect(lambda text, virus=virus: self.update_color(virus, text))
            layout.addWidget(color_line_edit)

            self.virus_config_layout.addLayout(layout)
            self.virus_config[virus] = (checkbox, marker_combo, color_label, color_line_edit)

    def select_color(self, virus):
        color = QColorDialog.getColor()
        if color.isValid():
            checkbox, marker_combo, color_label, color_line_edit = self.virus_config[virus]
            color_label.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
            color_line_edit.setText(color.name())

    def update_color(self, virus, text):
        checkbox, marker_combo, color_label, color_line_edit = self.virus_config[virus]
        color_label.setStyleSheet(f"background-color: {text}; border: 1px solid black;")

    def generate_config(self):
        markers_colors_kih = {}
        for virus, (checkbox, marker_combo, color_label, color_line_edit) in self.virus_config.items():
            if checkbox.isChecked():
                color = color_line_edit.text()
                marker = marker_combo.currentText()
                markers_colors_kih[virus] = (color, marker)

        print("Generated Config:")
        print(markers_colors_kih)
        return markers_colors_kih

    def save_config(self):
        config = self.generate_config()
        self.display_serum_config(config)
        self.serum_scroll_area.setVisible(True)

    def display_serum_config(self, config):
        unique_serums = self.data['serum'].unique()
        self.serum_config = {}

        for serum in unique_serums:
            if any(serum in self.data[self.data['virus'] == virus]['serum'].values for virus in config.keys()):
                layout = QHBoxLayout()

                label = QLabel(serum)
                layout.addWidget(label)

                checkbox = QCheckBox()
                layout.addWidget(checkbox)

                self.serum_config_layout.addLayout(layout)
                self.serum_config[serum] = checkbox

    def save_serum_config(self):
        selected_serums = [serum for serum, checkbox in self.serum_config.items() if checkbox.isChecked()]
        print("Selected Serums:")
        print(selected_serums)

    def plot_data(self):
        virus_config = self.generate_config()
        serum_config = [serum for serum, checkbox in self.serum_config.items() if checkbox.isChecked()]

        filtered_data = self.data[
            (self.data['virus'].isin(virus_config.keys())) &
            (self.data['serum'].isin(serum_config))
        ]

        plotData = PlotData(params=dict(
            font="Arial",  # Шрифт для текста на графиках
            markers_colors_dict=virus_config,  # Словарь, содержащий маркеры и цвета для каждого вируса
            title='BsAb neutrolizing values',  # Заголовок графика
            fheight=36,  # Высота фигуры в дюймах
            fwidth=18,  # Ширина фигуры в дюймах
            marker_size=5.,
            marker_size_px=10,  # Размер маркеров в пикселях
            marker_line_px=5,  # Ширина линии маркеров в пикселях
            marker_line_width=1.,
            xaxis_coords=(0.5, -0.13),  # Координаты для размещения подписи оси X
            yaxis_coords=(-0.09, 0.5),  # Координаты для размещения подписи оси Y
            xlabel_title_fontsize=20,  # Размер шрифта для подписи оси X
            ylabel_title_fontsize=20,  # Размер шрифта для подписи оси Y
            xlabel_fontsize=20,  # Размер шрифта для меток на оси X
            ylabel_fontsize=20,  # Размер шрифта для меток на оси Y
            xlabel_pad=35,  # Отступ подписи оси X от оси
            ylabel_pad=35,  # Отступ подписи оси Y от оси
            subplot_title_fontsize=30,  # Размер шрифта для заголовков подграфиков
            title_fontsize=35,  # Размер шрифта для общего заголовка графика
            title_padding=0.92,  # Отступ общего заголовка от верхней границы фигуры
            bottom_padding=0.16,
            legend_fontsize=25,  # Размер шрифта для легенды
            legend_borderaxespad=0,  # Отступ легенды от границы фигуры
            legend_ncols=5,  # Количество столбцов в легенде
            xtitle='[Ab] нг/мл',  # Подпись оси X
            ytitle='Уровень нейтрализации, %',  # Подпись оси Y
            wspace=0.17, # расстояние между графиками по горизонтали в масштабе
            hspace=0.39, # расстояние между графиками по вертикали в масштабе
            ncol=4,  # параметр для управления количеством столбцов
            fix_lims={
                'ymin': -5,
                'ymax': 110,
            },
            average_only=True,
            no_average=False,
            scale=True,
            reverse=True,
            infectivity_or_neutralized='infectivity', # or 'neutralized'
            serum_virus_dict=self.serum_virus_dict,
        ))

        fig, ax, fit_params = plotData.plotData(
            parsed_dataframe=filtered_data.dropna(),  # DataFrame с данными для построения графиков
        )

        plot_window = PlotWindow(fig, self.plot_data_func, virus_config, self.serum_virus_dict)
        plot_window.exec_()

    def plot_data_func(self, params):
        virus_config = self.generate_config()
        serum_config = [serum for serum, checkbox in self.serum_config.items() if checkbox.isChecked()]

        filtered_data = self.data[
            (self.data['virus'].isin(virus_config.keys())) &
            (self.data['serum'].isin(serum_config))
        ]

        plotData = PlotData(params=dict(
            font="Arial",  # Шрифт для текста на графиках
            markers_colors_dict=virus_config,  # Словарь, содержащий маркеры и цвета для каждого вируса
            title='BsAb',  # Заголовок графика
            fheight=params['fheight'],  # Высота фигуры в дюймах
            fwidth=params['fwidth'],  # Ширина фигуры в дюймах
            marker_size=5.,
            marker_size_px=params['marker_size_px'],  # Размер маркеров в пикселях
            marker_line_px=params['marker_line_px'],  # Ширина линии маркеров в пикселях
            marker_line_width=params['marker_line_width'],
            xaxis_coords=params['xaxis_coords'],  # Координаты для размещения подписи оси X
            yaxis_coords=params['yaxis_coords'],  # Координаты для размещения подписи оси Y
            xlabel_title_fontsize=params['xlabel_title_fontsize'],  # Размер шрифта для подписи оси X
            ylabel_title_fontsize=params['ylabel_title_fontsize'],  # Размер шрифта для подписи оси Y
            xlabel_fontsize=params['xlabel_fontsize'],  # Размер шрифта для меток на оси X
            ylabel_fontsize=params['ylabel_fontsize'],  # Размер шрифта для меток на оси Y
            xlabel_pad=params['xlabel_pad'],  # Отступ подписи оси X от оси
            ylabel_pad=params['ylabel_pad'],  # Отступ подписи оси Y от оси
            subplot_title_fontsize=params['subplot_title_fontsize'],  # Размер шрифта для заголовков подграфиков
            title_fontsize=params['title_fontsize'],  # Размер шрифта для общего заголовка графика
            title_padding=params['title_padding'],  # Отступ общего заголовка от верхней границы фигуры
            bottom_padding=params['bottom_padding'],
            legend_fontsize=params['legend_fontsize'],  # Размер шрифта для легенды
            legend_borderaxespad=params['legend_borderaxespad'],  # Отступ легенды от границы фигуры
            legend_ncols=params['legend_ncols'],  # Количество столбцов в легенде
            xtitle=params['xtitle'],  # Подпись оси X
            ytitle=params['ytitle'],  # Подпись оси Y
            wspace=params['wspace'], # расстояние между графиками по горизонтали в масштабе
            hspace=params['hspace'], # расстояние между графиками по вертикали в масштабе
            ncol=params['ncol'],  # параметр для управления количеством столбцов
            fix_lims={
                'ymin': -5,
                'ymax': 110,
            },
            average_only=params['average_only'],
            no_average=params['no_average'],
            scale=params['scale'],
            reverse=params['reverse'],
            infectivity_or_neutralized=params['infectivity_or_neutralized'],
            serum_virus_dict=params['serum_virus_dict'],
        ))

        fig, ax, fit_params = plotData.plotData(
            parsed_dataframe=filtered_data.dropna(),  # DataFrame с данными для построения графиков
        )

        return fig, ax, fit_params

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileSelectorApp()
    ex.show()
    sys.exit(app.exec_())
