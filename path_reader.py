from PyQt5 import QtWidgets
from plot_export import Widget
import data_generation as dg
import os

class UploadingWidget_data(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.model_button = QtWidgets.QPushButton('Choose DATA file', self)
        self.model_button.clicked.connect(self.search_path)
        self.model_button.move(15, 10)
        self.resize(100, 50)

    def search_path(self):
        wb_path = QtWidgets.QFileDialog.getOpenFileName()[0]
        dg.path_to_model = os.path.split(wb_path)[0]
        dg.model_name = os.path.basename(wb_path)[:-5]

        dg.path_cps_export = dg.path_to_model + '/CPS_3_exported'
        dg.path_reports_export = dg.path_to_model + '/REPORTS'

        self.w2 = Widget()
        self.w2.show()
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = UploadingWidget_data()
    widget.show()
    app.exec()