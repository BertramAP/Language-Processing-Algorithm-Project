import sys, cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget, QGridLayout, QPushButton, QLabel, QTextEdit, QStackedWidget, QMainWindow


class MainScreen(QMainWindow):
    def __init__(self):
        super().__init__()


        self.setWindowTitle('RNN')

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.textArea = QTextEdit()
        self.textArea.setReadOnly(False)
        #self.textArea.setLineWrapMode(QTextEdit.wrap)

        self.button = QPushButton("Tryk her")
        self.button.clicked.connect(self.open_file_dialog)

        self.layout.addWidget(self.textArea, 1, 0)
        self.layout.addWidget(self.button, 2, 0)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def open_file_dialog(self):
        filename = QFileDialog.getOpenFileName(self, filter="Text (*.txt)", caption="Select a File", directory="C:/Users")
        if filename:
            text = open(filename, "r")
            self.textArea.append(text)
            QApplication.processEvents()


app = QApplication(sys.argv)
window = MainScreen()

window.showMaximized()
app.exec()

