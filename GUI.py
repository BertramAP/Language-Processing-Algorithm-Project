import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget, QGridLayout, QPushButton, QLabel, QTextEdit, QStackedWidget, QMainWindow, QMenuBar, QMenu


class MainScreen(QMainWindow):
    def __init__(self):
        super().__init__()


        self.setWindowTitle('RNN')

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.textArea = QTextEdit()
        self.textArea.setReadOnly(False)
        self.textArea.textChanged.connect(self.text_changed)
        #self.textArea.setLineWrapMode(QTextEdit.wrap)

        self.buttons = [QPushButton("Tryk her 1"), 
                        QPushButton("Tryk her 2"),
                        QPushButton("Tryk her 3")
        ]

        self.buttons[0].clicked.connect(self.ins_option_1)
        self.buttons[1].clicked.connect(self.ins_option_2)
        self.buttons[2].clicked.connect(self.ins_option_3)

        self.options = {0: None, 1: None, 2: None}

        self.layout.addWidget(self.textArea, 1, 0, 1, 3)
        self.layout.addWidget(self.buttons[0], 2, 0)
        self.layout.addWidget(self.buttons[1], 2, 1)
        self.layout.addWidget(self.buttons[2], 2, 2)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.menubar = self.menuBar()
        self.filemenu = self.menubar.addMenu("file")
        self.filemenu.addAction("Vælg data", self.open_file_dialog)

    def ins_option_1(self):
        self.textArea.append(self.options[0])

    def ins_option_2(self):
        self.textArea.append(self.options[1])

    def ins_option_3(self):
        self.textArea.append(self.options[2])

    def text_changed(self):
        print("nogen har trykket")
        words = ["Bertrams", "AI", "magi"] #her er det selvfølgelig meninger der skal gives en liste af tensorflow genererede ord
        for index, word in enumerate(words):
            self.buttons[index].setText(word)
            self.options[index] = word

    def open_file_dialog(self):
        filename = QFileDialog.getOpenFileName(self, filter="Text (*.txt)", caption="Select a File", directory="C:/Users")
        print(filename)
        if filename:
            #text = open(filename[0], "r")
            self.textArea.append(filename[0])
            QApplication.processEvents()


app = QApplication(sys.argv)
window = MainScreen()

window.showMaximized()
app.exec()

