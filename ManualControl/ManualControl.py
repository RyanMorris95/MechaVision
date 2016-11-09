import sys
from PyQt4 import QtCore, QtGui, uic
import pygame
from ManualControlForm import Ui_MainWindow

class MyWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        
        # set up the user interface from Designer
        self.ui = Ui_MainWindow()
	self.ui.setupUi(self)	

	# Connect up the buttons
	self.ui.pushButton.clicked.connect(self.clicked)

    def clicked(self):
	print "Button was clicked"
'''
def PyGameController:
    pygame.init()
    s = pygame.Surface((640, 480))
    s.fill(( 
 '''


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
