# --------------------------------
# gui-disparity-image.py
# --------------------------------
#
# Flow:
# -start with baseline disparity-image
#   -initialize image
#   -render image
# -change parameters
#   -indicators
#   -(+/-)buttons
#     -update indicators
#   -update button
#     -(re)render image
# -update disparity-image
# 
# --------------------------------
#  Dependency (minimum) Versions:
# --------------------------------
#    python3          3.7.5
#    numpy            1.18.3
#    opencv-python    4.2.0.34
#    PyQt5            5.9.2
# --------------------------------
#  CHANGE LOG
# --------------------------------
#  v0.0.2:
#    - kernel size now bounded by image size[DONE]
#    - minor bug fixes[DONE]
#      -- disparities slider starting at 64, should be 16[DONE]
#      -- program crash when scaling was set to 100 using the mouse drag[DONE]
#    - change module importing to improve code readability[DONE]
#  v0.0.1:
#    - added string literals
#    - both images are now scaled to be of equal size
#    - scaling factor now a user input
#    - resize window as image shrinks
#  v0.0.0:
#    - initial deployment
# --------------------------------
#  FUTURE WORK
#   major
#    - implemented rectification previews[TODO]
#    - changed the layout style from vbox to grid[TODO]
#   minor
#    - increased window size at start[TODO]
#    - minor code clean up[TODO]
#    - consolidated image file picking code[TODO]
#    - implemented initial tesets[TODO]
# --------------------------------
import sys
import math
import enum
import operator as op
import numpy as np
import cv2 as cv
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

# enums
class Dimension(enum.IntEnum):
  WIDTH = 0
  HEIGHT = 1
class Shape(enum.IntEnum):
  HEIGHT = 0
  WIDTH = 1
class Kernel(enum.IntEnum):
  NOMINAL = 2
  MIN = 5
  MAX = 255
class Disparities(enum.IntEnum):
  NOMINAL = 16
  MAX = 240
class Opacity(enum.IntEnum):
  NOMINAL = 2
  MIN = 0
  MAX = 100
class Photo(enum.IntEnum):
  LEFT = 0
  RIGHT = 1
  DISPARITY = 2
class Scaling(enum.IntEnum):
  NOMINAL = 2
  MIN = 10
  MAX = 100

# constants
GUI_VERSION = "v0.0.2.a"

literals = {
  "DISPARITIES_LIT":       "Disparities: ",
  "KERNEL_LIT":            "Kernel Size: ",
  "OPACITY_LIT":           "Opacity: ",
  "WINDOW_TITLE_LIT":      "Disparity Image (GUI)",
  "PICK_LT_IMG_LIT":       "Choose Left Image",
  "PICK_RT_IMG_LIT":       "Choose Right Image",
  "NO_IMG_LIT":            "No image chosen",
  "SCALING_LIT":           "Scale Factor: ",
  "IMG_NO_UPDATE_LIT":     "IMAGE NOT UPDATED",
  "WARN_KERNEL_SIZE_LIT":  "Kernel size exceeds image Height or Width. Reduce the kernel size."
}

# main window class
class MainWindow(QtWidgets.QMainWindow):
  def __init__(self):
    self.disparities = op.mul(Disparities.NOMINAL,4)    # number of disparities; used by opencv methods
    self.kernel = op.mul(Kernel.MIN,3)                  # block/kernel size; used by opencv methods
    self.opacity = op.truediv(Opacity.MAX,2)            # opacity of original image shown in GUI
    self.scalingFactor = op.mul(Scaling.MIN,1)          # factor to scale the orignal image(s) by
    self.files = {}                                     # container for photo filepaths

    QtWidgets.QMainWindow.__init__(self)
    # window prep - TODO: change from VBox to Grid layout
    self.setWindowIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TitleBarMenuButton))
    self.setWindowTitle(literals["WINDOW_TITLE_LIT"])
    mainWidget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    # ensure the mainwindow and all element resize as window shrinks
    layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
    self.setFixedSize(layout.sizeHint())
    # buttons
    self.getLeftFile = QtWidgets.QPushButton(self)
    self.getLeftFile.setText(literals["PICK_LT_IMG_LIT"])
    self.getLeftFile.clicked.connect(self.findLFile)

    self.getRightFile = QtWidgets.QPushButton(self)
    self.getRightFile.setText(literals["PICK_RT_IMG_LIT"])
    self.getRightFile.clicked.connect(self.findRFile)
    # sliders
    self.sliderDisparities = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self.sliderDisparities.setRange(Disparities.NOMINAL,Disparities.MAX)
    self.sliderDisparities.setSliderPosition(self.disparities)
    self.sliderDisparities.setSingleStep(Disparities.NOMINAL)
    self.sliderDisparities.setPageStep(op.mul(Disparities.NOMINAL,3))
    self.sliderDisparities.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self.sliderDisparities.setTickInterval(Disparities.NOMINAL)
    self.sliderDisparities.actionTriggered.connect(self.checkActionDisp)
    self.sliderDisparities.sliderReleased.connect(self.snapSliderDisp)

    self.sliderKernel = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self.sliderKernel.setRange(Kernel.MIN,Kernel.MAX)
    self.sliderKernel.setSliderPosition(self.kernel)
    self.sliderKernel.setSingleStep(Kernel.NOMINAL)
    self.sliderKernel.setPageStep(op.mul(Kernel.NOMINAL,10))
    self.sliderKernel.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self.sliderKernel.setTickInterval(Kernel.NOMINAL)
    self.sliderKernel.actionTriggered.connect(self.checkActionKernel)
    self.sliderKernel.sliderReleased.connect(self.snapSliderKernel)

    self.sliderOpacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self.sliderOpacity.setRange(Opacity.MIN,Opacity.MAX)
    self.sliderOpacity.setSliderPosition(self.opacity)
    self.sliderOpacity.setSingleStep(Opacity.NOMINAL)
    self.sliderOpacity.setPageStep(op.mul(Opacity.NOMINAL,5))
    self.sliderOpacity.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self.sliderOpacity.setTickInterval(op.mul(Opacity.NOMINAL,5))
    self.sliderOpacity.actionTriggered.connect(self.checkActionOpacity)
    self.sliderOpacity.sliderReleased.connect(self.snapSliderOpacity)

    self.sliderScaling = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self.sliderScaling.setRange(Scaling.MIN,Scaling.MAX)
    self.sliderScaling.setSliderPosition(self.scalingFactor)
    self.sliderScaling.setSingleStep(Scaling.NOMINAL)
    self.sliderScaling.setPageStep(op.mul(Scaling.NOMINAL,5))
    self.sliderScaling.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self.sliderScaling.setTickInterval(op.mul(Scaling.NOMINAL,5))
    self.sliderScaling.actionTriggered.connect(self.checkActionScaling)
    self.sliderScaling.sliderReleased.connect(self.snapSliderScaling)
    # image placeholders
    self.dispImg = QtWidgets.QLabel()
    # indicators
    self.leftFileName = QtWidgets.QLabel(self)
    self.leftFileName.setText(literals["NO_IMG_LIT"])
    self.rightFileName = QtWidgets.QLabel(self)
    self.rightFileName.setText(literals["NO_IMG_LIT"])
    self.dispVal = QtWidgets.QLabel(self)
    self.dispVal.setText(literals["DISPARITIES_LIT"] + str(self.disparities))
    self.kernelVal = QtWidgets.QLabel(self)
    self.kernelVal.setText(literals["KERNEL_LIT"] + str(self.kernel))
    self.opacityVal = QtWidgets.QLabel(self)
    self.opacityVal.setText(literals["OPACITY_LIT"] + str(op.truediv(self.opacity,Opacity.MAX)))
    self.scalingVal = QtWidgets.QLabel(self)
    self.scalingVal.setText(literals["SCALING_LIT"] + str(self.scalingFactor))
    self.guiVersion = QtWidgets.QLabel(self)
    self.guiVersion.setText(GUI_VERSION)
    self.guiVersion.setAlignment(QtCore.Qt.AlignRight)
    # specify window layout
    layout.addWidget(self.leftFileName)
    layout.addWidget(self.getLeftFile)
    layout.addWidget(self.rightFileName)
    layout.addWidget(self.getRightFile)
    layout.addWidget(self.dispVal)
    layout.addWidget(self.sliderDisparities)
    layout.addWidget(self.kernelVal)
    layout.addWidget(self.sliderKernel)
    layout.addWidget(self.opacityVal)
    layout.addWidget(self.sliderOpacity)
    layout.addWidget(self.scalingVal)
    layout.addWidget(self.sliderScaling)
    layout.addWidget(self.dispImg)
    layout.addWidget(self.guiVersion)
    mainWidget.setLayout(layout)
    self.setCentralWidget(mainWidget)
    # show initial disparity image
    self.drawDispImg()
  # ------------------------------------------
  # DISPARITY slider actions
  def checkActionDisp(self, action):
    if op.eq(action,QtWidgets.QSlider.SliderSingleStepAdd) or op.eq(action,QtWidgets.QSlider.SliderSingleStepSub) or \
       op.eq(action,QtWidgets.QSlider.SliderPageStepAdd) or op.eq(action,QtWidgets.QSlider.SliderPageStepSub):
      self.sliderDisparities.setValue(self.sliderDisparities.sliderPosition())
      self.disparities = self.sliderDisparities.value()
      self.dispVal.setText(literals["DISPARITIES_LIT"] + str(self.disparities))
      self.drawDispImg()

  def snapSliderDisp(self):
    val = op.mul(math.floor(op.add(op.truediv(self.sliderDisparities.value(),Disparities.NOMINAL),0.5)),Disparities.NOMINAL)
    self.sliderDisparities.setSliderPosition(val)
    self.disparities = self.sliderDisparities.value()
    self.dispVal.setText(literals["DISPARITIES_LIT"] + str(self.disparities))
    self.drawDispImg()
  # ------------------------------------------
  # KERNEL slider actions
  def checkActionKernel(self, action):
    if op.eq(action,QtWidgets.QSlider.SliderSingleStepAdd) or op.eq(action,QtWidgets.QSlider.SliderSingleStepSub) or \
       op.eq(action,QtWidgets.QSlider.SliderPageStepAdd) or op.eq(action,QtWidgets.QSlider.SliderPageStepSub):
      self.sliderKernel.setValue(self.sliderKernel.sliderPosition())
      self.kernel = self.sliderKernel.value()
      self.kernelVal.setText(literals["KERNEL_LIT"] + str(self.kernel))
      self.drawDispImg()

  def snapSliderKernel(self):
    if op.eq(op.mod(self.sliderKernel.value(),Kernel.NOMINAL),0):
      self.sliderKernel.setValue(op.add(self.sliderKernel.value(),1))
    self.sliderKernel.setSliderPosition(self.sliderKernel.value())
    self.kernel = self.sliderKernel.value()
    self.kernelVal.setText(literals["KERNEL_LIT"] + str(self.kernel))
    self.drawDispImg()
  # ------------------------------------------
  # OPACITY slider actions
  def checkActionOpacity(self, action):
    if op.eq(action,QtWidgets.QSlider.SliderSingleStepAdd) or op.eq(action,QtWidgets.QSlider.SliderSingleStepSub) or \
       op.eq(action,QtWidgets.QSlider.SliderPageStepAdd) or op.eq(action,QtWidgets.QSlider.SliderPageStepSub):
      self.sliderOpacity.setValue(self.sliderOpacity.sliderPosition())
      self.opacity = self.sliderOpacity.value()
      self.opacityVal.setText(literals["OPACITY_LIT"] + str(op.truediv(self.opacity,Opacity.MAX)))
      self.drawDispImg()

  def snapSliderOpacity(self):
    if op.ne(op.mod(self.sliderOpacity.value(),Opacity.NOMINAL),0):
      self.sliderOpacity.setValue(op.add(self.sliderOpacity.value(),1))
    self.sliderOpacity.setSliderPosition(self.sliderOpacity.value())
    self.opacity = self.sliderOpacity.value()
    self.opacityVal.setText(literals["OPACITY_LIT"] + str(op.truediv(self.opacity,Opacity.MAX)))
    self.drawDispImg()
  # ------------------------------------------
  # SCALING slider actions
  def checkActionScaling(self, action):
    if op.eq(action,QtWidgets.QSlider.SliderSingleStepAdd) or op.eq(action,QtWidgets.QSlider.SliderSingleStepSub) or \
       op.eq(action,QtWidgets.QSlider.SliderPageStepAdd) or op.eq(action,QtWidgets.QSlider.SliderPageStepSub):
      self.sliderScaling.setValue(self.sliderScaling.sliderPosition())
      self.scalingFactor = self.sliderScaling.value()
      self.scalingVal.setText(literals["SCALING_LIT"] + str(self.scalingFactor))
      self.drawDispImg()

  def snapSliderScaling(self):
    if op.ne(op.mod(self.sliderScaling.value(),Scaling.NOMINAL),0):
      self.sliderScaling.setValue(op.add(self.sliderScaling.value(),1))
    self.sliderScaling.setSliderPosition(self.sliderScaling.value())
    self.scalingFactor = self.sliderScaling.value()
    self.scalingVal.setText(literals["SCALING_LIT"] + str(self.scalingFactor,Scaling.MAX))
    self.drawDispImg()
  # ------------------------------------------

  # render the disparity image
  def drawDispImg(self):
    # ensure two images have been chosen before continuing
    if not self.files or op.lt(len(self.files),2):
      return
    # read in images
    cvImgL = cv.imread(self.files[Photo.LEFT],cv.IMREAD_GRAYSCALE)
    cvImgR = cv.imread(self.files[Photo.RIGHT],cv.IMREAD_GRAYSCALE)
    # find smaller image of the two
    if op.lt(cvImgL.shape[Shape.WIDTH],cvImgR.shape[Shape.WIDTH]):
      dim = (int(op.mul(cvImgL.shape[Shape.WIDTH],op.truediv(self.scalingFactor,100))), \
             int(op.mul(cvImgL.shape[Shape.HEIGHT],op.truediv(self.scalingFactor,100))))
    else:
      dim = (int(op.mul(cvImgR.shape[Shape.WIDTH],op.truediv(self.scalingFactor,100))), \
             int(op.mul(cvImgR.shape[Shape.HEIGHT],op.truediv(self.scalingFactor,100))))
    # resize images
    r_cvImgL = cv.resize(cvImgL, dim, cv.INTER_AREA)
    r_cvImgR = cv.resize(cvImgR, dim, cv.INTER_AREA)
    # create disparity image - must be 8bit format to properly render 8bit grayscale qimage
    if op.gt(self.sliderKernel.value(),dim[Dimension.HEIGHT]) or \
       op.gt(self.sliderKernel.value(),dim[Dimension.WIDTH]):
         # warn user of kernel size and do not redraw
         self.showWarning(literals["IMG_NO_UPDATE_LIT"],literals["WARN_KERNEL_SIZE_LIT"])
         return
    stereoObj = cv.StereoBM_create(self.disparities, self.kernel)
    cvImgDisp = np.uint8(stereoObj.compute(r_cvImgL,r_cvImgR))
    # convert opencv image to qt image
    bytesPerLine = op.mul(cvImgDisp.shape[Shape.WIDTH],1)
    height, width = r_cvImgL.shape
    qImgL = QtGui.QImage(r_cvImgL.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
    height, width = cvImgDisp.shape
    qImgDisp = QtGui.QImage(cvImgDisp.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
    # convert qt image to qt pixmap
    qPmapL = QtGui.QPixmap(qImgL)
    qPmapOverlay = QtGui.QPixmap(qImgDisp)
    # overlay images
    painter = QtGui.QPainter(qPmapOverlay)
    painter.setOpacity(op.truediv(self.opacity,Opacity.MAX))
    painter.drawPixmap(QtCore.QPoint(), qPmapL)
    painter.end()
    # render GUI image
    self.dispImg.setPixmap(qPmapOverlay)
  # ------------------------------------------
  # TODO: consolidate file picking mechanism
  def findLFile(self):
    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.DontUseNativeDialog
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Select Left Photo", "","Images (*.png *.jpg)", options=options)
    if fileName:
      self.files[Photo.LEFT] = fileName
      self.leftFileName.setText(self.files[Photo.LEFT])
      self.drawDispImg()

  def findRFile(self):
    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.DontUseNativeDialog
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Select Right Photo", "","Images (*.png *.jpg)",options=options)
    if fileName:
      self.files[Photo.RIGHT] = fileName
      self.rightFileName.setText(self.files[Photo.RIGHT])
      self.drawDispImg()
  # ------------------------------------------
  def showWarning(self,title,msg):
    msgbox = QtWidgets.QMessageBox
    msgbox.warning(self,title,msg)
  # ------------------------------------------

if __name__ == '__main__':
  app = QtWidgets.QApplication(sys.argv)
  mainwindow = MainWindow()
  mainwindow.show()
  app.exec_()
