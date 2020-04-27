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
#    - kernel size now bounded by image size[TODO]
#    - consolidated image file picking code[TODO]
#    - minor code clean up[TODO]
#    - minor bug fixes[TODO]
#      -- disparities slider starting at 64, should be 16[TODO]
#      -- program crash when scaling was set to 100 using the mouse drag[DONE]
#    - change module importing to improve code readability [TODO]
#  v0.0.1:
#    - added string literals
#    - both images are now scaled to be of equal size
#    - scaling factor now a user input
#    - resize window as image shrinks
#  v0.0.0:
#    - initial deployment
# --------------------------------
import sys
import math
import operator as op
import numpy as np
import cv2 as cv
import enum
import PyQt5 as qt
#from PyQt5.QtCore import *
#from PyQt5.QtWidgets import *
#from PyQt5.QtGui import *

# enums
class Dimension(enum.IntEnum):
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
literals = {
  "DISPARITIES_LIT":    "Disparities: ",
  "KERNEL_LIT":	        "Kernel Size: ",
  "OPACITY_LIT":        "Opacity: ",
  "WINDOW_TITLE_LIT":   "Disparity Image (GUI)",
  "PICK_LT_IMG_LIT":    "Choose Left Image",
  "PICK_RT_IMG_LIT":    "Choose Right Image",
  "NO_IMG_LIT":         "No image chosen",
  "SCALING_LIT":        "Scale Factor: "
}
GUI_VERSION = "v0.0.1"

class MainWindow(QMainWindow):
  def __init__(self):
    self.disparities = op.mul(Disparities.NOMINAL,4)    # number of disparities; used by opencv methods
    self.kernel = op.mul(Kernel.MIN,3)                  # block/kernel size; used by opencv methods
    self.opacity = op.truediv(Opacity.MAX,2)            # opacity of original image shown in GUI
    self.scalingFactor = op.mul(Scaling.NOMINAL,5)      # factor to scale the orignal image(s) by
    self.photos = {}                                    # container for photo filepaths

    QMainWindow.__init__(self)
    # window prep - TODO: change from VBox to Grid layout
    self.setWindowIcon(self.style().standardIcon(qt.QStyle.SP_TitleBarMenuButton))
    #self.setGeometry(300,100,300,200)
    self.setWindowTitle(literals["WINDOW_TITLE_LIT"])
    mainWidget = qt.QWidget()
    layout = qt.QVBoxLayout()
    # ensure the mainwindow and all element resize as window shrinks
    layout.setSizeConstraint(qt.QLayout.SetFixedSize)
    self.setFixedSize(layout.sizeHint())
    # buttons
    self.getLeftFile = qt.QPushButton(self)
    self.getLeftFile.setText(literals["PICK_LT_IMG_LIT"])
    self.getLeftFile.clicked.connect(self.findLFile)

    self.getRightFile = qt.QPushButton(self)
    self.getRightFile.setText(literals["PICK_RT_IMG_LIT"])
    self.getRightFile.clicked.connect(self.findRFile)
    # sliders
    self.sliderDisparities = qt.QSlider(Qt.Horizontal)
    self.sliderDisparities.setRange(Disparities.NOMINAL,Disparities.MAX)
    self.sliderDisparities.setSliderPosition(Disparities.NOMINAL)
    self.sliderDisparities.setSingleStep(Disparities.NOMINAL)
    self.sliderDisparities.setPageStep(op.mul(Disparities.NOMINAL,3))
    self.sliderDisparities.setTickPosition(QSlider.TicksBelow)
    self.sliderDisparities.setTickInterval(Disparities.NOMINAL)
    self.sliderDisparities.actionTriggered.connect(self.checkActionDisp)
    self.sliderDisparities.sliderReleased.connect(self.snapSliderDisp)

    self.sliderKernel = qt.QSlider(Qt.Horizontal)
    self.sliderKernel.setRange(Kernel.MIN,Kernel.MAX)
    self.sliderKernel.setSliderPosition(op.mul(Kernel.MIN,3))
    self.sliderKernel.setSingleStep(Kernel.NOMINAL)
    self.sliderKernel.setPageStep(op.mul(Kernel.NOMINAL,10))
    self.sliderKernel.setTickPosition(QSlider.TicksBelow)
    self.sliderKernel.setTickInterval(Kernel.NOMINAL)
    self.sliderKernel.actionTriggered.connect(self.checkActionKernel)
    self.sliderKernel.sliderReleased.connect(self.snapSliderKernel)

    self.sliderOpacity = qt.QSlider(Qt.Horizontal)
    self.sliderOpacity.setRange(Opacity.MIN,Opacity.MAX)
    self.sliderOpacity.setSliderPosition(op.truediv(Opacity.MAX,2))
    self.sliderOpacity.setSingleStep(Opacity.NOMINAL)
    self.sliderOpacity.setPageStep(op.mul(Opacity.NOMINAL,5))
    self.sliderOpacity.setTickPosition(QSlider.TicksBelow)
    self.sliderOpacity.setTickInterval(op.mul(Opacity.NOMINAL,5))
    self.sliderOpacity.actionTriggered.connect(self.checkActionOpacity)
    self.sliderOpacity.sliderReleased.connect(self.snapSliderOpacity)

    self.sliderScaling = qt.QSlider(Qt.Horizontal)
    self.sliderScaling.setRange(Scaling.MIN,Scaling.MAX)
    self.sliderScaling.setSliderPosition(op.mul(Scaling.NOMINAL,5))
    self.sliderScaling.setSingleStep(Scaling.NOMINAL)
    self.sliderScaling.setPageStep(op.mul(Scaling.NOMINAL,5))
    self.sliderScaling.setTickPosition(QSlider.TicksBelow)
    self.sliderScaling.setTickInterval(op.mul(Scaling.NOMINAL,5))
    self.sliderScaling.actionTriggered.connect(self.checkActionScaling)
    self.sliderScaling.sliderReleased.connect(self.snapSliderScaling)
    # image placeholders
    self.dispImg = qt.QLabel()
    # indicators
    self.leftFileName = qt.QLabel(self)
    self.leftFileName.setText(literals["NO_IMG_LIT"])
    self.rightFileName = qt.QLabel(self)
    self.rightFileName.setText(literals["NO_IMG_LIT"])
    self.dispVal = qt.QLabel(self)
    self.dispVal.setText(literals["DISPARITIES_LIT"] + str(self.disparities))
    self.kernelVal = qt.QLabel(self)
    self.kernelVal.setText(literals["KERNEL_LIT"] + str(self.kernel))
    self.opacityVal = qt.QLabel(self)
    self.opacityVal.setText(literals["OPACITY_LIT"] + str(op.truediv(self.opacity,Opacity.MAX)))
    self.scalingVal = qt.QLabel(self)
    self.scalingVal.setText(literals["SCALING_LIT"] + str(self.scalingFactor))
    self.guiVersion = qt.QLabel(self)
    self.guiVersion.setText(GUI_VERSION)
    self.guiVersion.setAlignment(Qt.AlignRight)
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
    # show initial photos
    self.drawDispImg()
  # ------------------------------------------
  # DISPARITY slider actions
  def checkActionDisp(self, action):
    if op.eq(action,qt.QSlider.SliderSingleStepAdd) or op.eq(action,qt.QSlider.SliderSingleStepSub) or \
       op.eq(action,qt.QSlider.SliderPageStepAdd) or op.eq(action,qt.QSlider.SliderPageStepSub):
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
    if op.eq(action,qt.QSlider.SliderSingleStepAdd) or op.eq(action,qt.QSlider.SliderSingleStepSub) or \
       op.eq(action,qt.QSlider.SliderPageStepAdd) or op.eq(action,qt.QSlider.SliderPageStepSub):
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
    if op.eq(action,qt.QSlider.SliderSingleStepAdd) or op.eq(action,qt.QSlider.SliderSingleStepSub) or \
       op.eq(action,qt.QSlider.SliderPageStepAdd) or op.eq(action,qt.QSlider.SliderPageStepSub):
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
    if op.eq(action,qt.QSlider.SliderSingleStepAdd) or op.eq(action,qt.QSlider.SliderSingleStepSub) or \
       op.eq(action,qt.QSlider.SliderPageStepAdd) or op.eq(action,qt.QSlider.SliderPageStepSub):
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
    if not self.photos or op.lt(len(self.photos),2):
      return
    # read in images
    cvImgL = cv.imread(self.photos[Photo.LEFT],cv.IMREAD_GRAYSCALE)
    cvImgR = cv.imread(self.photos[Photo.RIGHT],cv.IMREAD_GRAYSCALE)
    # find smaller image of the two
    if op.lt(cvImgL.shape[Dimension.WIDTH],cvImgR.shape[Dimension.WIDTH]):
      dim = (int(op.mul(cvImgL.shape[Dimension.WIDTH],op.truediv(self.scalingFactor,100))), \
             int(op.mul(cvImgL.shape[Dimension.HEIGHT],op.truediv(self.scalingFactor,100))))
    else:
      dim = (int(op.mul(cvImgR.shape[Dimension.WIDTH],op.truediv(self.scalingFactor,100))), \
             int(op.mul(cvImgR.shape[Dimension.HEIGHT],op.truediv(self.scalingFactor,100))))
    # resize images
    r_cvImgL = cv.resize(cvImgL, dim, cv.INTER_AREA)
    r_cvImgR = cv.resize(cvImgR, dim, cv.INTER_AREA)
    # create disparity image - must be 8bit format to properly render 8bit grayscale qimage
    # TODO: programs crashes if Kernel.value is larger than image width or height
    stereoObj = cv.StereoBM_create(self.disparities, self.kernel)
    cvImgDisp = np.uint8(stereoObj.compute(r_cvImgL,r_cvImgR))
    # convert opencv image to qt image
    bytesPerLine = op.mul(cvImgDisp.shape[Dimension.WIDTH],1)
    height, width = r_cvImgL.shape
    qImgL = qt.QImage(r_cvImgL.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
    height, width = cvImgDisp.shape
    qImgDisp = qt.QImage(cvImgDisp.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
    # convert qt image to qt pixmap
    qPmapL = qt.QPixmap(qImgL)
    qPmapOverlay = qt.QPixmap(qImgDisp)
    # overlay images
    painter = qt.QPainter(qPmapOverlay)
    painter.setOpacity(op.truediv(self.opacity,Opacity.MAX))
    painter.drawPixmap(qt.QPoint(), qPmapL)
    painter.end()
    # render GUI image
    self.dispImg.setPixmap(qPmapOverlay)
  # ------------------------------------------
  # TODO: consolidate file picking mechanism
  def findLFile(self):
    options = qt.QFileDialog.Options()
    options |= qt.QFileDialog.DontUseNativeDialog
    fileName, _ = qt.QFileDialog.getOpenFileName(self,"Select Left Photo", "","Images (*.png *.jpg)", options=options)
    if fileName:
      self.photos[Photo.LEFT] = fileName
      self.leftFileName.setText(self.photos[Photo.LEFT])
      self.drawDispImg()

  def findRFile(self):
    options = qt.QFileDialog.Options()
    options |= qt.QFileDialog.DontUseNativeDialog
    fileName, _ = qt.QFileDialog.getOpenFileName(self,"Select Right Photo", "","Images (*.png *.jpg)",options=options)
    if fileName:
      self.photos[Photo.RIGHT] = fileName
      self.rightFileName.setText(self.photos[Photo.RIGHT])
      self.drawDispImg()
  # ------------------------------------------

if __name__ == '__main__':
  app = qt.QApplication(sys.argv)
  mainwindow = MainWindow()
  mainwindow.show()
  app.exec_()
