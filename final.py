# Imported Libraries
from copy import deepcopy
import pandas as pd

from fil_finder import FilFinder2D
import astropy.units as u

import numpy as np
import cv2
import time
from math import sqrt, ceil
import statistics

from matplotlib import pyplot as plt
import matplotlib.widgets as widgets

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os


# The class that creates the UI
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(523, 351)
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(90, 70, 355, 194))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.submitButton = QtWidgets.QPushButton(self.layoutWidget)
        self.submitButton.setObjectName("submitButton")
        self.gridLayout_2.addWidget(self.submitButton, 2, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_6.addWidget(self.lineEdit)
        self.browseButton = QtWidgets.QPushButton(self.layoutWidget)
        self.browseButton.setObjectName("browseButton")
        self.horizontalLayout_6.addWidget(self.browseButton)
        self.gridLayout.addLayout(self.horizontalLayout_6, 0, 0, 1, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout.addWidget(self.lineEdit_2)
        self.horizontalLayout_5.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_6 = QtWidgets.QLabel(self.layoutWidget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.horizontalLayout.addWidget(self.lineEdit_5)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_2.addWidget(self.lineEdit_3)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_4)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_3.addWidget(self.lineEdit_4)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4.addLayout(self.verticalLayout_3)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.gridLayout.addLayout(self.verticalLayout_5, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Browse Image"))
        self.submitButton.setText(_translate("Form", "Submit"))
        self.browseButton.setText(_translate("Form", "Browse"))
        self.label.setText(_translate("Form", "Enter magnification"))
        self.label_6.setText(_translate("Form", "Enter scale"))
        self.lineEdit_5.setText(_translate("Form", "2"))
        self.label_7.setText(_translate("Form", "μm"))
        self.label_3.setText(_translate("Form", "Enter film height"))
        self.lineEdit_3.setText(_translate("Form", "89"))
        self.label_2.setText(_translate("Form", "mm"))
        self.label_4.setText(_translate("Form", "Enter film width"))
        self.lineEdit_4.setText(_translate("Form", "63"))
        self.label_5.setText(_translate("Form", "mm"))
        self.submitButton.setText(_translate("Form", "Submit"))
        self.browseButton.clicked.connect(self.browse_handler)
        self.submitButton.clicked.connect(self.submit_handler)
        if final_path != "":
            sys.exit()

    # Error messages if user doesn't give an image or magnification
    def error_handler(self, error_message):
        error = QMessageBox()
        error.setIcon(QMessageBox.Warning)
        error.setWindowTitle("Error!")
        error.setText(error_message)
        error.exec()

    def browse_handler(self):
        self.open_dialog_box()

    # The dialog box that enables the user to search an image
    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        self.lineEdit.setText(path)

    def submit_handler(self):
        # Check if all the boxes contain information
        if self.lineEdit.text() == '':
            self.error_handler('Error, please insert an image!')
            return
        if self.lineEdit_2.text() == '':
            self.error_handler('Error, please insert magnification!')
            return
        if self.lineEdit_5.text() == '':
            self.error_handler('Error, please insert scale!')
            return
        if self.lineEdit_3.text() == '':
            self.error_handler('Error, please insert film height in millimeters!')
            return
        if self.lineEdit_4.text() == '':
            self.error_handler('Error, please insert film width in millimeters!')
            return
        # assign values to variables
        path = self.lineEdit.text()
        global final_path, film_height, film_width, film_magnification, film_scale
        final_path = path
        film_height = int(self.lineEdit_3.text())
        film_width = int(self.lineEdit_4.text())
        film_magnification = int(self.lineEdit_2.text())
        film_scale = int(self.lineEdit_5.text())
        Form.close()


# global variables
contour_points = []
distances = []
counter = 0
contour_coord = []
count = 0
coord = []
skeleton = []
n_points = 10
og_y = 0
og_x = 0
new_real_x = 0
new_real_y = 0
cropped_y = 0
cropped_x = 0

starting_x = 0
starting_y = 0
last_x = 0
last_y = 0

og_top_left = [0, 0]
og_top_right = [0, 0]
og_bottom_left = [0, 0]
og_bottom_right = [0, 0]

new_top_left = [0, 0]
new_top_right = [0, 0]
new_bottom_left = [0, 0]
new_bottom_right = [0, 0]


# This function is called to enable the user to choose a part of the film
def onselect(eclick, erelease):
    if eclick.ydata > erelease.ydata:
        eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
    if eclick.xdata > erelease.xdata:
        eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata

    y_dist = abs(erelease.ydata - eclick.ydata)
    x_dist = abs(erelease.xdata - eclick.xdata)
    global og_top_left, og_top_right, og_bottom_left, og_bottom_right, starting_x, starting_y, last_y, last_x
    # Transform the rectangle into a square
    if y_dist >= x_dist:
        starting_x = eclick.xdata
        starting_y = eclick.ydata
        last_x = erelease.ydata - eclick.ydata + eclick.xdata
        last_y = erelease.ydata
    else:
        starting_x = eclick.xdata
        starting_y = eclick.ydata
        last_x = erelease.xdata
        last_y = erelease.xdata - eclick.xdata + eclick.ydata

    # find the corners
    og_top_left = [starting_x, starting_y]
    og_top_right = [last_x, starting_y]
    og_bottom_left = [starting_x, last_y]
    og_bottom_right = [last_x, last_y]

    # present it on a plot
    ax.set_ylim(last_y, starting_y)
    ax.set_xlim(starting_x, last_x)
    fig.canvas.draw()

    # save the plot as image
    plt.savefig('temp.tiff', bbox_inches='tight')
    timer = fig.canvas.new_timer(
        interval=1000)  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(plt.close())
    timer.start()


# This function is called to enable the user to choose the points of the membrane
def click_event(event, x, y, flags, params):
    global count
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates on the Shell
        global coord
        coord.append([x, y])

        cv2.drawMarker(temp_img, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=10, thickness=1, line_type=cv2.LINE_AA)
        cv2.imshow('Please select the area you want to calculate!', temp_img)


# This function removes the background of the contour area
def remove_background(img, new_img):
    img_y = img.shape[0]
    img_x = img.shape[1]

    for y in range(0, img_y):
        for x in range(0, img_x):
            # check for pixels that are not red
            if not ((new_img[y][x][0] >= 0 and new_img[y][x][0] <= 10) and (
                    new_img[y][x][1] >= 0 and new_img[y][x][1] <= 10) and (
                            new_img[y][x][2] >= 220 and new_img[y][x][2] <= 255)):
                # if the pixels are not red remove them from the original image
                img[y][x][0] = 0
                img[y][x][1] = 0
                img[y][x][2] = 0


# this function transforms a point into real scale
def calc(ImageDim, RealDim, Point):
    return (Point[0] * RealDim[0] / ImageDim[0], Point[1] * RealDim[1] / ImageDim[1])


# This function finds the perpendicular lines on the skeleton
def getPerpCoord(a, b, img):
    t_img = deepcopy(img)
    global contour_coord, counter
    # get the point with its predecessor
    aX = a[0]
    aY = a[1]
    bX = b[0]
    bY = b[1]
    # the length of the line
    length = (sqrt((aX - bX) ** 2 + (aY - bY) ** 2)) * 100
    vX = bX - aX
    vY = bY - aY
    if (vX == 0 or vY == 0):
        return 0, 0, 0, 0
    mag = sqrt(vX * vX + vY * vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0 - vY
    vY = temp
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    # draw the line from the initial point
    cv2.line(t_img, (int(cX), int(cY)), (int(dX), int(dY)), [0, 255, 0], 1)
    # find the points of intersection with the contour area
    new_point = compare_contour_and_lines(cont, t_img)
    # if there are 2 points add the markers
    if len(new_point) == 2:
        cv2.drawMarker(t_img, (new_point[0][1], new_point[0][0]), (255, 255, 255), markerType=cv2.MARKER_SQUARE,
                       markerSize=10, thickness=2, line_type=cv2.LINE_AA)
        cv2.drawMarker(t_img, (new_point[1][1], new_point[1][0]), (255, 255, 255), markerType=cv2.MARKER_SQUARE,
                       markerSize=10, thickness=2, line_type=cv2.LINE_AA)
        # save the points to find the distance later
        contour_coord.append([new_point[0], new_point[1]])


# find all the points of the perimeter of the contour
def get_contour_points(imag):
    l = []
    img_y = imag.shape[0]
    img_x = imag.shape[1]
    # search for red pixels
    for y in range(0, img_y):
        for x in range(0, img_x):
            if (imag[y][x][0] >= 0 and imag[y][x][0] <= 10) and (imag[y][x][1] >= 0 and imag[y][x][1] <= 10) and (
                    imag[y][x][2] >= 220 and imag[y][x][2] <= 255):
                # save them in a list
                l.append((y, x))
    return l


# this function finds the points of intersection of the perpendiculars with the contour
def compare_contour_and_lines(l, imag):
    f = []
    flag = 0
    # search in the list with the contour points
    for item in l:
        # if the point in the image is not red anymore save it in a list
        if not (((imag[item[0]][item[1]][0] >= 0) and (imag[item[0]][item[1]][0] <= 10)) and (
                (imag[item[0]][item[1]][1] >= 0) and (imag[item[0]][item[1]][1] <= 10)) and (
                        (imag[item[0]][item[1]][2] >= 220) and (imag[item[0]][item[1]][1] <= 255))) and flag == 0:
            f.append(item)

    # if we didn't find two points of intersection clear the list
    if len(f) != 2:
        f.clear()

    return f


# this function finds the skeleton of the membrane
def find_skeleton(skeleton):
    # We use filfinder
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
                    use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=8 * u.pix, skel_thresh=100 * u.pix, prune_criteria='length')
    skeleton_path = fil.skeleton_longpath
    plt.imshow(t_img, cmap='gray')
    plt.axis('off')
    # we take only the biggest branch of the skeleton
    for idx, filament in enumerate(fil.filaments):
        data = filament.branch_properties.copy()
        data_df = pd.DataFrame(data)
        data_df['offset_pixels'] = data_df['pixels'].apply(lambda x: x + filament.pixel_extents[0])

        longest_branch_idx = data_df.length.idxmax()
        longest_branch_pix = data_df.offset_pixels.iloc[longest_branch_idx]

        y, x = longest_branch_pix[:, 0], longest_branch_pix[:, 1]
        # present it in a plot
        plt.scatter(x, y, s=1, color='b')
    # save the plot as image
    plt.savefig('ls.tiff', bbox_inches='tight')
    plt.close()
    return fil.skeleton_longpath


# this function finds 20 points on the skeleton
def get_skeleton_coordinates(image, img):
    global skeleton
    img_y = image.shape[0]
    img_x = image.shape[1]
    # first search all the image for the skeleton points
    for x in range(0, img_x):
        for y in range(0, img_y):
            # if a point is blue it belongs to the skeleton
            if (image[y][x][0] >= 220 and image[y][x][0] <= 255) and (
                    image[y][x][1] >= 0 and image[y][x][1] <= 10) and (
                    image[y][x][2] >= 0 and image[y][x][2] <= 10):
                img[y][x][0] = 255
                img[y][x][1] = 0
                img[y][x][2] = 0
                # save the point in a list
                skeleton.append((x, y))

    max_x = skeleton[0][0]
    min_x = skeleton[0][0]
    max_y = skeleton[0][1]
    min_y = skeleton[0][1]
    # find the max y coordinate, the min y coordinate, the max x coordinate and the min x coordinate
    for i in skeleton:
        if i[1] >= max_y:
            max_y = i[1]
        if i[1] <= min_y:
            min_y = i[1]
        if i[0] >= max_x:
            max_x = i[0]
        if i[0] <= min_x:
            min_x = i[0]
    # find the differences of the y and x axes
    difference_x = max_x - min_x
    difference_y = max_y - min_y
    # if x difference is bigger then skeleton is horizontal
    if difference_x > difference_y:
        skeleton.clear()
        # find all points and make the check vertically
        for x in range(0, img_x):
            for y in range(0, img_y):
                if (image[y][x][0] >= 220 and image[y][x][0] <= 255) and (
                        image[y][x][1] >= 0 and image[y][x][1] <= 10) and (
                        image[y][x][2] >= 0 and image[y][x][2] <= 10):
                    img[y][x][0] = 255
                    img[y][x][1] = 0
                    img[y][x][2] = 0
                    # save the coordinates
                    skeleton.append((x, y))
    # if y difference is bigger then skeleton is vertical
    else:
        skeleton.clear()
        # find all points and make the check horizontally
        for y in range(0, img_y):
            for x in range(0, img_x):
                if (image[y][x][0] >= 220 and image[y][x][0] <= 255) and (
                        image[y][x][1] >= 0 and image[y][x][1] <= 10) and (
                        image[y][x][2] >= 0 and image[y][x][2] <= 10):
                    img[y][x][0] = 255
                    img[y][x][1] = 0
                    img[y][x][2] = 0
                    # save the coordinates
                    skeleton.append((x, y))
    length = len(skeleton)
    # find 20 points on the skeleton
    num_points = ceil(length / 20)
    t_list = []
    # find 20 points on the skeleton
    for i in range(0, length, num_points):
        t_list.append(skeleton[i])
    skeleton.clear()
    skeleton = t_list


# this function uses a string and presents a message box with the results
def present_results(message):
    res = QMessageBox()
    res.setIcon(QMessageBox.Information)
    res.setWindowTitle("Results!")
    res.setText(message)
    res.exec()


if __name__ == "__main__":
    # vaiables declaration
    final_path = ""
    film_height = 0
    film_width = 0
    film_magnification = 0
    film_scale = 0

    import sys

    # call the UI
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)

    Form.show()
    app.exec_()

    if final_path == "":
        sys.exit()
    # vaiables declaration
    path = final_path
    img = cv2.imread(path)
    magnification = film_magnification
    scale = film_scale
    og_y = img.shape[0]
    og_x = img.shape[1]
    # convert to micrometers
    real_y = film_height * 1000
    real_x = film_width * 1000
    # present the film on a plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    arr = np.asarray(img)
    plt_image = plt.imshow(arr)
    # use a rectangle selector to select an area on the plot
    rs = widgets.RectangleSelector(
        ax, onselect, drawtype='box',
        rectprops=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))
    plt.axis("off")
    plt.show()
    time.sleep(1)
    # read the saved fig again and remove the white background
    img = cv2.imread('temp.tiff')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    # Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # Crop and save it
    x, y, w, h = cv2.boundingRect(cnt)
    img = img[y:y + h, x:x + w]

    # find the real scale dimensions of the image
    new_y = last_y - starting_y
    new_x = last_x - starting_x
    new_real_x = new_x * real_x / og_x
    new_real_y = new_y * real_y / og_y

    temp_img = deepcopy(img)
    new_img = deepcopy(img)
    # save the dimensions of the image
    dim = (img.shape[0], img.shape[1])
    # save the image as a sample
    cv2.imwrite("samples/sample.tiff", img)
    cropped_y = img.shape[0]
    cropped_x = img.shape[1]

    temp_img = deepcopy(img)
    new_img = deepcopy(img)
    # present the image that enables the user to select the contour
    cv2.imshow('Please select the area you want to calculate!', temp_img)
    # setting mouse hadler for the image and calling the click_event() function
    cv2.setMouseCallback('Please select the area you want to calculate!', click_event)
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    # close the window
    cv2.destroyAllWindows()
    contours = [np.array([coord], dtype=np.int32)]
    # use the contour coordinates to draw the contour area
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 0)
    # show the contour
    cv2.imshow("contour area", img)
    cv2.waitKey(0)
    enclosed_area = deepcopy(img)
    t_img = deepcopy(new_img)
    # fill the contour with red colour
    cv2.fillPoly(enclosed_area, pts=[cnt], color=(0, 0, 255))
    # remove the background of the contour area
    remove_background(t_img, enclosed_area)
    gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
    # find the skeleton of the membrane
    skeleton_path = find_skeleton(gray)
    # read the image with the skeleton and remove the white perimeter
    ls_img = cv2.imread("ls.tiff")
    gray_ls = cv2.cvtColor(ls_img, cv2.COLOR_BGR2GRAY)
    th_ls, threshed_ls = cv2.threshold(gray_ls, 240, 255, cv2.THRESH_BINARY_INV)

    # Morph-op to remove noise
    kernel_ls = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed_ls = cv2.morphologyEx(threshed_ls, cv2.MORPH_CLOSE, kernel_ls)

    # Find the max-area contour
    cnts_ls = cv2.findContours(morphed_ls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt_ls = sorted(cnts_ls, key=cv2.contourArea)[-1]

    # Crop and save it
    x, y, w, h = cv2.boundingRect(cnt_ls)
    ls_img = ls_img[y:y + h, x:x + w]
    ls_img = cv2.resize(ls_img, dim, interpolation=cv2.INTER_AREA)
    # save the image with the contour area
    cv2.imwrite("samples/contour area.tiff", img)
    skeleton.clear()
    # find the 20 skeleton coordinates
    get_skeleton_coordinates(ls_img, img)
    cv2.imshow("skeleton", img)
    cv2.waitKey(0)
    img_for_perpendiculars = deepcopy(img)
    # save the skeleton image
    cv2.imwrite("samples/skeleton.tiff", img_for_perpendiculars)
    cont = []
    # find the coordinates of the contour
    cont = get_contour_points(img)
    # find each perpendicular line on the skeleton points
    for index, item in enumerate(skeleton):
        if index != 0:
            getPerpCoord(skeleton[index - 1], item, img_for_perpendiculars)

    line_num = 0
    # present the lines on the image
    for i in contour_coord:
        line_num += 1
        # draw each line
        cv2.line(img, (i[0][1], i[0][0]), (i[1][1], i[1][0]), [0, 255, 0], 2)
        # transform the points to film scale
        Point_one = calc((cropped_y, cropped_x), (new_real_y, new_real_x), (i[0][1], i[0][0]))
        Point_two = calc((cropped_y, cropped_x), (new_real_y, new_real_x), (i[1][1], i[1][0]))
        # calculate the distance
        distance = sqrt((Point_one[0] - Point_two[0]) ** 2 + (Point_one[1] - Point_two[1]) ** 2)
        # transform the distance into real scale
        distance = distance / magnification * 1000
        distance = round(distance, 2)
        distances.append(distance)
        # put the line number in the middle of the line
        midpoint_x = int((i[0][1] + i[1][1]) / 2)
        midpoint_y = int((i[0][0] + i[1][0]) / 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(line_num), (midpoint_x, midpoint_y), font,
                    0.5, (0, 0, 255), 2)
    # remove unnecessary images
    os.remove("temp.tiff")
    os.remove("ls.tiff")
    # present the distances found
    print("distances:", len(distances))
    # the string that we will give in present_reults()
    res_str = "distances:" + str(len(distances)) + "\n" + "\n"
    line_count = 0
    # print every distance
    for i in distances:
        line_count += 1
        print("line", str(line_count) + ":", i, "nm")
        res_str = res_str + "line " + str(line_count) + ": " + str(i) + " nm" + "\n"
    # find and print min distance
    print("min: ", min(distances), "nm")
    # find and print max distance
    print("max: ", max(distances), "nm")
    # find and print average
    print("average: ", round(sum(distances) / len(distances), 2), "nm")
    # find and print st. dev.
    print("standard deviation: ", round(statistics.stdev(distances), 3), "nm")
    # find and print median
    print("median: ", statistics.median(distances), "nm")
    # add them into the message box string
    res_str = res_str + "\n" + "min: " + str(min(distances)) + " nm" + "\n" + "max: " + str(
        max(distances)) + " nm" + "\n" + "average: " + str(
        round(sum(distances) / len(distances), 2)) + " nm" + "\n" + "standard deviation: " + str(
        round(statistics.stdev(distances), 3)) + " nm" + "\n" + "median: " + str(
        statistics.median(distances)) + " nm" + "\n"
    # First quartile (Q1)
    Q1 = np.percentile(distances, 25, interpolation='midpoint')

    # Third quartile (Q3)
    Q3 = np.percentile(distances, 75, interpolation='midpoint')

    # Interquaritle range (IQR)
    IQR = Q3 - Q1

    print("first quartile: ", round(Q1, 3), "nm")
    print("third quartile: ", round(Q3, 3), "nm")
    print("interquartile: ", round(IQR, 3), "nm")
    # add the quartiles in the message box string
    res_str = res_str + "first quartile: " + str(round(Q1, 3)) + " nm" + "\n" + "third quartile: " + str(
        round(Q3, 3)) + " nm" + "\n" + "interquartile: " + str(round(IQR, 3)) + " nm" + "\n"

    cv2.imshow("results", img)
    # show the message box
    present_results(res_str)
    cv2.waitKey(0)
    # save the image
    cv2.imwrite("samples/final.tiff", img)
