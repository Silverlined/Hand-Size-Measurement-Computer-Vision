import os.path
import sys
import cv2
from skimage import measure, draw
from scipy import optimize, ndimage
from matplotlib.pyplot import plot, imshow, show
import imutils
import numpy as np
from numpy import where, cos, sin, count_nonzero
import pandas as pd
from distances import getSlopeIntercept, lineAt, vertDistFingers, horDistFingers
import matplotlib.pyplot as plt

print("OpenCV Version:", cv2.__version__)

path = 'image_2.jpg' 

size_coin1 = 23.25 # [mm] 
size_coin2 = 25.75 # [mm]

def segmentImage(path):
    img = cv2.imread(path)
    assert img is not None, "No Image"
    img = imutils.resize(img, width=1700)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray,(5,5),0) 

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_TRUNC | cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations = 3)

    _, img_marked = cv2.connectedComponents(closed)

    stats = measure.regionprops_table(img_marked, img, properties=['label',
                                                                   'area',
                                                                   'centroid',
                                                                   'bbox',
                                                                   'orientation',
                                                                   'major_axis_length',
                                                                   'minor_axis_length',
                                                                   'equivalent_diameter'])
    properties = pd.DataFrame(stats)

    return img_marked, properties

def getHandSegment(img_segmented, properties):
    index = properties['area'].idxmax() + 1
    hand = where(img_segmented == index, np.uint8(255), np.uint8(0))
    return hand, index

def getCoin(img_segmented, index, properties):
    diameter_coin = properties.loc[properties['label'] == index].equivalent_diameter.item()
    coin = where(img_segmented == index, np.uint8(255), np.uint8(0))
    return coin, diameter_coin

def fitCircle(img_segmented, properties, hand_index, hand_segment):
    y0_centre = properties.loc[properties["label"] == hand_index]['centroid-0'].item()
    x0_centre = properties.loc[properties["label"] == hand_index]['centroid-1'].item()
    radius0 = properties.loc[properties["label"] == hand_index].major_axis_length / 8.0

    orientation = properties.loc[properties['label'] == hand_index]['orientation'].item()
    minor_axis_length = properties.loc[properties['label'] == hand_index]['minor_axis_length'].item()
    major_axis_length = properties.loc[properties['label'] == hand_index]['major_axis_length'].item()
    x1 = x0_centre + cos(orientation) * 0.5 * minor_axis_length
    y1 = y0_centre - sin(orientation) * 0.5 * minor_axis_length
    x2 = x0_centre - sin(orientation) * 0.5 * major_axis_length
    y2 = y0_centre - cos(orientation) * 0.5 * major_axis_length
    allowed_margin = [axis / 30 for axis in hand_segment.shape]

    def _lossFunction(params):
        isInside = 1
        x, y, r = params
        coords = draw.disk((y, x), r, shape=hand_segment.shape)
        template = np.zeros_like(hand_segment)
        template[coords] = 255

        # Check if circle exceeds hand area
        if count_nonzero(cv2.subtract(template, hand_segment)) > 0:
            isInside = 0

        # Check if circle is close to centroid
        elif abs(y - y0_centre) > allowed_margin[0] or abs(x - x0_centre) > allowed_margin[1]:
            isInside = 0

        return -np.sum(template == hand_segment) * isInside

    x_centre, y_centre, radius = optimize.fmin(_lossFunction, (x0_centre, y0_centre, radius0))
    diameter = 2 * radius

    return diameter

def fitConvexPolygon(hand_segment):
    cnts = cv2.findContours(hand_segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(cnts)

    convex_hull = [cv2.convexHull(c) for c in contours]

    thresh_low = 20
    thresh_high = 200

    groups = {}
    tmp = []
    previous = convex_hull[0][0][0]

    convex_points = convex_hull[0][1:].reshape(-1,2)
    n = 0
    tmp.append(previous)
    for c in convex_points:
        distance = np.linalg.norm(previous-c)   #Euclidean distance

        if distance > thresh_high and tmp:
            groups[n] = np.array(tmp)
            tmp = []
            n += 1

        if distance < thresh_low:
            tmp.append(c)

        previous = c

    return groups

def labelPoints(groups, labels):
    points = {}

    def _getAverage(group):
        return int(sum(group[0]) / len(group[0])), int(sum(group[1]) / len(group[1]))

    for key in groups.keys():
        points[key] = _getAverage(groups[key].T)

    max_x, min_y = 0, 2338
    wristKeys = [0, 0]

    for key, value in points.items():
        if value[1] < min_y:
            min_y = value[1]
            wristKeys[0] = key

        if value[0] > max_x:
            max_x = value[0]
            wristKeys[1] = key

    if wristKeys[0] == wristKeys[1]:
        wristKeys[1] += 1
    
    
    
    counter = 0
    for key in points.fromkeys(points):
        if key >= wristKeys[1] and key < wristKeys[1] + 5:
            points[labels[counter]] = points.pop(key)
            counter += 1
        else:
            if key != wristKeys[0]:
                points.pop(key)
    
    points["wrist"] = points.pop(wristKeys[0])
    return points

def update_goodhandscans_csv():
    
    # extract the name of the good scans
    df = pd.read_csv(r"%s" % str(sys.argv[1]), 
                     usecols=['file name','Quality','hand'],
                     index_col=False)
    
    GoodImages = df.loc[df['Quality']=='good']
    GoodImages.pop('Quality')
    
    # check if there is already a csv with the good scans else create one
    if os.path.isfile('GoodHandscans.csv'):
        print ("Opening hand scans file...")
    else:
        print ("Creating good images file...")
        df2 = pd.DataFrame({'file name':[],
                                'hand':[]})
        df2.to_csv('GoodHandscans.csv',index=False)
    
    df_good = pd.read_csv('GoodHandscans.csv')
    
    # find new scans that are not in the GoodHandscans.csv
    GoodImages_copy = GoodImages.copy()
    GoodImages_copy['exists'] = GoodImages['file name'].isin(df_good['file name'])
    newImages = GoodImages_copy.loc[GoodImages_copy['exists']==False]
    newImages.pop('exists')
    #newImages.set_index('file name', inplace=True)
    
    # append new scans to the csv
    newImages.to_csv('GoodHandscans.csv', mode='a', header=False)
    
    if newImages.empty:
        print("No new images")
    else:
        print("New images:\n %s" % newImages.to_string())
        print("Good images saved")
    
    return 

def get_unprocessed_images():

    # check if there is already a csv with the measurements else create one
    if os.path.isfile('measurements.csv'):
        print ("Opening measurements file...")
    else:
        print ("Creating measurements file...")
        df2 = pd.DataFrame({'file name':[],
                            'hand':[],
                            'width': [],
                            'length': [],
                            'distance_1_2': [],
                            'distance_1_3': [],
                            'distance_2_3': [],
                            'distance_2_5': [],
                            'distance_1_5': []})
        df2.to_csv('measurements.csv',index=False)

    measurements = pd.read_csv('measurements.csv')
    good_images = pd.read_csv('GoodHandscans.csv')
    
    # find new scans in 'GoodHandscans.csv' that are not in the measurements.csv
    newImages = good_images.copy()
    newImages['exists'] = good_images['file name'].isin(measurements['file name'])
    newImages = newImages.loc[newImages['exists']==False]
    newImages.pop('exists')
    # newImages.set_index('file name', inplace=True)
    
    return newImages

def main():
    
    # Check the handscans folder, add images that are already
    # not in the good hand scans folder.
    update_goodhandscans_csv()    

    img_to_process = get_unprocessed_images()

    data_out = pd.read_csv('measurements.csv')
    if data_out.empty:
        data_out = pd.DataFrame({'file name':[],
                            'hand':[],
                            'width': [],
                            'length': [],
                            'distance_1_2': [],
                            'distance_1_3': [],
                            'distance_2_3': [],
                            'distance_2_5': [],
                            'distance_1_5': []})

    for file, side in zip(img_to_process['file name'], img_to_process['hand']):    
        labels = ["thumb", "index", "middle", "ring", "pinky"]
        if side == "right":
                labels = labels[::-1]
                
        img_segmented, properties = segmentImage(file)    
        
        hand_segment, hand_index = getHandSegment(img_segmented, properties)
        
        index1 = properties.loc[properties['area'] < 30000].label.item()
        index2 = properties.loc[(properties['area'] > 30000) & (properties['area'] < 35000)].label.item()
        coin1, diameter1 = getCoin(img_segmented, index1, properties)
        coin2, diameter2 = getCoin(img_segmented, index2, properties)
    
        diameter = fitCircle(img_segmented, properties, hand_index, hand_segment)
        groups = fitConvexPolygon(hand_segment)
        
        points = labelPoints(groups, labels)
        wrist_point = points["wrist"]
    
        wrist_point2 = (wrist_point[0] + 1, wrist_point[1])
        slope, intercept = getSlopeIntercept(wrist_point, wrist_point2)
    
        pixels_per_mm_1 = diameter1 / size_coin1
        pixels_per_mm_2 = diameter2 / size_coin2
        pixels_per_cm = (pixels_per_mm_1 + pixels_per_mm_2) * 0.5 * 10
    
        width = diameter / pixels_per_cm
        length = vertDistFingers(points["middle"], wrist_point, slope, intercept) / pixels_per_cm
        distance_1_2 = vertDistFingers(points["thumb"], points["index"], slope, intercept) / pixels_per_cm
        distance_1_3 = vertDistFingers(points["thumb"], points["middle"], slope, intercept) / pixels_per_cm
        distance_2_3 = vertDistFingers(points["index"], points["middle"], slope, intercept) / pixels_per_cm
        distance_2_5 = horDistFingers(points["index"], points["pinky"], slope, intercept) / pixels_per_cm
        distance_1_5 = horDistFingers(points["thumb"], points["pinky"], slope, intercept) / pixels_per_cm

        print("Hand Width: %.2f cm" % width)
        print("Hand Length: %.2f cm" % length)
        print("1) Vertical distance between thumb and index: %.2f cm" % distance_1_2)
        print("2) Vertical distance between thumb and middle: %.2f cm" % distance_1_3)
        print("3) Vertical distance between index and middle: %.2f cm" % distance_2_3)
        print("4) Horizontal distance between index and pinky: %.2f cm" % distance_2_5)
        print("5) Horizontal distance between thumb and pinky: %.2f cm" % distance_1_5)
        
        data_out = data_out.append({'file name': file,
                                'hand': side,
                                'width': "%.2f" %width,
                                'length': "%.2f" %length,
                                'distance_1_2': "%.2f" %distance_1_2,
                                'distance_1_3': "%.2f" %distance_1_3,
                                'distance_2_3': "%.2f" %distance_2_3,
                                'distance_2_5': "%.2f" %distance_2_5,
                                'distance_1_5': "%.2f" %distance_1_5}, ignore_index=True)
        
    data_out.to_csv('measurements.csv', header=True)


if __name__ == "__main__":
    main()
