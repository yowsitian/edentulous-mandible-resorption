import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from itertools import product

def groupRegion(code):
  if(code == 0):
    return 0
  if(code == 1 or code == 2 or code == 3):
    return 1
  if(code == 4 or code == 5 or code == 6):
    return 2
  if(code == 7 or code == 8 or code == 9):
    return 3
  return -1

def groupSeverity(code):
  if(code == 0):
    return 0
  if(code == 1 or code == 4 or code == 7):
    return "mild"
  if(code == 2 or code == 5 or code == 8):
    return "moderate"
  if(code == 3 or code == 6 or code == 9):
    return "severe"
  return ""

right = [0, 128, 128]
center = [0, 0, 128]
left = [0, 128, 0]

LABEL_TO_COLOR = {1:[128,0,0], 2:[0,128,0], 3:[128,128,0], 4:[0,0,128], 5:[128,0,128], 6:[0,128,128], 7:[128,128,128], 8:[64,0,0], 9:[192,0,0]}
def rgb2mask(rgb):
  ant, lp, rp = {}, {}, {}
  for k,v in LABEL_TO_COLOR.items():
    if(True in np.all(rgb==v,axis=2)):
      region = groupRegion(k)
      severity = groupSeverity(k)
      if(region == 1):
        ant = {'code': v, 'sev': severity}
      elif(region == 2):
        lp = {'code': v, 'sev': severity}
      elif(region == 3):
        rp = {'code': v, 'sev': severity}
  return ant, lp, rp

def addContour(outputBg, mask, maskColor, image):
  # create NumPy arrays from the boundaries
  mask = np.array(mask, dtype="uint8")

  # find the colors within the specified boundaries and apply
  # the mask
  mask = cv2.inRange(image, mask, mask)

  ret,thresh = cv2.threshold(mask, 40, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  c = []
  if len(contours) != 0:

    # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    cv2.drawContours(outputBg, c, -1, maskColor, 3)
    cv2.fillPoly(outputBg, pts =[c], color=maskColor)
  return c

font = cv2.FONT_HERSHEY_COMPLEX

def plotEdgePoints(outputBg, cr):
  # Reading image
  img2 = outputBg

  approx = cv2.approxPolyDP(cr, 0.009 * cv2.arcLength(cr, True), True)

  # draws boundary of contours.
  cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)

  # Used to flatted the array containing
  # the co-ordinates of the vertices.
  n = approx.ravel()
  i = 0
  edges = []

  for j in n :
    if(i % 2 == 0):
      x = n[i]
      y = n[i + 1]

      # String containing the co-ordinates.
      string = str(x) + " " + str(y)

      if(i == 0):
        # text on topmost co-ordinate.
        # cv2.putText(img2, "Arrow tip", (x-50, y-50),
        #         font, 0.5, (255, 0, 0))
        pass
      edges.append((x,y))
      # text on remaining co-ordinates.
      cv2.putText(img2, string, (x, y),
        font, 0.5, (0, 255, 0))
    i = i + 1
  return edges

def find_gradient(p1,p2):
  return (p1[1] - p2[1]) / (p1[0] - p2[0])

def finalPlot(filtered_x_upper_coors, filtered_x_lower_coors, middle_coors, found_match, bottom_left, bottom_right, test_x, test_y, matched, fileName, ori):

  x_plt = [bottom_left[0], bottom_right[0]]
  y_plt = [bottom_left[1], bottom_right[1]]

  matched_px = [matched['upper'][0], matched['lower'][0]]
  matched_py = [matched['upper'][1], matched['lower'][1]]

  fig, ax = plt.subplots(figsize=(8,8))
  ax.set_xlim(0, 1000)
  ax.set_xticks(range(0,2000, 100))

  ax.set_ylim(0, 1000)
  ax.set_yticks(range(0,969, 100))


  # plt.plot(other_plt_x,other_plt_y, color='y', linewidth=5)
  plt.plot(x_plt,y_plt, color='yellow', linewidth=3)
  plt.plot(matched_px,matched_py, color='red', linewidth=3)

  plt.imshow(ori)

  ax.set_axis_off()
  fig.add_axes(ax)

  plt.gca().invert_yaxis()
  plt.savefig(f'my_plot_{fileName}.png', transparent=True,bbox_inches='tight')

def getUpAndLowCoor(squeezed_cr, index_tl, index_tr, index_bl, index_br, top_left, top_right):
  upper_coors, lower_coors = [], []
  firstCoor = min([index_tl, index_tr, index_bl, index_br])

  if(firstCoor == index_tl): 
    upper_coors = squeezed_cr[:index_tl] + squeezed_cr[index_tr:]
    lower_coors = squeezed_cr[index_bl:index_br+1]
  
  else:
    upper_coors = squeezed_cr[index_tr:index_tl+1]
    lower_coors = squeezed_cr[index_bl:index_br+1]

  return [upper_coors, lower_coors]

def corners(np_array):
    ind = np.argwhere(np_array)
    res = []
    for f1, f2 in product([min,max], repeat=2):
        res.append(f1(ind[ind[:, 0] == f2(ind[:, 0])], key=lambda x:x[1]))
    return res

def checkIfDuplicates_3(listOfElems):
    ''' Check if given list contains any duplicates '''    
    for elem in listOfElems:
        if listOfElems.count(elem) > 1:
            return True, elem
    return False, None

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect.tolist()

def getHeight(img,outputBg, contourCoorList, fileName, ori):
  ori_edges = plotEdgePoints(outputBg, contourCoorList)
  ori_edges = [list(ele) for ele in ori_edges]
  squeezed_cr = contourCoorList.copy().squeeze().tolist()

  if(np.array(squeezed_cr).ndim < 2):
    return 0

  edges = order_points(np.array(squeezed_cr))
  top_left = edges[0]
  top_right = edges[1]
  bottom_right = edges[2]
  bottom_left = edges[3]

  flag, dup = checkIfDuplicates_3(edges)

  temp_edges = ori_edges.copy()

  while(flag):
    if(dup in temp_edges):
      temp_edges.remove(dup)
    
    if(np.array(temp_edges).ndim < 2):
      break

    f_edges = order_points(np.array(temp_edges))
    flag, dup = checkIfDuplicates_3(f_edges)
    if(flag):
      temp_edges.remove(dup)
    else:
      if(f_edges[0] == edges[0] or f_edges[0][0] <= edges[0][0]):
        top_left = f_edges[0]
      if(f_edges[1] == edges[1] or f_edges[1][0] >= edges[1][0]):
        top_right = f_edges[1]
      if(f_edges[2] == edges[2] or f_edges[2][0] >= edges[2][0]):
        bottom_right = f_edges[2]
      if(f_edges[3] == edges[3] or f_edges[3][0] <= edges[3][0]):
        bottom_left = f_edges[3]  
  
  if flag:
    return 0

  index_tl = squeezed_cr.index(top_left)
  index_tr = squeezed_cr.index(top_right)
  index_br = squeezed_cr.index(bottom_right)
  index_bl = squeezed_cr.index(bottom_left)

  coors = getUpAndLowCoor(squeezed_cr, index_tl, index_tr, index_bl, index_br, top_left, top_right)  
  upper_coors = coors[0]
  lower_coors = coors[1]

  x_upper_coors = np.array(upper_coors)[:,0]
  x_lower_coors = np.array(lower_coors)[:,0]
  same_x_coors = np.intersect1d(x_upper_coors, x_lower_coors)

  filtered_x_upper_coors, filtered_x_lower_coors, exist_x_upper, exist_x_lower = [], [], [], []
  for i in upper_coors:
    if(i[0] in same_x_coors and i[0] not in exist_x_upper):
      exist_x_upper.append(i[0])
      filtered_x_upper_coors.append(i)
  
  for i in lower_coors:
    if(i[0] in same_x_coors and i[0] not in exist_x_lower):
      exist_x_lower.append(i[0])
      filtered_x_lower_coors.append(i)

  def sorting(coor):
    return same_x_coors.tolist().index(coor[0])

  filtered_x_upper_coors.sort(key=sorting)
  filtered_x_lower_coors.sort(key=sorting)

  middle_coors = []

  for idx, i in enumerate(filtered_x_lower_coors):
    mx = (i[0] + filtered_x_upper_coors[idx][0])/2
    my = (i[1] + filtered_x_upper_coors[idx][1])/2
    middle_coors.append([mx, my])

  m = 0
  test_x = 0
  test_y = 220
  if(bottom_left[1] == bottom_right[1]):
    test_x = bottom_left[0]
  else:
    m = -1 / find_gradient(bottom_left, bottom_right)
    test_x = (test_y-bottom_left[1])/m + bottom_left[0]

  if(np.array(filtered_x_upper_coors).ndim < 2):
    return 0

  filtered_x_upper_y = np.array(filtered_x_upper_coors)[:,1]
  target_coors = []
  for idx, i in enumerate(filtered_x_lower_coors):
    x1 = i[0]
    y1 = i[1]
    for y2 in filtered_x_upper_y:
      x2 = 0
      if(m == 0):
        x2 = x1
      else:
        x2 = (y2 - y1) / m + x1
      target_coors.append({'upper': [math.floor(x2), y2], 'lower':[x1,y1]})
  
  existing_y = []
  existing_x = []
  found_coor = []
  target = [d['upper'] for d in target_coors]
  found_idx = []

  for i in filtered_x_upper_coors:
    if i in target and i[0] not in existing_x and i[1] not in existing_y:
      found_coor.append(i)
      found_idx.append(target.index(i))
      existing_x.append(i[0])
      existing_y.append(i[1])  

  found_match = [target_coors[i] for i in found_idx]
  distance_list = [math.dist(i['upper'], i['lower']) for i in found_match] 

  if len(distance_list) == 0:
    return 0

  height = min(distance_list)

  height_index = distance_list.index(height)
  matched = found_match[height_index]

  finalPlot(filtered_x_upper_coors, filtered_x_lower_coors, middle_coors, found_match, bottom_left, bottom_right, test_x, test_y, matched, fileName, ori)

  height_cm = height * 2.54 / 96
  return height_cm

def getBoneHeightWithImage(ori_input_image, maskRGB_arr):
    ori = cv2.cvtColor(ori_input_image, cv2.COLOR_BGR2RGB)
    image = maskRGB_arr.copy()
    rgb2mask(image)

    right = [0, 128, 0]
    center = [128, 0, 0]
    left = [128, 128, 0]

    bg = [255, 255, 255]

    # create NumPy arrays from the boundaries
    bg = np.array(bg, dtype="uint8")

    maskBg = cv2.inRange(image, bg, bg)
    outputBgRight = cv2.bitwise_and(image, image, mask=maskBg)
    outputBgLeft = cv2.bitwise_and(image, image, mask=maskBg)
    outputBgCenter = cv2.bitwise_and(image, image, mask=maskBg)

    cr = addContour(outputBgRight, right, (0, 128, 128), image)
    cc = addContour(outputBgCenter, center, (0, 0, 128), image)
    cl = addContour(outputBgLeft, left, (0, 128, 0), image)

    if(len(cr) == 0 or len(cc) == 0 or len(cl) == 0):
      return 0,0,0

    imgR = cv2.cvtColor(outputBgRight, cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor(outputBgLeft, cv2.COLOR_BGR2GRAY)
    imgC = cv2.cvtColor(outputBgCenter, cv2.COLOR_BGR2GRAY)

    boneHeightC = getHeight(imgC, outputBgCenter, cc, "center", ori)
    boneHeightL = getHeight(imgL, outputBgLeft, cl, "left", ori)
    boneHeightR = getHeight(imgR, outputBgRight, cr, "right", ori)

    return boneHeightC, boneHeightL, boneHeightR




