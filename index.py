# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Pipeline Description
# 1. Calibrate images using chessboard images, get camera matrix and distortion coefficients.
# 2. Undistort each frame in video using computed camera matrix and distortion coefficients.
# 3. Filter out unnecessary noise in the image, focus on detecting lines:
#     1. Apply gradient threshold.
#     1. Apply color thresholding on S channel in HLS color space.
# 4. Define region of interest, apply perspective transform to warp image into bird-eye view.
# 5. Find the start of the lines using histogram peaks.
# 6. Fit the polynomial by applying sliding window.
# 7. Once polynomials exist from X previous frames, search lines from avg of prior polynomials within margin.
# 8. If lines cannot be detected using search from prior, fallback to histogram peak & sliding window search again.

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math


# %%
# Helper function to plot all images at once
def show_images(images, img_names, save=False, save_prefix=''):
    cols = 2
    rows = math.ceil(len(images)/cols)
    plt.figure(figsize=(15, 15))
    for i in range(0, len(images)):
        img_name = img_names[i]
        plt.subplot(rows, cols, i+1)
        img = images[i]
        cmap = None
        if len(img.shape) < 3:
            cmap = 'gray'

        plt.title(img_names[i])
        plt.imshow(img, cmap=cmap)
        if save:
            img_to_save = img
            if len(img.shape) is 3:
                img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('output_images/' + save_prefix + img_name.split('/')[1], img_to_save)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Camera Calibration: Prepare Object and Image Points

# %%
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points like (0,0,0),(1,0,0),(2,0,0),...(nx-1,ny-1,0)
nx = 9
ny = 6
patternSize = (nx, ny)
objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)


# %%
calibFileNames = glob.glob('camera_cal/calibration*.jpg')
cornersNotFoundCount = 0
lastImgWithCorners = None
for fname in calibFileNames:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    patternWasFound, corners = cv2.findChessboardCorners(gray, patternSize, None)
    if patternWasFound == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        img = cv2.drawChessboardCorners(img, patternSize, corners, patternWasFound)
        lastImgWithCorners = img
        # save images locally
        # imgToSave = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('output_images/' + fname.split('/')[1], imgToSave)
    else:
        cornersNotFoundCount += 1

plt.imshow(lastImgWithCorners)
print("total images ", len(calibFileNames))
print("corners found ", len(objpoints))
print("corners not found ", cornersNotFoundCount)

# %% [markdown]
# ## Camera Calibration: Calibrate, Undistort

# %%
# Calibrate camera -> get camera matrix and distortion coefficients
imgSizeXY = (lastImgWithCorners.shape[1], lastImgWithCorners.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgSizeXY, None, None)

# %% [markdown]
# ### Test image undistortion

# %%
distortedImg = mpimg.imread('camera_cal/calibration1.jpg')
undist = cv2.undistort(distortedImg, mtx, dist, None, mtx)
show_images([distortedImg, undist], ['Distorted', 'Undistorted'])

# %% [markdown]
# ## Gradient Threshold on Test Road Images

# %%
testRoadImgFnames = glob.glob('test_images/test*.jpg')
testRoadImages = list(map(lambda fname: mpimg.imread(fname), testRoadImgFnames))
show_images(testRoadImages, testRoadImgFnames)

# %% [markdown]
## Undistort Images

#%% 
testRoadImagesUndist = list(map(lambda img: cv2.undistort(img, mtx, dist, None, mtx), testRoadImages))
show_images(testRoadImagesUndist, testRoadImgFnames, save=False, save_prefix='undistorted_')


# %%
def absoluteSobelThresh(img, orient='x', sobelKernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    xflag = yflag = 0
    if orient == 'x':
        xflag = 1
    elif orient == 'y':
        yflag = 1
    sobel = cv2.Sobel(gray, cv2.CV_64F, xflag, yflag, ksize=sobelKernel)

    sobelAbs = np.absolute(sobel)
    sobelScaled = (255 * (sobelAbs / np.max(sobelAbs))).astype(np.uint8)

    binaryInThresh = np.zeros_like(sobelScaled)
    binaryInThresh[(sobelScaled >= thresh[0]) & (sobelScaled < thresh[1])] = 1
    return binaryInThresh

def magnitudeThresh(img, sobelKernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobelKernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobelKernel)

    sobelMag = np.sqrt(sobelX**2 + sobelY**2)
    sobelScaled = (255 * (sobelMag/ np.max(sobelMag))).astype(np.uint8)

    binaryInThresh = np.zeros_like(sobelScaled)
    binaryInThresh[(sobelScaled >= thresh[0]) & (sobelScaled < thresh[1])] = 1
    return binaryInThresh

def directionThresh(img, sobelKernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobelKernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobelKernel)

    absGradDir = np.arctan2(np.absolute(sobelY), np.absolute(sobelX))

    binaryInThresh = np.zeros_like(absGradDir)
    binaryInThresh[(binaryInThresh >= thresh[0]) & (binaryInThresh < thresh[1])] = 1
    return binaryInThresh

def combinedGradientThresh(img):
    ksize = 15
    gradX = absoluteSobelThresh(img, orient='x', sobelKernel=ksize, thresh=(20,100))
    gradY = absoluteSobelThresh(img, orient='y', sobelKernel=ksize, thresh=(20,100))
    gradMag = magnitudeThresh(img, sobelKernel=ksize, thresh=(30, 100))
    gradDir = directionThresh(img, sobelKernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(gradX)
    combined[((gradX == 1) & (gradY == 1)) | (gradMag == 1) & (gradDir == 1)] = 1
    return combined

def hlsSChannelThresh(img, sThresh=(170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    sChan = hls[:,:,2]

    sBinary = np.zeros_like(sChan)
    sBinary[(sChan >= sThresh[0]) & (sChan < sThresh[1])] = 1
    return sBinary

def combinedGradAndColorThresh(img, debug=False):
    sChanThresh = hlsSChannelThresh(img)
    combinedGradient = combinedGradientThresh(img)

    # s channel - red; gradient - green
    if debug == True:
        combinedImg = (np.dstack((sChanThresh, combinedGradient, np.zeros_like(sChanThresh))) * 255).astype(np.uint8)
        return np.vstack((img, combinedImg))
    binary = np.zeros_like(sChanThresh).astype(np.uint8)
    binary[(sChanThresh == 1) | (combinedGradient == 1)] = 1
    return binary


# %%
threshImgsDebug = list(map(lambda img: combinedGradAndColorThresh(img, debug=True), testRoadImagesUndist))
show_images(threshImgsDebug, testRoadImgFnames, save=False, save_prefix='combinedThreshDebug_')


# %%
threshImgs = list(map(lambda img: combinedGradAndColorThresh(img), testRoadImagesUndist))
show_images(threshImgs, testRoadImgFnames)

# %% [markdown]
# ## Apply Perspective Transform, Bird-Eye View

# %%
# Draw trapezoid, to see which params fit lanes best

def getTrapezoid():
    yBottom = 685
    yTop = 450
    xLeftBottom = 250
    xRightBottom = 1100
    xOffset = 385
    xLeftUp = xLeftBottom + xOffset
    xRightUp = xRightBottom - xOffset
    return np.array([[xRightBottom,yBottom],[xLeftBottom,yBottom],[xLeftUp,yTop],[xRightUp,yTop]], np.int32)

def drawTrapezoid(img):
    return cv2.polylines(np.copy(img), [getTrapezoid()], True, (255,0,0), thickness=2)

imgsWithTrapezoid = list(map(lambda img: drawTrapezoid(img), testRoadImagesUndist))
show_images(imgsWithTrapezoid, testRoadImgFnames, save=False, save_prefix='trapezoid')


# %%
s = testRoadImages[0].shape
X = s[1]
Y = s[0]
srcPerspective = getTrapezoid().astype(np.float32)
warpXOffset = 350
dstPerspective = np.float32([(X-warpXOffset, Y), (warpXOffset, Y), (warpXOffset, 0), (X-warpXOffset, 0)])
M = cv2.getPerspectiveTransform(srcPerspective, dstPerspective)
MInv = cv2.getPerspectiveTransform(dstPerspective, srcPerspective)

warpedOriginal = list(map(lambda img: cv2.warpPerspective(img, M, (X, Y), flags=cv2.INTER_LINEAR), testRoadImagesUndist))
show_images(warpedOriginal, testRoadImgFnames, save=False, save_prefix='warpedOriginal_')


# %%
warpedImgs = list(map(lambda img: cv2.warpPerspective(img, M, (X, Y), flags=cv2.INTER_LINEAR), threshImgs))
show_images(warpedImgs, testRoadImgFnames, save=False, save_prefix='warpedThresh_')


# %%
def hist(img):
    bottomHalf = img[img.shape[0]//2:, :]
    histogram = np.sum(bottomHalf, axis=0)
    return histogram

def findLanePixels(warpedImg):
    histogram = hist(warpedImg)
    midpoint = histogram.shape[0]//2
    leftBase = np.argmax(histogram[:midpoint])
    rightBase = midpoint + np.argmax(histogram[midpoint:])

    ySize = warpedImg.shape[0]
    xSize = warpedImg.shape[1]
    margin = 90
    minpix = 50
    nwindows = 9
    height = ySize // nwindows

    nonzero = warpedImg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    allLeftLaneIndices = []
    allRightLaneIndices = []

    currentBaseLeft = leftBase
    currentBaseRight = rightBase
    outImg = None
    # outImg = np.dstack((warpedImg,)*3) * 255
    for window in range(nwindows):
        leftYBottom = rightYBottom = ySize - window * height
        leftYTop = rightYTop = ySize - (window+1) * height
        leftXLeft = currentBaseLeft - margin
        leftXRight = currentBaseLeft + margin
        rightXLeft = currentBaseRight - margin
        rightXRight = currentBaseRight + margin

        # cv2.rectangle(outImg,(leftXLeft,leftYBottom),(leftXRight,leftYTop),(0,255,0), 2)
        # cv2.rectangle(outImg,(rightXLeft,rightYBottom),(rightXRight,rightYTop),(0,0,255), 2)

        leftIndices = ((nonzerox >= leftXLeft) & (nonzerox < leftXRight) & (nonzeroy >= leftYTop) & (nonzeroy < leftYBottom)).nonzero()[0]
        rightIndices = ((nonzerox >= rightXLeft) & (nonzerox < rightXRight) & (nonzeroy >= rightYTop) & (nonzeroy < rightYBottom)).nonzero()[0]

        if len(leftIndices) > minpix:
            currentBaseLeft = np.int(np.mean(nonzerox[leftIndices]))
        if len(rightIndices) > minpix:
            currentBaseRight = np.int(np.mean(nonzerox[rightIndices]))

        allLeftLaneIndices.append(leftIndices)
        allRightLaneIndices.append(rightIndices)

    allLeftLaneIndices = np.concatenate(allLeftLaneIndices)
    allRightLaneIndices = np.concatenate(allRightLaneIndices)

    leftX = nonzerox[allLeftLaneIndices]
    leftY = nonzeroy[allLeftLaneIndices]
    rightX = nonzerox[allRightLaneIndices]
    rightY = nonzeroy[allRightLaneIndices]
    return leftX, leftY, rightX, rightY, outImg

def fitPoly(imgShape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial
    leftFit = np.polyfit(lefty, leftx, 2)
    rightFit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    plotY = np.linspace(0, imgShape[0]-1, imgShape[0])
    leftFitX = leftFit[0]*plotY**2 + leftFit[1]*plotY + leftFit[2]
    rightFitX = rightFit[0]*plotY**2 + rightFit[1]*plotY + rightFit[2]
    return leftFitX, rightFitX, plotY, leftFit, rightFit

def fitPolynomialFromScratch(warpedImg):
    leftX, leftY, rightX, rightY, outImg = findLanePixels(warpedImg)
    
    return fitPoly(warpedImg.shape, leftX, leftY, rightX, rightY)
    # UNCOMMENT FOR DEBUGGING
    # outImg[leftY, leftX] = [0, 255, 0]
    # outImg[rightY, rightX] = [0, 0, 255]

    # polyline1 = np.array(list(zip(leftFitX, plotY)))
    # polyline2 = np.array(list(zip(rightFitX, plotY)))
    # cv2.polylines(outImg, np.int32([polyline1, polyline2]), False, (255,255,0), thickness=4)

def searchAroundPoly(warpedImg, leftFit, rightFit):
    margin = 100
    nonzero = warpedImg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftLaneIndices = ((nonzerox >= leftFit[0]*nonzeroy**2 + leftFit[1]*nonzeroy + leftFit[2] - margin) & 
                      (nonzerox <= leftFit[0]*nonzeroy**2 + leftFit[1]*nonzeroy + leftFit[2] + margin)).nonzero()[0]
    rightLaneIndices = ((nonzerox >= rightFit[0]*nonzeroy**2 + rightFit[1]*nonzeroy + rightFit[2] - margin) & 
                       (nonzerox <= rightFit[0]*nonzeroy**2 + rightFit[1]*nonzeroy + rightFit[2] + margin)).nonzero()[0]
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[leftLaneIndices]
    lefty = nonzeroy[leftLaneIndices] 
    rightx = nonzerox[rightLaneIndices]
    righty = nonzeroy[rightLaneIndices]

    # Fit new polynomials
    return fitPoly(warpedImg.shape, leftx, lefty, rightx, righty)


# %%
leftX, leftY, rightX, rightY, imgWithLanePixels = findLanePixels(warpedImgs[1])
show_images([imgWithLanePixels], [testRoadImgFnames[1]], save=False, save_prefix='laneRectanglesAdjusted_')


# %%
leftFitX, rightFitX, plotY, imgWithPoly = fitPolynomialFromScratch(warpedImgs[1])
show_images([imgWithPoly], [testRoadImgFnames[1]], save=False, save_prefix='fitPoly_')

# %%

def drawLane(undistImage, binaryWarped, Minv, leftFitX, rightFitX, plotY):
    warp_zero = np.zeros_like(binaryWarped).astype(np.uint8)
    colorWarp = np.dstack((warp_zero, warp_zero, warp_zero))

    ptsLeft = np.array([np.transpose(np.vstack([leftFitX, plotY]))])
    ptsRight = np.array([np.flipud(np.transpose(np.vstack([rightFitX, plotY])))])
    pts = np.hstack((ptsLeft, ptsRight))
    cv2.fillPoly(colorWarp, np.int_([pts]), (0, 255, 0))

    ySize = undistImage.shape[0]
    xSize = undistImage.shape[1]
    newWarp = cv2.warpPerspective(colorWarp, Minv, (xSize, ySize))
    result = cv2.addWeighted(undistImage, 1, newWarp, 0.3, 0)
    return result

# %% 

fittedLanes = list(map(lambda warped: fitPolynomialFromScratch(warped), warpedImgs))
imgsWithLanes = []
for i in range(len(testRoadImages)):
    imgsWithLanes.append(drawLane(testRoadImages[i], warpedImgs[i], MInv, fittedLanes[i][0], fittedLanes[i][1], fittedLanes[i][2]))

show_images(imgsWithLanes, testRoadImgFnames, save=False, save_prefix='laneOnRoad_')

# %% [markdown]
## Find lanes on video
#%% 
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = []
        self.last_xfitted = None
        self.ploty = None
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def measureCurvatureRadiusOfCurrentFit(self):
        ym_per_pix = 30/720
        xm_per_pix = 3.7/700
        fit_cr = np.polyfit(self.ploty*ym_per_pix, self.last_xfitted*xm_per_pix, 2)
        y_eval = np.max(self.ploty)
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix+fit_cr[1])**2)**(3/2)) / abs(2*fit_cr[0])


# TODO: calculate position of a vehicle with respect to lane's center
class LaneFinder():
    def __init__(self):
        self.leftLine = Line()
        self.rightLine = Line()

    def processNextFrame(self, img):
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        binaryFiltered = combinedGradAndColorThresh(undist)
        xSize = img.shape[1]
        ySize = img.shape[0]
        warped = cv2.warpPerspective(binaryFiltered, M, (xSize, ySize), flags=cv2.INTER_LINEAR)

        if self.leftLine.detected and self.rightLine.detected:
            leftFitX, rightFitX, plotY, leftFit, rightFit = searchAroundPoly(warped, self.leftLine.current_fit, self.rightLine.current_fit)
        else:
            leftFitX, rightFitX, plotY, leftFit, rightFit = fitPolynomialFromScratch(warped)
        self.leftLine.current_fit = leftFit
        self.rightLine.current_fit = rightFit
        self.leftLine.last_xfitted = leftFitX
        self.leftLine.ploty = plotY
        self.rightLine.last_xfitted = rightFitX
        self.rightLine.ploty = plotY
        
        # TODO implement line detected/not detected algorithm
        self.leftLine.detected = True
        self.rightLine.detected = True

        self.leftLine.measureCurvatureRadiusOfCurrentFit()
        self.rightLine.measureCurvatureRadiusOfCurrentFit()

        imgWithLane = drawLane(undist, warped, MInv, leftFitX, rightFitX, plotY)

        # print("L " + str(self.leftLine.radius_of_curvature) + " R " + str(self.rightLine.radius_of_curvature))
        radiusOfCurvatureMean = np.mean([self.leftLine.radius_of_curvature, self.rightLine.radius_of_curvature])

        text = 'Radius of Curvature = ' + str(radiusOfCurvatureMean) + '(m)'
        cv2.putText(imgWithLane, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        return imgWithLane
        

# %%

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# %%

outputFname1 = 'output_videos/project_video.mp4'
clip1 = VideoFileClip('project_video.mp4').subclip(0, 5)
laneFinder = LaneFinder()
processedClip1 = clip1.fl_image(laneFinder.processNextFrame)
processedClip1.write_videofile(outputFname1, audio=False)

# %%
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(outputFname1))

# %%
