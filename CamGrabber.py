import cv2
import numpy as np
import os
import sys
from Logger import write_log, logger
import pickle

WINDOW_POS = None
OUT_FOLDER = None
exit = False
gui = None
capture = False
capture_if_no_chessboard = False

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def getNextFileIdx(out_folder):
    i = 0
    while os.path.exists(os.path.join(out_folder, str(i).rjust(4, '0') + ".jpg")):
        i += 1
    return i

def getFileName(out_folder, idx):
    fname = os.path.join(out_folder, str(idx).rjust(4, '0') + ".jpg")
    return fname

def dumpDict(fname):
    if gui:
        reg_val_widget = gui.refresh_values()
        reg_val = {}
        for address in reg_val_widget:
            val, widget = reg_val_widget[address]
            reg_val[address] = val
        rob_pos = gui.read_robot_pos()

        f = open(fname, "wb")
        pickle.dump([reg_val, rob_pos], f, 0)
        f.close()

def run(out_folder):
    global capture

    if out_folder is None:
        out_folder = logger.outputdir
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    resolution = (960, 720)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, resolution[1])

    diff_frame = cv2.imread("out/2017_5_17__14_15_12/0000.jpg")
    diff_frame = np.zeros_like(diff_frame, dtype="uint8")

    fileIdx = getNextFileIdx(out_folder)
    if not cap.isOpened():
        write_log("ERROR: webcam open failed")
    else:
        cv2.namedWindow("frame")
        if WINDOW_POS:
            x, y = WINDOW_POS
            cv2.moveWindow("frame", x, y)

        while cap.isOpened():
            if exit:
                break
                
            r, frame = cap.read()
            if not r:
                continue

            if resolution[0] not in frame.shape or resolution[1] not in frame.shape:
                # bad resolution
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.putText(frame, "BAD RESOLUTION: %s" % (str(gray.shape)), (0, gray.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key != -1:
                    print "pressed key: %d" % key
                if key == 27:
                    break
                continue

            # cv2.circle(frame, (960 / 2, 720 / 2), 10, (255, 0, 0), 4)
            cv2.imshow("frame", frame)
            diff = np.uint8(np.abs(np.int16(diff_frame) - np.int16(frame)))
            cv2.imshow("frame", diff)
            # enter: 13, escape: 27, space: 32
            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == 32 or capture:
                if capture: capture = False

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    key = 13
                    write_log("Chessboard found")
                else:
                    if capture_if_no_chessboard:
                        key = 13
                    write_log("Chessboard not found.")

            if key == 13:
                filename = getFileName(out_folder, fileIdx)

                success = cv2.imwrite(filename, frame)
                if success:
                    write_log("Success. File saved: %s" % filename)
                    dictfile = os.path.splitext(filename)[0]+".p"
                    dumpDict(dictfile)
                else:
                    write_log("Failed to write to: %s" % filename)
                fileIdx += 1

if __name__ == '__main__':
    run(OUT_FOLDER)

