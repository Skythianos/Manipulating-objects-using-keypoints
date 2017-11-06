from Tkinter import *
import tkFont
from pyModbusTCP.client import ModbusClient
from threading import Thread, RLock
from Logger import write_log, logger
import CamGrabber
import CalibPoints
import time
from arrangement_calib.chessboard_test import img_test_from_files
from pprint import  pprint

SERVER_HOST = "192.168.0.104"
SERVER_PORT = 502
COUNTER_REGISTER_OUT = 540
COUNTER_REGISTER_IN = 1041

PRINT_ALL_MEMORY_ON_WRITE = True
START_OPENCV_THREAD = True

MAINFRAME_POS = 475, 60
CALIBFRAME_POS = 475, 350
OPENCV_POS = 800, 200
CamGrabber.WINDOW_POS = OPENCV_POS

CALIB_POINTS, CALIB_NUM_ROT_IMGS = CalibPoints.points2
FIND_POINTS = CalibPoints.find_points
ARRANGEMENT_CALIB_DIR = "2017_5_19__17_40_11"

outfile = None
break_wait = False
calib_stepthrough = False
capture_if_no_chessboard = True

def intToUint16(val):
    assert -32768 <= val <= 32767
    return val if val >= 0 else 65536 + val

def uintToInt16(val):
    assert 0 <= val <= 65536
    return val if val <= 32767 else val - 65536

class ClientGUI:
    def __init__(self):
        self.lock = RLock()
        self.calibgui = None
        self.client = ModbusClient()
        self.register_values_widgets = {}
        self.counter = 1
        self.find_thread = None
        self.obj_data = None
        self.stop_signal = False
        self.__build_ui()

    def run_ui(self):
        self.root.mainloop()

    def __build_ui(self):
        # ui hierarchy:
        #
        #root
        #   connectframe
        #       connectlabel
        #       connectbutton
        #       snapshotbutton
        #       calibbuton
        #   mainframe
        #       registerframe
        #           reglabel
        #           registergridframe
        #               ...
        #       outputframe
        #           outputlabel
        #           outputtext

        root = Tk()
        self.root = root
        root.wm_title("RemoteSurf Modbus Client")
        root.protocol("WM_DELETE_WINDOW", self.__delete_window)

        self.font = tkFont.Font(root = root, family = "Helvetica", size = 12)

        connectframe = Frame(root)
        connectbutton = Button(connectframe, text = "Connect", command = self.__connectbutton_click)
        connectlabel = Label(connectframe, text = "Not connected.")
        calibbutton = Button(connectframe, text = "Calibrate", command = self.__calibbutton_click)
        homebutton = Button(connectframe, text = "Home", command = self.__homebutton_click)
        findbutton = Button(connectframe, text = "Find", command = self.__findbutton_click)
        mainframe = Frame(root)
        registerframe = Frame(mainframe)
        reglabel = Label(registerframe, text = "Set registers")
        registergridframe = Frame(registerframe)
        # outputframe = Frame(mainframe)
        # outputlabel = Label(outputframe, text = "Output")
        # vscrollbar = Scrollbar(outputframe)
        # hscrollbar = Scrollbar(outputframe)
        # outputtext = ThreadSafeConsole(outputframe, root, vscrollbar, font = self.font, wrap = NONE)

        connectframe.pack(side = TOP, fill = X)
        connectlabel.pack(side = BOTTOM, anchor = W)
        homebutton.pack(side = RIGHT)
        findbutton.pack(side = RIGHT)
        calibbutton.pack(side = RIGHT)
        connectbutton.pack(side = RIGHT)
        mainframe.pack(side = BOTTOM, fill = BOTH, expand = YES)
        registerframe.pack(side = TOP, expand = YES, anchor = W)
        # outputframe.pack(side = BOTTOM, fill = BOTH, expand = YES)
        reglabel.pack(side = TOP, anchor = CENTER)
        registergridframe.pack(side = BOTTOM, anchor = W)
        # registerframe.config(bg = "cyan")
        # mainframe.config(bg = "pink")
        # registergridframe.config(bg = "red")

        registergridframe.columnconfigure(0, weight = 1)
        registergridframe.columnconfigure(1, weight = 1)
        registergridframe.columnconfigure(2, weight = 1)
        registergridframe.columnconfigure(3, weight = 1)

        self.x_pad = 10
        registergrid_widgets = []
        titles = ["Address", "Label", "Value", ""]
        col = 0
        for title in titles:
            title_label = Label(registergridframe, text = title)
            title_label.grid(row = 0, column = col, padx = self.x_pad)
            registergrid_widgets.append(title_label)
            col += 1

        registers_data = [(500, "x"),
                     (501, "y"),
                     (502, "z"),
                     (503, "A"),
                     (504, "B"),
                     (505, "C"),
                     ]

        for i in range(len(registers_data)):
            reg_data = registers_data[i]
            row = i + 1
            self.__add_register(registergridframe, reg_data, row, registergrid_widgets)

        # hscrollbar.config(orient = HORIZONTAL, command = outputtext.xview)
        # hscrollbar.pack(side = BOTTOM, fill = X)
        # outputtext.config(state = DISABLED, yscrollcommand = vscrollbar.set, xscrollcommand = hscrollbar.set)  #must change to NORMAL before writing text programmatically
        # outputtext.pack(side = LEFT, fill = BOTH, expand = YES, padx = x_padding, pady = y_padding)
        # vscrollbar.config(command = outputtext.yview)
        # vscrollbar.pack(side = RIGHT, fill = Y)

        self.connectframe = connectframe
        self.connectlabel = connectlabel
        self.connectbutton = connectbutton
        self.mainframe = mainframe
        self.registerframe = registerframe
        self.reglabel = reglabel
        self.registergridframe = registergridframe
        self.calibbutton = calibbutton
        # self.outputframe = outputframe
        # self.outputlabel = outputlabel
        # self.vscrollbar = vscrollbar
        # self.hscrollbar = hscrollbar
        # self.outputtext = outputtext

        root.update()
        w, h = root.winfo_width(), root.winfo_height()
        root.minsize(w, h)
        x, y = MAINFRAME_POS
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def __homebutton_click(self):
        values = {
            500: 300,
            501: 0,
            502: 500,
            503: 180,
            504: 0,
            505: 180,
        }
        self.set_values(values, go_to_value = False)

    def __add_register(self, master, data, row, widget_list):
        regaddresslabel = Label(master, text=str(data[0]))
        regaddresslabel.grid(row=row, column=0)
        reglabellabel = Label(master, text=data[1])
        reglabellabel.grid(row=row, column=1)
        regvalueentry = AccessibleEntry(master, justify = RIGHT)
        regvalueentry.set("0")
        regvalueentry.grid(row=row, column=2, padx=self.x_pad)
        regsetbtn = Button(master, text="Set", command = self.__setbutton_click)
        regsetbtn.grid(row=row, column=3)
        widget_list.append(regaddresslabel)
        widget_list.append(reglabellabel)
        widget_list.append(regvalueentry)
        widget_list.append(regsetbtn)
        self.register_values_widgets[data[0]] = (0, regvalueentry)

    def __calibbutton_click(self):
        if not self.calibgui:
            self.calibgui = CalibGUI(self)

    def __findbutton_click(self):
        if self.find_thread is None:
            self.find_thread = Thread(target=self.__find_object)
            self.find_thread.start()

    def __find_object(self):
        import DataCache as DC
        from glob import glob
        from os.path import join
        import numpy as np
        from SFMSolver import SFMSolver, find_ext_params
        import Utils

        print "FINDING"

        np.set_printoptions(precision=3, suppress=True)

        files_dir = "out/2017_3_8__14_51_22/"
        files = glob(join(files_dir, "*.jpg"))
        masks = []
        for f in files:
            m = f.replace(".jpg", "_mask.png")
            masks.append(m)
        sfm = SFMSolver(files, masks)
        if self.obj_data is None:
            imgs, kpts, points, data = sfm.calc_data_from_files_triang_simple()
            self.obj_data = imgs, kpts, points, data
        else:
            imgs, kpts, points, data = self.obj_data

        arr_calib = DC.getData("out/%s/arrangement_calib.p" % ARRANGEMENT_CALIB_DIR)
        ttc = arr_calib["ttc"]
        tor = arr_calib["tor"]
        if "cam_mtx" in arr_calib:
            print "camMtx, distcoeffs load"
            Utils.camMtx = arr_calib["cam_mtx"]
            Utils.dist_coeffs = arr_calib["dist_coeffs"]

        if self.stop_signal:
            self.stop_signal = False
            return

        for point in FIND_POINTS:
            values = {
                500: point[0],
                501: point[1],
                502: point[2],
                503: point[3],
                504: point[4],
                505: point[5],
            }
            print "set_values call"
            self.set_values(values, True)
            print "set_values return"

            time.sleep(0.5)
            CamGrabber.capture_if_no_chessboard = True
            CamGrabber.capture = True
            time.sleep(0.5)

            if self.stop_signal:
                self.stop_signal = False
                return

        find_dir = logger.outputdir
        files = glob("%s/*.jpg" % find_dir)
        print files
        # files_dir = "out/2017_4_5__15_57_20/"
        # files = glob(join(files_dir, "*.jpg"))
        files.sort()
        files = files[-len(FIND_POINTS):]
        results = []

        for f in files:
            res = find_ext_params(f, imgs, kpts, points, data, tor, ttc)
            results.append(res)
            if self.stop_signal:
                self.stop_signal = False
                return

        for i in range(len(results)):
            print i, results[i]
            write_log((i, results[i]))
        result = max(results, key=lambda x: x[2])
        write_log(result)


        values = {
            500: int(result[0][0] * 10),
            501: int(result[0][1] * 10),
            502: int(result[0][2] * 10) + 200,
            503: int(result[1][2]),
            504: int(result[1][1]),
            505: int(result[1][0]),
        }

        print "num inl: ", result[2]
        pprint(values)
        self.set_values(values, go_to_value=False)
        self.find_thread = None

    def __connectbutton_click(self):
        if self.client.is_open():
            self.client.close()
        else:
            self.client.host(SERVER_HOST)
            self.client.port(SERVER_PORT)
            if self.client.open():
                write_log("Connection established")
                self.refresh_values()
                self.read_robot_pos()
            else:
                write_log("ERROR: Connecting failed")
        self.__update_gui()

    def read_robot_pos(self):
        write_log("Reading robot position:")
        posdict = {}
        for i in range(1000, 1006):
            if self.client.is_open():
                with self.lock:
                    real_val_uint = self.client.read_input_registers(i)[0]
                    real_val_holding_uint = self.client.read_holding_registers(i)[0]
                assert real_val_uint == real_val_holding_uint
                real_val_int = uintToInt16(real_val_uint)
                posdict[i] = real_val_int
                write_log("%d, %d" % (i, real_val_int))
            else:
                write_log("ERROR: Read could not be completed, client not connected.")
                self.__update_gui()
                break
        write_log("Read done.")
        return posdict

    def refresh_values(self):
        for address in self.register_values_widgets:
            if self.client.is_open():
                value, widget = self.register_values_widgets[address]
                with self.lock:
                    real_val_uint = self.client.read_input_registers(address)[0]
                    real_val_holding_uint = self.client.read_holding_registers(address)[0]
                assert real_val_uint == real_val_holding_uint
                real_val_int = uintToInt16(real_val_uint)
                widget.set(str(real_val_int))
                self.register_values_widgets[address] = (real_val_int, widget)
            else:
                write_log("ERROR: Read could not be completed, client not connected.")
                self.__update_gui()
                break
        write_log("Refresh done.")
        return self.register_values_widgets

    def __update_gui(self):
        if self.client.is_open():
            self.connectlabel.config(text = "Connected to: %s:%d" % (SERVER_HOST, SERVER_PORT))
            self.connectbutton.config(text = "Disconnect")
        else:
            self.connectbutton.config(text = "Connect")
            self.connectlabel.config(text = "Not connected.")
        self.root.update()

    def __print_memory(self):
        self.refresh_values()
        write_log("Memory dump:")
        write_log("------------")
        for address in self.register_values_widgets:
            val, widget = self.register_values_widgets[address]
            write_log("%d, %d" % (address, val))
        write_log("------------")

    def __setbutton_click(self, wait = False):
        if not self.client.is_open():
            write_log("ERROR: Not connected to client")
            return

        # writing message counter
        retval = self.__write_register(COUNTER_REGISTER_OUT, self.counter)
        if not retval:
            self.__update_gui()
            return

        # writing registers
        for address in self.register_values_widgets:
            value, widget = self.register_values_widgets[address]
            widgetvalue_int = None
            try:
                widgetvalue_int = int(widget.get())
            except ValueError:
                write_log("ERROR: Wrong input format in value entry for address: %d" % address)
                continue

            if value == widgetvalue_int:
                continue

            retval = self.__write_register(address, widgetvalue_int)
            if retval:
                self.register_values_widgets[address] = (widgetvalue_int, widget)
            else:
                self.__update_gui()
        self.refresh_values()

        # message counter wait
        if wait:
            global break_wait
            while not break_wait:
                with self.lock:
                    counter = self.client.read_input_registers(COUNTER_REGISTER_IN)[0]
                if counter == self.counter:
                    break
                time.sleep(0.1)
            break_wait = False

        # counter increment
        self.counter = (self.counter + 1) % 20

        if PRINT_ALL_MEMORY_ON_WRITE:
            self.__print_memory()
            self.read_robot_pos()

    def __write_register(self, address, value):
        if not (-32768 <= value <= 32767):
            write_log("ERROR: -32768 <= value <= 32767 is false for address: %d" % address)
            return False

        widgetvalue_uint = intToUint16(value)
        if self.client.is_open():
            with self.lock:
                retval = self.client.write_single_register(address, widgetvalue_uint)
            if retval:
                write_log("Register written. Address: %d, value: %d" % (address, value))
                return True
            else:
                write_log("ERROR: Write failed. Address: %d, value: %d" % (address, value))
        else:
            write_log("ERROR: client not connected.")
        return False

    def set_values(self, values, wait = True, go_to_value = True):
        """
        :param values: dictionary of { address : value} both int
        :return:
        """
        for address in values:
            if address not in self.register_values_widgets:
                continue

            val, widget = self.register_values_widgets[address]
            widget.set(str(values[address]))
        if go_to_value:
            self.__setbutton_click(wait)

    def __delete_window(self):
        CamGrabber.exit = True
        self.stop_signal = True
        self.client.close()
        self.root.quit()

class CalibGUI:
    def __init__(self, parent):
        self.parent = parent
        self.next_point_idx = 0
        self.calib_thread = None
        self.stop_signal = False
        self.stepthrough_next_point = False

        self.__build_ui(parent)

    def __build_ui(self, parent):
        # ui hierarchy:
        #
        # window
        #   statuslabel
        #   forwardbutton

        window = Toplevel(parent.root)
        window.wm_title("Calibration")
        window.protocol("WM_DELETE_WINDOW", self.__delete_window)
        self.window = window

        statuslabel = Label(window, text = "Ready")
        forwardbutton = Button(window, text = ">>", command = self.__forwardbutton_click)

        forwardbutton.pack(side = RIGHT)
        statuslabel.pack(side = LEFT)

        self.forwardbutton = forwardbutton
        self.statuslabel = statuslabel

        window.update()
        w, h = window.winfo_width(), window.winfo_height()
        window.minsize(w, h)
        x, y = CALIBFRAME_POS
        window.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def __forwardbutton_click(self):
        if self.calib_thread is None:
            self.calib_thread = Thread(target=self.__calibrate)
            self.calib_thread.start()
        else:
            global calib_stepthrough
            if calib_stepthrough:
                self.stepthrough_next_point = True

    def __calibrate(self):
        global break_wait
        break_wait = False

        while self.next_point_idx < len(CALIB_POINTS):
            print "next point idx: %d" % self.next_point_idx
            point = CALIB_POINTS[self.next_point_idx]
            values = {
                500: point[0],
                501: point[1],
                502: point[2],
                503: point[3],
                504: point[4],
                505: point[5],
            }
            self.parent.set_values(values, True)

            time.sleep(0.5)
            CamGrabber.capture_if_no_chessboard = capture_if_no_chessboard
            CamGrabber.capture = True
            time.sleep(0.5)

            self.next_point_idx += 1

            if calib_stepthrough:
                while not self.stepthrough_next_point and not self.stop_signal:
                    time.sleep(0.05)
                self.stepthrough_next_point = False

            if self.stop_signal:
                self.stop_signal = False
                return

        img_test_from_files(logger.outputdir)
        self.calib_thread = None

    def __delete_window(self):
        if self.calib_thread is not None:
            global break_wait
            break_wait = True
            self.stop_signal = True
            self.calib_thread.join()

        self.parent.calibgui = None
        self.window.destroy()

class AccessibleEntry(Entry):
    def __init__(self, master, cnf = {}, **kw):
        Entry.__init__(self, master, cnf, **kw)
        self.var = StringVar()
        self.config(textvariable = self.var)

    def get(self):
        return self.var.get()

    def set(self, val):
        self.var.set(val)

def runOpencv():
    CamGrabber.run(None)

if __name__ == '__main__':
    opencvThread = None
    gui = ClientGUI()
    if START_OPENCV_THREAD:
        opencvThread = Thread(target=runOpencv)
        CamGrabber.gui = gui
        opencvThread.start()

    gui.run_ui()
    if opencvThread is not None:
        opencvThread.join()