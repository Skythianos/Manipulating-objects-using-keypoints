import datetime
from os import makedirs
from os.path import exists, dirname, join

PRINT_LOG_TO_STDOUT = True

class MyLogger:
    __singleton = None
    def __init__(self, outfile = None):
        print "mylogger starting"
        self.outfile = None
        if MyLogger.__singleton is None:
            print "singleton not created yet"
            MyLogger.__singleton = self

        if outfile is not None:
            print "overwriting outfile location"
            self.outfile = outfile

        if self.outfile is None:
            print "new outfile location"
            self.create_new_file()

        self.outputdir = dirname(self.outfile)

        print "mylogger started, logging to: %s" % self.outfile

    def create_new_file(self):
        datestr = self.getDateString()
        dir = join("out", datestr)
        self.outfile = join(dir, "out.txt")

    def getDateString(self):
        now = datetime.datetime.now()
        return "%d_%d_%d__%d_%d_%d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)

    def write_log(self, text):
        if PRINT_LOG_TO_STDOUT: print text
        if self.outfile is None: return
        if not exists(self.outputdir):
            print "creating output dir"
            makedirs(self.outputdir)

        f = file(self.outfile, "a")
        f.write("%s\n" % text)
        f.close()

logger = MyLogger(None)

def write_log(text):
    logger.write_log(str(text))