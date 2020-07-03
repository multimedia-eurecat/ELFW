import sys, os, time

# Rafael Redondo, Jaume Gibert - Eurecat (c) 2019

class bcolors:
    PURPLE = '\033[95m'
    BLUE   = '\033[94m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'
    ENDC   = '\033[0m'
    BOLD   = '\033[1m'
    CYAN   = '\033[96m'

class AverageMeter(object):
    def __init__(self,offset=0):
        self.reset(offset)

    def reset(self, offset=0):
        self.val = 0
        self.avg = 0
        self.sum = offset
        self.count = 0
        self.offset = offset

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = (self.sum - self.offset) / (self.count + 1E-3)

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        return False
    else:
        return True

class AverageConsole(object):
    def __init__(self,split,iters):
        self.split = split
        self.max_iters = iters
        self.meter = AverageMeter(time_millis())
        self.snapshot = 0

    def snap(self):
        self.snapshot = time_millis() - self.meter.sum

    def updateprint(self,i):
        self.meter.update(time_millis() - self.meter.sum)

        p = (i+1) * 100 / self.max_iters
        t_load = self.snapshot
        t_run = self.meter.val - self.snapshot
        total = (self.meter.sum - self.meter.offset ) / 1000

        if not i:
            sys.stdout.flush()
            print('')
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')

        msg = self.split.ljust(5) + ' [' + str(int(p)).rjust(3) + '%,' + str(int(total)).rjust(4) + 'sec]' + \
              ' < ' + str(i).rjust(4) + '-it' + \
              ' (load: ' + str(int(t_load)) + 'ms, run: ' + str(int(t_run)) + 'ms, avg: ' + str(int(self.meter.avg)) + 'ms)'
        print(msg)

def time_millis():
    return int(round(time.time() * 1000))

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, "a")

    def __del__(self):
        sys.stdout = self.terminal
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

class EarlyStop(object):
    def __init__(self, patience, aim='minimum'):
        self.patience = patience
        self.counter = 0
        self.aim = aim

        # aiming a minimum use a high value as init, low value otherwise
        if self.aim == 'minimum':
            self.best_score = 1E10
        else:
            self.best_score = -1E10

    def step(self, score):

        should_stop = score < self.best_score
        if self.aim == 'minimum':
            should_stop = not should_stop

        if should_stop:
            self.counter += 1
        else:
            self.counter = 0
            self.best_score = score

        return self.patience <= self.counter