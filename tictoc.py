class ticktock():
    from time import time
    def __index__(self):
        self.clock = 0
        self.start = 0
        self.end = 0
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.total = 0
    def click(self):
        self.start = self.time()

    def stop(self):
        if (self.start > 0):
            self.end = self.time()
            diff = int(self.end - self.start)
            self.hours, self.minutes, self.seconds, self.total = diff // 3600, diff // 60, diff % 60, diff
        else:
            print("Start the clock with click()")

    def getTime(self):
        self.stop()
        return self.hours, self.minutes, self.seconds
    def getNow(self):
        self.stop()
        return self.total
    def reset(self):
        self.start = 0
        self.end = 0
        self.hour = 0
        self.minute = 0
        self.second = 0

