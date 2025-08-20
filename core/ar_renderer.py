"""AR renderer stub."""

class ARRenderer:
    def __init__(self):
        self.running = False

    def start(self):
        self.running = True
        print("ARRenderer started (stub)")

    def stop(self):
        self.running = False
        print("ARRenderer stopped (stub)")
