"""Session recorder stub."""

class SessionRecorder:
    def __init__(self):
        self.records = []

    def record(self, data):
        self.records.append(data)
