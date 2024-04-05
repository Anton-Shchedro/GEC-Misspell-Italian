import sys

class EliCoDe_init:
    started = False
    elicode_path = ''

    def __init__(self, path):
        self.elicode_path = path

    def start(self):
        sys.path.insert(1, self.elicode_path)
        self.started = True

    def end(self):
        if self.started == True:
            try:
                sys.path.remove(self.elicode_path)
                started = False
            except ValueError:
                print('Path is wrong. Try remove it manually')
        else:
            print("Elicode not started. Start with elicode_start(path)")
