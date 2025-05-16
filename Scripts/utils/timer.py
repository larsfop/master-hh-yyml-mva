import time


class Timer(object):
    def __init__(self, start_msg: str = None, exit_msg: str = None):
        self.start_msg = start_msg
        self.exit_msg = exit_msg
    
    def __enter__(self):
        if self.start_msg != None:
            print(self.start_msg)
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self.exit_msg == None:
            print(f'Time: {(time.time() - self.start):.2f}s')
        else:
            print(f'{self.exit_msg} | Time: {(time.time() - self.start):.2f}s')
            
    def set_msg(self, msg: str):
        self.exit_msg = msg



if __name__=="__main__":
    with Timer():
        time.sleep(2)