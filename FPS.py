import time

class GETFPS:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.start_time = time.time()
        self.frame_count = 0
        self.current_fps = None  # Will store the last calculated FPS
        self.is_first = True

    def get_fps(self):
        end_time = time.time()
        # If first frame processed, reset start time for accurate timing
        if self.is_first:
            self.start_time = end_time
            self.is_first = False

        self.frame_count += 1
        elapsed = end_time - self.start_time

        if elapsed > 5:
            # Calculate FPS for the last 5 seconds
            fps = self.frame_count / 5.0
            self.current_fps = fps
            print("**********************FPS*****************************************")
            print("Fps of stream", self.stream_id, "is", fps)
            # Reset for next measurement period
            self.frame_count = 0
            self.start_time = end_time
            return self.current_fps
        else:
            # Not enough time has passed to calculate a new FPS value
            return self.current_fps

    def print_data(self):
        print('frame_count=', self.frame_count)
        print('start_time=', self.start_time)
        print('current_fps=', self.current_fps)
