"""
one_euro.py
====================
Helps optimise hand detection code

Reference: github by casiez
Link: https://github.com/casiez/OneEuroFilter/blob/main/python/OneEuroFilter/OneEuroFilter.py
"""
# Adding libraries
import math
import time

# Set up low pass filter
class LowPassFilter:
    def __init__(self, alpha):
        self.__y = None
        self.__alpha = alpha

    def __call__(self, value, alpha=None):
        if alpha is not None:
            self.__alpha = alpha
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__y
        self.__y = s
        return s

# Optimised filter  
class EuroOne:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.__freq = freq
        self.__mincutoff = mincutoff
        self.__beta = beta
        self.__dcutoff = dcutoff
        self.__x = LowPassFilter(self.__alpha(mincutoff))
        self.__dx = LowPassFilter(self.__alpha(dcutoff))
        self.__last_time = None

    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        
        if self.__last_time is not None:
            dt = timestamp - self.__last_time
            if dt > 0:
                self.__freq = 1.0 / dt
        self.__last_time = timestamp

        # Filter the derivative (velocity) to estimate speed
        prev_x = self.__x._LowPassFilter__y
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq
        edx = self.__dx(dx, self.__alpha(self.__dcutoff))
        
        # Use speed to update the cutoff frequency dynamically
        cutoff = self.__mincutoff + self.__beta * abs(edx)
        return self.__x(x, self.__alpha(cutoff))


landmark_filters = {}
LANDMARK_NUM = 21
INITIAL_FREQ = 30.0 
MIN_CUTOFF = 0.5     
BETA = 0.05  
# --- Creates filters for each landmark in each hand -----------------
def create_landmark_filters():
    return {
        i: {
            "x": EuroOne(freq=INITIAL_FREQ, mincutoff=MIN_CUTOFF, beta=BETA),
            "y": EuroOne(freq=INITIAL_FREQ, mincutoff=MIN_CUTOFF, beta=BETA),
            "z": EuroOne(freq=INITIAL_FREQ, mincutoff=MIN_CUTOFF, beta=BETA)
        }
        for i in range(LANDMARK_NUM)
    }

# --- Optimises each hands landmark detections ----------------
# Stops jitters and good for overlap, still bugs with wrist joint tho
def optimise_landmarks(result):
    time_stamp = time.time()
    if result.hand_landmarks:
        # Loop through every detected point
        for h_index, hand in enumerate(result.hand_landmarks):
            if h_index not in landmark_filters:
                landmark_filters[h_index] = create_landmark_filters()
            
            for landmark_index, landmark in enumerate(hand):
                if landmark_index in landmark_filters[h_index]:
                    # Filter each axis independently using the current time
                    filtered_x = landmark_filters[h_index][landmark_index]['x'](landmark.x, time_stamp)
                    filtered_y = landmark_filters[h_index][landmark_index]['y'](landmark.y, time_stamp)
                    filtered_z = landmark_filters[h_index][landmark_index]['z'](landmark.z, time_stamp)

                    # Overwrite raw coordinates with smooth coordinates
                    landmark.x = filtered_x
                    landmark.y = filtered_y
                    landmark.z = filtered_z
    return result