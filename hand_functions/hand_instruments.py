"""
hand_instrument.py
======================
Functions for arduino implementation

"""

# Adding dependencies
import time
from pymata4 import pymata4

# Arduino / buzzer setup 
BUZZER_PIN = 8                                 # the 'S' (signal) leg of the buzzer -> D8
board = pymata4.Pymata4()                      # auto-detects the Arduino's USB port

# Beep speeds (ms). 
INTERVAL_RIGHT = 0.40                           # right hand -> slow beeps
INTERVAL_LEFT  = 0.15                           # left hand  -> fast beeps
INTERVAL_BOTH  = 0.06                           # both hands -> fastest beeps idk

 # initial buzzer state
buzzer_is_on = False
last_toggle = time.time()


def buzz_detection(detected):

    board.set_pin_mode_digital_output(BUZZER_PIN)
    board.digital_write(BUZZER_PIN, 0)             # start silent

    global buzzer_is_on, last_toggle
    detected_R, detected_L = detected
    # link hand detection to buzzer state
    # Decide how fast to beep (or not at all)
    if detected_L and detected_R:
        interval = INTERVAL_BOTH        # both hands -> speedy
    elif detected_R:
        interval = INTERVAL_RIGHT       # right only -> slow
    elif detected_L:
        interval = INTERVAL_LEFT        # left only  -> fast
    else:
        interval = None                 # no hands   -> silence


    # Non-blocking beep: flip the pin only once enough time has passed,
    # so the video feed never freezes (no time.sleep!).
    now = time.time()
    if interval is None:
        if buzzer_is_on:                            # make sure it's off
            board.digital_write(BUZZER_PIN, 0)
            buzzer_is_on = False
    else:
        if now - last_toggle >= interval:           # time to flip on/off
            buzzer_is_on = not buzzer_is_on
            board.digital_write(BUZZER_PIN, 1 if buzzer_is_on else 0)
            last_toggle = now

