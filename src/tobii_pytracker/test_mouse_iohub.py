from psychopy import core, visual
from psychopy.iohub import launchHubServer

window = visual.Window(
    size=(1000, 700),
    units="pix",
    fullscr=False,
)

try:
    io = launchHubServer(
        **{
            "eyetracker.hw.mouse.EyeTracker": {
                "name": "tracker",
                "enable": True,
                "save_events": True,
                "stream_events": True,
                "event_buffer_length": 1024,
                "calibration": {
                    "enable": False,
                },
                "runtime_settings": {
                    "sampling_rate": 50,
                    "track_eyes": "RIGHT_EYE",
                },
                "controls": {
                    "move": "RIGHT_BUTTON",
                    "blink": ["LEFT_BUTTON"],
                    "saccade_threshold": 0.5,
                },
                "monitor_event_types": [
                    "MonocularEyeSampleEvent",
                    "FixationStartEvent",
                    "FixationEndEvent",
                    "SaccadeStartEvent",
                    "SaccadeEndEvent",
                ],
            }
        },
        window=window,
    )

    tracker = io.devices.tracker
    tracker.setRecordingState(True)

    print("Mouse tracker started:", tracker)

    clock = core.Clock()
    while clock.getTime() < 5:
        events = tracker.getEvents()
        if events:
            print("Events:", len(events), events[-1])
        core.wait(0.05)

finally:
    if "io" in locals():
        io.quit()
    window.close()