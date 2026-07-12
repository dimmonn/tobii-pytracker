import yaml
from psychopy import core, monitors, visual
from psychopy.iohub import launchHubServer

with open("/Users/dima/tobii/tobii-pytracker/configs/mouse_eyetracker_config.yaml", "r") as file:
    config = yaml.safe_load(file)

monitor = monitors.Monitor("spectrum_monitor")
monitor.setSizePix((1920, 1080))
monitor.setWidth(35)
monitor.setDistance(60)

window = visual.Window(
    size=(1920, 1080),
    monitor=monitor,
    units="pix",
    fullscr=True,
)

io = None

try:
    print("Configuration being loaded:")
    print(config)

    io = launchHubServer(**config, window=window)

    print("Available devices:", io.devices)
    tracker = io.devices.tracker
    print("Tracker:", tracker)

    tracker.setRecordingState(True)
    print("Recording started")

    clock = core.Clock()

    while clock.getTime() < 10:
        events = tracker.getEvents()

        if events:
            print("Events received:", len(events))
            print(events[-1])

        core.wait(0.05)

finally:
    if io is not None:
        io.quit()

    window.close()