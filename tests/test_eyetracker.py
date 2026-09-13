import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

psychopy = ModuleType("psychopy")
iohub = ModuleType("psychopy.iohub")
iohub.launchHubServer = MagicMock()

psychopy.iohub = iohub

sys.modules["psychopy"] = psychopy
sys.modules["psychopy.iohub"] = iohub


import tobii_pytracker.utils.eyetracker as eyetracker

class MockConfig:

    def __init__(self, width=2.0, height=2.0):
        self.width = width
        self.height = height

    def get_area_of_interest_size(self):
        return self.width, self.height


class EyeSampleEvent:
    """Simple fake binocular PsychoPy eye sample event."""

    def __init__(
        self,
        event_id=1,
        logged_time=123.4,
        left_gaze_x=None,
        left_gaze_y=None,
        right_gaze_x=None,
        right_gaze_y=None,
        left_pupil_measure1=None,
        right_pupil_measure1=None,
    ):
        self.event_id = event_id
        self.logged_time = logged_time

        self.left_gaze_x = left_gaze_x
        self.left_gaze_y = left_gaze_y
        self.right_gaze_x = right_gaze_x
        self.right_gaze_y = right_gaze_y

        self.left_pupil_measure1 = left_pupil_measure1
        self.right_pupil_measure1 = right_pupil_measure1


class MonocularEyeSampleEvent:
    """Simple fake monocular PsychoPy eye sample event."""

    def __init__(
        self,
        event_id=1,
        time=123.4,
        gaze_x=None,
        gaze_y=None,
        pupil_measure1=None,
        pupil_measure2=None,
    ):
        self.event_id = event_id
        self.time = time

        self.gaze_x = gaze_x
        self.gaze_y = gaze_y

        self.pupil_measure1 = pupil_measure1
        self.pupil_measure2 = pupil_measure2

class TestGetTrackerClass:

    def test_returns_eyetracker_class_key(self):
        config = {
            "eyetracker.hw.tobii.EyeTracker": {
                "name": "Tobii"
            }
        }

        result = eyetracker.get_tracker_class(config)

        assert result == "eyetracker.hw.tobii.EyeTracker"

    def test_finds_tracker_case_insensitively(self):
        config = {
            "EyeTracker.HW.Mouse.EyeTracker": {
                "name": "Mouse"
            }
        }

        result = eyetracker.get_tracker_class(config)

        assert result == "EyeTracker.HW.Mouse.EyeTracker"

    def test_raises_when_no_tracker_exists(self):
        config = {
            "window": {},
            "display": {},
        }

        with pytest.raises(
            ValueError,
            match="No eyetracker configuration found"
        ):
            eyetracker.get_tracker_class(config)

class TestIsMouseEyetracker:

    def test_returns_true_for_mouse_tracker(self):
        config = {
            "eyetracker.hw.mouse.EyeTracker": {}
        }

        assert eyetracker.is_mouse_eyetracker(config) is True

    def test_returns_false_for_tobii_tracker(self):
        config = {
            "eyetracker.hw.tobii.EyeTracker": {}
        }

        assert eyetracker.is_mouse_eyetracker(config) is False

    def test_is_case_insensitive(self):
        config = {
            "eyetracker.hw.MOUSE.EyeTracker": {}
        }

        assert eyetracker.is_mouse_eyetracker(config) is True

class TestGetMouseMoveButtonIdx:

    @pytest.mark.parametrize(
        ("button", "expected"),
        [
            ("LEFT_BUTTON", 0),
            ("MIDDLE_BUTTON", 1),
            ("RIGHT_BUTTON", 2),
        ],
    )
    def test_returns_correct_button_index(self, button, expected):
        config = {
            "eyetracker.hw.mouse.EyeTracker": {
                "controls": {
                    "move": button
                }
            }
        }

        assert eyetracker.get_mouse_move_button_idx(config) == expected

    def test_button_name_is_case_insensitive(self):
        config = {
            "eyetracker.hw.mouse.EyeTracker": {
                "controls": {
                    "move": "left_button"
                }
            }
        }

        assert eyetracker.get_mouse_move_button_idx(config) == 0

    def test_uses_default_right_button_when_move_is_missing(self):
        config = {
            "eyetracker.hw.mouse.EyeTracker": {
                "controls": {}
            }
        }

        assert eyetracker.get_mouse_move_button_idx(config) == 2

    def test_uses_default_when_controls_are_missing(self):
        config = {
            "eyetracker.hw.mouse.EyeTracker": {}
        }

        assert eyetracker.get_mouse_move_button_idx(config) == 2

    def test_custom_default_is_used(self):
        config = {
            "eyetracker.hw.mouse.EyeTracker": {
                "controls": {
                    "move": "invalid_button"
                }
            }
        }

        assert eyetracker.get_mouse_move_button_idx(
            config,
            default="LEFT_BUTTON"
        ) == 0

    def test_invalid_button_falls_back_to_default(self):
        config = {
            "eyetracker.hw.mouse.EyeTracker": {
                "controls": {
                    "move": "INVALID_BUTTON"
                }
            }
        }

        assert eyetracker.get_mouse_move_button_idx(config) == 2

class TestPollTrackerEvents:

    def test_adds_new_events_to_buffer(self):
        tracker = MagicMock()

        event1 = SimpleNamespace(event_id=1)
        event2 = SimpleNamespace(event_id=2)

        tracker.getEvents.return_value = [event1, event2]

        buffer = {}

        result_buffer, last_id = eyetracker.poll_tracker_events(
            tracker,
            buffer
        )

        assert last_id == 2
        assert "SimpleNamespace" in result_buffer
        assert result_buffer["SimpleNamespace"] == [event1, event2]

    def test_ignores_duplicate_events(self):
        tracker = MagicMock()

        event1 = SimpleNamespace(event_id=1)
        event2 = SimpleNamespace(event_id=2)
        event3 = SimpleNamespace(event_id=3)

        tracker.getEvents.return_value = [event1, event2, event3]

        buffer = {}

        result_buffer, last_id = eyetracker.poll_tracker_events(
            tracker,
            buffer,
            last_event_id=2
        )

        assert last_id == 3
        assert result_buffer["SimpleNamespace"] == [event3]

    def test_ignores_events_without_event_id(self):
        tracker = MagicMock()

        event_without_id = SimpleNamespace(value=123)
        event_with_id = SimpleNamespace(event_id=5)

        tracker.getEvents.return_value = [
            event_without_id,
            event_with_id,
        ]

        buffer = {}

        result_buffer, last_id = eyetracker.poll_tracker_events(
            tracker,
            buffer
        )

        assert last_id == 5
        assert result_buffer["SimpleNamespace"] == [event_with_id]

    def test_returns_same_buffer_when_no_events(self):
        tracker = MagicMock()
        tracker.getEvents.return_value = []

        buffer = {
            "ExistingEvent": ["event"]
        }

        result_buffer, last_id = eyetracker.poll_tracker_events(
            tracker,
            buffer,
            last_event_id=10
        )

        assert result_buffer is buffer
        assert result_buffer == {
            "ExistingEvent": ["event"]
        }
        assert last_id == 10

    def test_handles_type_error_from_get_events(self):
        tracker = MagicMock()
        tracker.getEvents.side_effect = TypeError("bad event call")

        buffer = {}

        result_buffer, last_id = eyetracker.poll_tracker_events(
            tracker,
            buffer,
            last_event_id=10
        )

        assert result_buffer == {}
        assert last_id == 10

    def test_does_not_call_clear_events(self):
        tracker = MagicMock()

        tracker.getEvents.return_value = [
            SimpleNamespace(event_id=1)
        ]

        eyetracker.poll_tracker_events(tracker, {})

        tracker.clearEvents.assert_not_called()

    def test_preserves_existing_buffer(self):
        tracker = MagicMock()

        existing_event = SimpleNamespace(event_id=1)
        new_event = SimpleNamespace(event_id=2)

        tracker.getEvents.return_value = [new_event]

        buffer = {
            "SimpleNamespace": [existing_event]
        }

        result_buffer, last_id = eyetracker.poll_tracker_events(
            tracker,
            buffer,
            last_event_id=1
        )

        assert result_buffer["SimpleNamespace"] == [
            existing_event,
            new_event,
        ]
        assert last_id == 2

class TestExtractFullRawEvent:

    def test_extracts_all_public_non_callable_attributes(self):
        event = SimpleNamespace(
            event_id=42,
            gaze_x=0.25,
            gaze_y=-0.10,
            pupil=3.2,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_full_raw_event(
            buffer,
            system_time=999.0
        )

        assert len(result) == 1

        record = result[0]

        assert record["event_id"] == 42
        assert record["gaze_x"] == 0.25
        assert record["gaze_y"] == -0.10
        assert record["pupil"] == 3.2
        assert record["event_type"] == "EyeSampleEvent"
        assert record["system_time"] == 999.0

    def test_removes_events_from_buffer(self):
        event = SimpleNamespace(event_id=1)

        buffer = {
            "EyeSampleEvent": [event]
        }

        eyetracker.extract_full_raw_event(buffer, 123)

        assert buffer == {}

    def test_handles_multiple_event_types(self):
        event1 = SimpleNamespace(event_id=1)
        event2 = SimpleNamespace(event_id=2)

        buffer = {
            "EyeSampleEvent": [event1],
            "OtherEvent": [event2],
        }

        result = eyetracker.extract_full_raw_event(buffer, 123)

        assert len(result) == 2

        event_types = {record["event_type"] for record in result}

        assert event_types == {
            "EyeSampleEvent",
            "OtherEvent",
        }

        assert buffer == {}

    def test_handles_empty_buffer(self):
        buffer = {}

        result = eyetracker.extract_full_raw_event(buffer, 123)

        assert result == []
        assert buffer == {}

class TestExtractEyeGazeEvents:

    def test_extracts_binocular_event_inside_aoi(self):
        event = EyeSampleEvent(
            event_id=1,
            logged_time=100.0,
            left_gaze_x=0.2,
            left_gaze_y=0.1,
            right_gaze_x=0.4,
            right_gaze_y=0.3,
            left_pupil_measure1=3.0,
            right_pupil_measure1=4.0,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=200.0,
            config=MockConfig(width=2, height=2),
        )

        assert len(result) == 1

        record = result[0]

        assert record["event_type"] == "EyeSampleEvent"
        assert record["event_id"] == 1
        assert record["logged_time"] == 100.0
        assert record["system_time"] == 200.0

        assert record["gaze_x_left"] == 0.2
        assert record["gaze_y_left"] == 0.1
        assert record["gaze_x_right"] == 0.4
        assert record["gaze_y_right"] == 0.3

        assert record["pupil_left"] == 3.0
        assert record["pupil_right"] == 4.0

        assert record["avg_gaze_x"] == pytest.approx(0.3)
        assert record["avg_gaze_y"] == pytest.approx(0.2)
        assert record["avg_pupil_size"] == pytest.approx(3.5)

    def test_filters_binocular_event_outside_aoi(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=5.0,
            left_gaze_y=5.0,
            right_gaze_x=6.0,
            right_gaze_y=6.0,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=200,
            config=MockConfig(width=2, height=2),
        )

        assert result == []
        assert buffer == {}

    def test_boundary_of_aoi_is_included(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=1.0,
            left_gaze_y=1.0,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=200,
            config=MockConfig(width=2, height=2),
        )

        assert len(result) == 1

    def test_none_coordinates_are_not_inside_aoi(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=None,
            left_gaze_y=None,
            right_gaze_x=None,
            right_gaze_y=None,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=200,
            config=MockConfig(width=2, height=2),
        )

        assert result == []

    def test_uses_right_eye_when_left_eye_is_invalid(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=5.0,
            left_gaze_y=5.0,
            right_gaze_x=0.2,
            right_gaze_y=0.3,
            left_pupil_measure1=3.0,
            right_pupil_measure1=4.0,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=200,
            config=MockConfig(width=2, height=2),
        )

        assert len(result) == 1
        assert result[0]["avg_gaze_x"] == pytest.approx(2.6)
        assert result[0]["avg_gaze_y"] == pytest.approx(2.65)

    def test_monocular_event_is_extracted(self):
        event = MonocularEyeSampleEvent(
            event_id=10,
            time=50.0,
            gaze_x=0.25,
            gaze_y=-0.25,
            pupil_measure1=3.5,
            pupil_measure2=3.0,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=100.0,
            config=MockConfig(width=2, height=2),
        )

        assert len(result) == 1

        record = result[0]

        assert record["event_id"] == 10
        assert record["logged_time"] == 50.0

        assert record["gaze_x_left"] == 0.25
        assert record["gaze_y_left"] == -0.25

        assert record["gaze_x_right"] is None

        assert record["pupil_left"] == 3.5
        assert record["pupil_right"] == 3.0

        assert record["avg_gaze_x"] == 0.25
        assert record["avg_gaze_y"] == -0.25
        assert record["avg_pupil_size"] == pytest.approx(3.25)

    def test_uses_pupil_measure_fallback(self):
        class EyeSampleEventWithFallback:
            def __init__(self):
                self.event_id = 1
                self.logged_time = 10
                self.left_gaze_x = 0.1
                self.left_gaze_y = 0.1
                self.left_pupil_measure = 3.0
                self.right_gaze_x = 0.2
                self.right_gaze_y = 0.2
                self.right_pupil_measure = 4.0

        event = EyeSampleEventWithFallback()

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=20,
            config=MockConfig(),
        )

        assert len(result) == 1
        assert result[0]["pupil_left"] == 3.0
        assert result[0]["pupil_right"] == 4.0
        assert result[0]["avg_pupil_size"] == pytest.approx(3.5)

    def test_measure1_takes_precedence_over_fallback_measure(self):
        class EyeSampleEventWithBoth:
            def __init__(self):
                self.event_id = 1
                self.logged_time = 10
                self.left_gaze_x = 0.1
                self.left_gaze_y = 0.1
                self.left_pupil_measure1 = 3.0
                self.left_pupil_measure = 99.0

        event = EyeSampleEventWithBoth()

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=20,
            config=MockConfig(),
        )

        assert result[0]["pupil_left"] == 3.0

    def test_skips_non_eye_sample_events(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=0.1,
            left_gaze_y=0.1,
        )

        buffer = {
            "OtherEvent": [event],
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=100,
            config=MockConfig(),
        )

        assert result == []
        assert "OtherEvent" in buffer
        assert buffer["OtherEvent"] == [event]

    def test_removes_processed_eye_events_from_buffer(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=0.1,
            left_gaze_y=0.1,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=100,
            config=MockConfig(),
        )

        assert "EyeSampleEvent" not in buffer

    def test_event_without_event_id_is_kept_in_buffer(self):
        class EventWithoutId:
            left_gaze_x = 0.1
            left_gaze_y = 0.1

        event = EventWithoutId()

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=100,
            config=MockConfig(),
        )

        assert result == []
        assert "EyeSampleEvent" in buffer
        assert buffer["EyeSampleEvent"] == [event]

    def test_empty_buffer_returns_empty_list(self):
        buffer = {}

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=100,
            config=MockConfig(),
        )

        assert result == []
        assert buffer == {}

    def test_avg_gaze_uses_left_when_right_is_missing(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=0.4,
            left_gaze_y=0.6,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=100,
            config=MockConfig(),
        )

        record = result[0]

        assert record["avg_gaze_x"] == 0.4
        assert record["avg_gaze_y"] == 0.6

    def test_avg_gaze_uses_right_when_left_is_missing(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=None,
            left_gaze_y=None,
            right_gaze_x=0.4,
            right_gaze_y=0.6,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=100,
            config=MockConfig(),
        )

        record = result[0]

        assert record["avg_gaze_x"] == 0.4
        assert record["avg_gaze_y"] == 0.6

    def test_avg_pupil_uses_left_when_right_missing(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=0.1,
            left_gaze_y=0.1,
            left_pupil_measure1=3.2,
            right_pupil_measure1=None,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=100,
            config=MockConfig(),
        )

        assert result[0]["avg_pupil_size"] == 3.2

    def test_avg_pupil_uses_right_when_left_missing(self):
        event = EyeSampleEvent(
            event_id=1,
            left_gaze_x=0.1,
            left_gaze_y=0.1,
            left_pupil_measure1=None,
            right_pupil_measure1=4.2,
        )

        buffer = {
            "EyeSampleEvent": [event]
        }

        result = eyetracker.extract_eye_gaze_events(
            buffer,
            system_time=100,
            config=MockConfig(),
        )

        assert result[0]["avg_pupil_size"] == 4.2

class TestLaunchHubServer:

    def test_launches_server_and_starts_recording(self, monkeypatch):
        config = {
            "eyetracker.hw.tobii.EyeTracker": {
                "name": "Tobii"
            }
        }

        fake_tracker = MagicMock()
        fake_io = SimpleNamespace(
            devices=SimpleNamespace(
                tracker=fake_tracker
            )
        )

        launch_hub_server_mock = MagicMock(return_value=fake_io)

        monkeypatch.setattr(
            eyetracker,
            "launchHubServer",
            launch_hub_server_mock,
        )

        monkeypatch.setattr(
            eyetracker.CustomConfig,
            "read_config",
            MagicMock(return_value=config),
        )

        window = MagicMock()

        io, tracker = eyetracker.launch_hub_server(
            "config.yaml",
            window,
        )

        assert io is fake_io
        assert tracker is fake_tracker

        launch_hub_server_mock.assert_called_once_with(
            **config,
            window=window,
        )

        fake_tracker.runSetupProcedure.assert_called_once()
        fake_tracker.setRecordingState.assert_called_once_with(True)

    def test_launches_without_window_without_setup(self, monkeypatch):
        config = {
            "eyetracker.hw.tobii.EyeTracker": {
                "name": "Tobii"
            }
        }

        fake_tracker = MagicMock()
        fake_io = SimpleNamespace(
            devices=SimpleNamespace(
                tracker=fake_tracker
            )
        )

        launch_mock = MagicMock(return_value=fake_io)

        monkeypatch.setattr(
            eyetracker,
            "launchHubServer",
            launch_mock,
        )

        monkeypatch.setattr(
            eyetracker.CustomConfig,
            "read_config",
            MagicMock(return_value=config),
        )

        io, tracker = eyetracker.launch_hub_server(
            "config.yaml",
            None,
        )

        assert io is fake_io
        assert tracker is fake_tracker

        launch_mock.assert_called_once_with(**config)

        fake_tracker.runSetupProcedure.assert_not_called()
        fake_tracker.setRecordingState.assert_called_once_with(True)

    def test_window_is_added_to_launch_kwargs(self, monkeypatch):
        config = {
            "eyetracker.hw.tobii.EyeTracker": {
                "name": "Tobii"
            },
            "other": "value",
        }

        fake_io = SimpleNamespace(
            devices=SimpleNamespace(
                tracker=MagicMock()
            )
        )

        launch_mock = MagicMock(return_value=fake_io)

        monkeypatch.setattr(
            eyetracker,
            "launchHubServer",
            launch_mock,
        )

        monkeypatch.setattr(
            eyetracker.CustomConfig,
            "read_config",
            MagicMock(return_value=config),
        )

        window = object()

        eyetracker.launch_hub_server(
            "config.yaml",
            window,
        )

        launch_mock.assert_called_once_with(
            **config,
            window=window,
        )

    def test_tobii_startup_falls_back_to_mouse_config(
        self,
        monkeypatch,
        tmp_path,
    ):
        tobii_config = {
            "eyetracker.hw.tobii.EyeTracker": {
                "name": "Tobii"
            }
        }

        mouse_config = {
            "eyetracker.hw.mouse.EyeTracker": {
                "name": "Mouse"
            }
        }

        fake_tracker = MagicMock()
        fake_io = SimpleNamespace(
            devices=SimpleNamespace(
                tracker=fake_tracker
            )
        )

        launch_mock = MagicMock(
            side_effect=[
                RuntimeError("Tobii failed"),
                fake_io,
            ]
        )

        monkeypatch.setattr(
            eyetracker,
            "launchHubServer",
            launch_mock,
        )

        config_file = tmp_path / "tobii_config.yaml"
        fallback_file = tmp_path / "mouse_eyetracker_config.yaml"

        config_file.write_text("tobii")
        fallback_file.write_text("mouse")

        def read_config(path):
            if path == str(config_file):
                return tobii_config
            if path == str(fallback_file):
                return mouse_config
            raise AssertionError(f"Unexpected path: {path}")

        monkeypatch.setattr(
            eyetracker.CustomConfig,
            "read_config",
            read_config,
        )

        window = MagicMock()

        io, tracker = eyetracker.launch_hub_server(
            config_file,
            window,
        )

        assert io is fake_io
        assert tracker is fake_tracker

        assert launch_mock.call_count == 2

        first_call = launch_mock.call_args_list[0]
        second_call = launch_mock.call_args_list[1]

        assert first_call.kwargs == {
            **tobii_config,
            "window": window,
        }

        assert second_call.kwargs == {
            **mouse_config,
            "window": window,
        }

        fake_tracker.runSetupProcedure.assert_called_once()
        fake_tracker.setRecordingState.assert_called_once_with(True)

    def test_non_tobii_error_is_wrapped_and_reraised(
        self,
        monkeypatch,
        tmp_path,
    ):
        config = {
            "eyetracker.hw.mouse.EyeTracker": {
                "name": "Mouse"
            }
        }