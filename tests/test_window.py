import os
from unittest.mock import MagicMock

import pytest

from tobii_pytracker.utils import gui

class FakeRect:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.draw = MagicMock()


class FakeTextStim:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)
        self.draw = MagicMock()
        self.boundingBox = kwargs.get(
            "boundingBox",
            (100, 50),
        )


class FakeCircle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.draw = MagicMock()

@pytest.fixture
def mocked_psychopy(monkeypatch):
    """
    Patch the PsychoPy objects as imported by gui.py.

    The project test bootstrap already provides PsychoPy test doubles, so
    patching sys.modules here would create different objects from the ones
    actually used by gui.py.
    """

    rect_mock = MagicMock(side_effect=FakeRect)
    text_mock = MagicMock(side_effect=FakeTextStim)
    circle_mock = MagicMock(side_effect=FakeCircle)
    window_mock = MagicMock()
    monitor_mock = MagicMock()
    wait_mock = MagicMock()
    wait_keys_mock = MagicMock()

    monkeypatch.setattr(gui.visual, "Rect", rect_mock)
    monkeypatch.setattr(gui.visual, "TextStim", text_mock)
    monkeypatch.setattr(gui.visual, "Circle", circle_mock)
    monkeypatch.setattr(gui.visual, "Window", window_mock)
    monkeypatch.setattr(gui.monitors, "Monitor", monitor_mock)
    monkeypatch.setattr(gui.core, "wait", wait_mock)
    monkeypatch.setattr(gui.event, "waitKeys", wait_keys_mock)

    return {
        "Rect": rect_mock,
        "TextStim": text_mock,
        "Circle": circle_mock,
        "Window": window_mock,
        "Monitor": monitor_mock,
        "wait": wait_mock,
        "waitKeys": wait_keys_mock,
    }

def test_prepare_monitor(mocked_psychopy):
    config = MagicMock()

    config.get_monitor_config.return_value = {
        "name": "test-monitor",
        "width": 53.0,
        "distance": 70.0,
        "resolution": [1920, 1080],
    }

    monitor = MagicMock()
    mocked_psychopy["Monitor"].return_value = monitor

    result = gui.prepare_monitor(config)

    mocked_psychopy["Monitor"].assert_called_once_with(
        name="test-monitor"
    )

    monitor.setWidth.assert_called_once_with(53.0)
    monitor.setDistance.assert_called_once_with(70.0)
    monitor.setSizePix.assert_called_once_with(
        [1920, 1080]
    )
    monitor.saveMon.assert_called_once_with()

    assert result is monitor

def test_prepare_window_uses_monitor_and_display_config(
    mocked_psychopy,
):
    config = MagicMock()

    config.get_monitor_config.return_value = {
        "name": "test-monitor",
        "display_number": 2,
    }

    monitor = MagicMock()
    monitor.name = "test-monitor"
    monitor.getSizePix.return_value = [
        1920,
        1080,
    ]

    window = MagicMock()
    mocked_psychopy["Window"].return_value = window

    result = gui.prepare_window(
        config,
        monitor,
    )

    mocked_psychopy["Window"].assert_called_once_with(
        fullscr=True,
        size=[1920, 1080],
        checkTiming=False,
        winType="pyglet",
        allowGUI=True,
        allowStencil=False,
        monitor="test-monitor",
        color="black",
        screen=2,
        colorSpace="rgb",
        blendMode="avg",
        useFBO=True,
        units="pix",
    )

    assert result is window


def test_prepare_window_defaults_display_number_to_zero(
    mocked_psychopy,
):
    config = MagicMock()

    config.get_monitor_config.return_value = {
        "name": "test-monitor",
    }

    monitor = MagicMock()
    monitor.name = "test-monitor"
    monitor.getSizePix.return_value = [
        1280,
        720,
    ]

    gui.prepare_window(
        config,
        monitor,
    )

    kwargs = mocked_psychopy["Window"].call_args.kwargs

    assert kwargs["screen"] == 0
    assert kwargs["size"] == [
        1280,
        720,
    ]

def test_create_button_creates_rect_and_text(
    mocked_psychopy,
):
    window = MagicMock()

    result = gui.create_button(
        window=window,
        width=200,
        height=80,
        position=(10, 20),
        text="YES",
        label="yes",
        fill_color="green",
        text_size=30,
        text_color="white",
    )

    assert len(result) == 3

    rect, text, label = result

    assert isinstance(rect, FakeRect)
    assert isinstance(text, FakeTextStim)
    assert label == "yes"

    mocked_psychopy["Rect"].assert_called_once_with(
        win=window,
        width=200,
        height=80,
        pos=(10, 20),
        fillColor="green",
        lineColor="white",
        lineWidth=3,
    )

    mocked_psychopy["TextStim"].assert_called_once_with(
        win=window,
        text="YES",
        pos=(10, 20),
        height=30,
        color="white",
        bold=True,
    )


def test_create_button_accepts_custom_line_settings(
    mocked_psychopy,
):
    window = MagicMock()

    gui.create_button(
        window,
        100,
        50,
        (0, 0),
        "EXIT",
        "functional_quit",
        "red",
        20,
        "white",
        line_color="yellow",
        line_width=7,
        bold=False,
    )

    rect_kwargs = mocked_psychopy["Rect"].call_args.kwargs
    text_kwargs = mocked_psychopy["TextStim"].call_args.kwargs

    assert rect_kwargs["lineColor"] == "yellow"
    assert rect_kwargs["lineWidth"] == 7
    assert text_kwargs["bold"] is False

def test_fit_text_to_area_returns_initial_size_when_it_fits(
    mocked_psychopy,
):
    window = MagicMock()

    mocked_psychopy["TextStim"].side_effect = lambda *args, **kwargs: (
        FakeTextStim(
            *args,
            boundingBox=(100, 40),
            **kwargs,
        )
    )

    result = gui.fit_text_to_area(
        window,
        "HELLO",
        max_width=200,
        max_height=100,
        initial_size=40,
    )

    assert result == 40
    assert mocked_psychopy["TextStim"].call_count == 1


def test_fit_text_to_area_reduces_size_until_text_fits(
    mocked_psychopy,
):
    window = MagicMock()

    created_sizes = []

    def make_text(*args, **kwargs):
        size = kwargs["height"]
        created_sizes.append(size)

        if size > 30:
            bbox = (250, 50)
        else:
            bbox = (100, 50)

        return FakeTextStim(
            *args,
            boundingBox=bbox,
            **kwargs,
        )

    mocked_psychopy["TextStim"].side_effect = make_text

    result = gui.fit_text_to_area(
        window,
        "LONG TEXT",
        max_width=200,
        max_height=100,
        initial_size=35,
        min_size=10,
    )

    assert result == 30

    assert created_sizes == [
        35,
        34,
        33,
        32,
        31,
        30,
    ]


def test_fit_text_to_area_returns_min_size_when_nothing_fits(
    mocked_psychopy,
):
    window = MagicMock()

    mocked_psychopy["TextStim"].side_effect = (
        lambda *args, **kwargs: FakeTextStim(
            *args,
            boundingBox=(1000, 1000),
            **kwargs,
        )
    )

    result = gui.fit_text_to_area(
        window,
        "VERY LONG TEXT",
        max_width=100,
        max_height=50,
        initial_size=12,
        min_size=10,
    )

    assert result == 10


def test_fit_text_to_area_checks_both_width_and_height(
    mocked_psychopy,
):
    window = MagicMock()

    def make_text(*args, **kwargs):
        return FakeTextStim(
            *args,
            boundingBox=(50, 101),
            **kwargs,
        )

    mocked_psychopy["TextStim"].side_effect = make_text

    result = gui.fit_text_to_area(
        window,
        "TEXT",
        max_width=100,
        max_height=100,
        initial_size=20,
        min_size=10,
    )

    assert result == 10


def test_fit_text_to_area_initial_size_below_min(
    mocked_psychopy,
):
    window = MagicMock()

    mocked_psychopy["TextStim"].side_effect = (
        lambda *args, **kwargs: FakeTextStim(
            *args,
            boundingBox=(1000, 1000),
            **kwargs,
        )
    )

    result = gui.fit_text_to_area(
        window,
        "TEXT",
        max_width=100,
        max_height=100,
        initial_size=5,
        min_size=10,
    )

    assert result == 10

def test_prepare_buttons_creates_one_button_per_class_plus_exit(
    mocked_psychopy,
    monkeypatch,
):
    window = MagicMock()
    window.size = (
        1000,
        800,
    )

    dataset = MagicMock()
    dataset.get_classes.return_value = [
        "Cat",
        "Dog",
        "Bird",
    ]

    config = MagicMock()

    config.get_button_config.return_value = {
        "size": (
            200,
            100,
        ),
        "color": "blue",
        "text": {
            "color": "white",
            "size": 40,
        },
    }

    fit_mock = MagicMock(
        return_value=30
    )

    create_mock = MagicMock(
        side_effect=lambda **kwargs: (
            kwargs["text"],
            kwargs["label"],
            kwargs["position"],
        )
    )

    monkeypatch.setattr(
        gui,
        "fit_text_to_area",
        fit_mock,
    )

    monkeypatch.setattr(
        gui,
        "create_button",
        create_mock,
    )

    buttons = gui.prepare_buttons(
        config,
        window,
        dataset,
    )

    assert len(buttons) == 4
    assert create_mock.call_count == 4
    assert fit_mock.call_count == 4

    first_call = create_mock.call_args_list[0].kwargs

    assert first_call["text"] == "CAT"
    assert first_call["label"] == "cat"
    assert first_call["fill_color"] == "blue"
    assert first_call["text_color"] == "white"
    assert first_call["text_size"] == 30

    exit_call = create_mock.call_args_list[-1].kwargs

    assert exit_call["text"] == "EXIT"
    assert exit_call["label"] == "functional_quit"
    assert exit_call["fill_color"] == "red"
    assert exit_call["text_color"] == "white"


def test_prepare_buttons_converts_class_names_to_strings(
    mocked_psychopy,
    monkeypatch,
):
    window = MagicMock()
    window.size = (
        800,
        600,
    )

    dataset = MagicMock()
    dataset.get_classes.return_value = [
        1,
        2,
    ]

    config = MagicMock()

    config.get_button_config.return_value = {
        "size": (
            200,
            80,
        ),
        "color": "blue",
        "text": {
            "color": "white",
            "size": 30,
        },
    }

    monkeypatch.setattr(
        gui,
        "fit_text_to_area",
        MagicMock(return_value=20),
    )

    create_mock = MagicMock(
        return_value=MagicMock()
    )

    monkeypatch.setattr(
        gui,
        "create_button",
        create_mock,
    )

    gui.prepare_buttons(
        config,
        window,
        dataset,
    )

    first_button = create_mock.call_args_list[0].kwargs
    second_button = create_mock.call_args_list[1].kwargs

    assert first_button["text"] == "1"
    assert first_button["label"] == "1"

    assert second_button["text"] == "2"
    assert second_button["label"] == "2"


def test_prepare_buttons_handles_single_class(
    mocked_psychopy,
    monkeypatch,
):
    window = MagicMock()
    window.size = (
        500,
        400,
    )

    dataset = MagicMock()
    dataset.get_classes.return_value = [
        "Only"
    ]

    config = MagicMock()

    config.get_button_config.return_value = {
        "size": (
            150,
            60,
        ),
        "color": "blue",
        "text": {
            "color": "white",
            "size": 25,
        },
    }

    monkeypatch.setattr(
        gui,
        "fit_text_to_area",
        MagicMock(return_value=20),
    )

    monkeypatch.setattr(
        gui,
        "create_button",
        MagicMock(return_value=MagicMock()),
    )

    buttons = gui.prepare_buttons(
        config,
        window,
        dataset,
    )

    assert len(buttons) == 2


def test_prepare_buttons_with_no_classes_raises_zero_division_error(
    mocked_psychopy,
):
    window = MagicMock()
    window.size = (
        1000,
        800,
    )

    dataset = MagicMock()
    dataset.get_classes.return_value = []

    config = MagicMock()

    config.get_button_config.return_value = {
        "size": (
            200,
            100,
        ),
        "color": "blue",
        "text": {
            "color": "white",
            "size": 40,
        },
    }

    with pytest.raises(ZeroDivisionError):
        gui.prepare_buttons(
            config,
            window,
            dataset,
        )


def test_prepare_buttons_limits_button_width_to_available_space(
    mocked_psychopy,
    monkeypatch,
):
    window = MagicMock()
    window.size = (
        1000,
        800,
    )

    dataset = MagicMock()
    dataset.get_classes.return_value = [
        "A",
        "B",
        "C",
    ]

    config = MagicMock()

    config.get_button_config.return_value = {
        "size": (
            500,
            100,
        ),
        "color": "blue",
        "text": {
            "color": "white",
            "size": 30,
        },
    }

    create_mock = MagicMock(
        return_value=MagicMock()
    )

    monkeypatch.setattr(
        gui,
        "fit_text_to_area",
        MagicMock(return_value=20),
    )

    monkeypatch.setattr(
        gui,
        "create_button",
        create_mock,
    )

    gui.prepare_buttons(
        config,
        window,
        dataset,
    )

    for call in create_mock.call_args_list[:3]:
        assert call.kwargs["width"] == 300


def test_prepare_buttons_places_buttons_near_bottom(
    mocked_psychopy,
    monkeypatch,
):
    window = MagicMock()
    window.size = (
        1000,
        800,
    )

    dataset = MagicMock()
    dataset.get_classes.return_value = [
        "A",
        "B",
    ]

    config = MagicMock()

    config.get_button_config.return_value = {
        "size": (
            200,
            100,
        ),
        "color": "blue",
        "text": {
            "color": "white",
            "size": 30,
        },
    }

    create_mock = MagicMock(
        return_value=MagicMock()
    )

    monkeypatch.setattr(
        gui,
        "fit_text_to_area",
        MagicMock(return_value=20),
    )

    monkeypatch.setattr(
        gui,
        "create_button",
        create_mock,
    )

    gui.prepare_buttons(
        config,
        window,
        dataset,
    )

    first_button = create_mock.call_args_list[0].kwargs

    assert first_button["position"][1] == -300


def test_prepare_buttons_exit_button_is_top_right(
    mocked_psychopy,
    monkeypatch,
):
    window = MagicMock()
    window.size = (
        1200,
        800,
    )

    dataset = MagicMock()
    dataset.get_classes.return_value = [
        "A"
    ]

    config = MagicMock()

    config.get_button_config.return_value = {
        "size": (
            200,
            100,
        ),
        "color": "blue",
        "text": {
            "color": "white",
            "size": 30,
        },
    }

    create_mock = MagicMock(
        return_value=MagicMock()
    )

    monkeypatch.setattr(
        gui,
        "fit_text_to_area",
        MagicMock(return_value=20),
    )

    monkeypatch.setattr(
        gui,
        "create_button",
        create_mock,
    )

    gui.prepare_buttons(
        config,
        window,
        dataset,
    )

    exit_kwargs = create_mock.call_args_list[-1].kwargs

    assert exit_kwargs["position"] == (
        550,
        350,
    )

    assert exit_kwargs["width"] == 100
    assert exit_kwargs["height"] == 100
    assert exit_kwargs["text"] == "EXIT"
    assert exit_kwargs["label"] == "functional_quit"
    assert exit_kwargs["fill_color"] == "red"
    assert exit_kwargs["text_color"] == "white"

def test_save_screenshot_adds_png_extension_when_missing(
    tmp_path,
):
    config = MagicMock()

    config.get_monitor_config.return_value = {
        "resolution": (
            1920,
            1080,
        ),
    }

    config.get_area_of_interest_size.return_value = (
        800,
        600,
    )

    screenshot = MagicMock()

    window = MagicMock()
    window._getFrame.return_value = screenshot

    data = {
        "id": "sample_001"
    }

    result = gui.save_screenshot(
        config,
        window,
        data,
        str(tmp_path),
    )

    expected_path = os.path.join(
        str(tmp_path),
        "sample_001.png",
    )

    assert result == expected_path

    window._getFrame.assert_called_once_with(
        buffer="front"
    )

    screenshot.crop.assert_called_once_with(
        (
            560.0,
            240.0,
            1360.0,
            840.0,
        )
    )

    cropped = screenshot.crop.return_value

    cropped.save.assert_called_once_with(
        expected_path,
        "PNG",
    )


@pytest.mark.parametrize(
    "filename",
    [
        "image.png",
        "image.jpg",
        "image.jpeg",
        "image.bmp",
        "image.gif",
        "IMAGE.PNG",
        "IMAGE.JPG",
    ],
)
def test_save_screenshot_does_not_duplicate_known_image_extension(
    filename,
):
    config = MagicMock()

    config.get_monitor_config.return_value = {
        "resolution": (
            1000,
            800,
        ),
    }

    config.get_area_of_interest_size.return_value = (
        400,
        200,
    )

    screenshot = MagicMock()

    window = MagicMock()
    window._getFrame.return_value = screenshot

    result = gui.save_screenshot(
        config,
        window,
        {
            "id": filename
        },
        "/output",
    )

    assert result == os.path.join(
        "/output",
        filename,
    )


def test_save_screenshot_calculates_crop_box_correctly():
    config = MagicMock()

    config.get_monitor_config.return_value = {
        "resolution": (
            2000,
            1200,
        ),
    }

    config.get_area_of_interest_size.return_value = (
        1000,
        400,
    )

    screenshot = MagicMock()

    window = MagicMock()
    window._getFrame.return_value = screenshot

    gui.save_screenshot(
        config,
        window,
        {
            "id": "test"
        },
        "/tmp",
    )

    screenshot.crop.assert_called_once_with(
        (
            500.0,
            400.0,
            1500.0,
            800.0,
        )
    )


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("a.png", "a.png"),
        ("a.jpg", "a.jpg"),
        ("a.jpeg", "a.jpeg"),
        ("a.bmp", "a.bmp"),
        ("a.gif", "a.gif"),
        ("a.PNG", "a.PNG"),
        ("a.JpG", "a.JpG"),
        ("a.txt", "a.txt.png"),
        ("a", "a.png"),
    ],
)
def test_save_screenshot_extension_handling(
    filename,
    expected,
):
    config = MagicMock()

    config.get_monitor_config.return_value = {
        "resolution": (
            800,
            600,
        ),
    }

    config.get_area_of_interest_size.return_value = (
        400,
        300,
    )

    screenshot = MagicMock()

    window = MagicMock()
    window._getFrame.return_value = screenshot

    result = gui.save_screenshot(
        config,
        window,
        {
            "id": filename
        },
        "/output",
    )

    assert result == os.path.join(
        "/output",
        expected,
    )

def test_draw_window_draws_stimulus_buttons_and_saves_screenshot(
    mocked_psychopy,
    monkeypatch,
):
    window = MagicMock()
    window.size = (
        1000,
        800,
    )

    config = MagicMock()

    config.get_button_config.return_value = {
        "text": {
            "color": "white",
            "size": 30,
        }
    }

    config.get_fixation_dot_config.return_value = {
        "size": 10,
        "color": "white",
    }

    config.get_area_of_interest_size.return_value = (
        600,
        400,
    )

    dataset = MagicMock()

    dataset.draw_stimulus.return_value = [
        "bbox1",
        "bbox2",
    ]

    sample = {
        "id": "sample.png",
        "data": "some data",
        "class": "cat",
        "type": "image",
    }

    rect1 = FakeRect()
    text1 = FakeTextStim()

    rect2 = FakeRect()
    text2 = FakeTextStim()

    exit_rect = FakeRect()
    exit_text = FakeTextStim()

    buttons = [
        (
            rect1,
            text1,
            "cat",
        ),
        (
            rect2,
            text2,
            "dog",
        ),
        (
            exit_rect,
            exit_text,
            "functional_quit",
        ),
    ]

    save_mock = MagicMock(
        return_value="/tmp/sample.png"
    )

    monkeypatch.setattr(
        gui,
        "save_screenshot",
        save_mock,
    )

    result = gui.draw_window(
        config=config,
        window=window,
        sample=sample,
        dataset=dataset,
        buttons=buttons,
        focus_time=0.5,
        output_folder="/tmp",
    )

    assert result == (
        "/tmp/sample.png",
        ["bbox1", "bbox2"],
    )

    dataset.draw_stimulus.assert_called_once_with(
        window,
        sample,
    )

    save_mock.assert_called_once_with(
        config,
        window,
        sample,
        "/tmp",
    )

    mocked_psychopy["wait"].assert_called_once_with(
        0.5
    )

    assert window.flip.call_count == 1

    assert mocked_psychopy["Rect"].call_count == 1
    assert mocked_psychopy["Circle"].call_count == 1

    assert rect1.pos == pytest.approx(
        (
            -1000 / 2 + 1000 / 3,
            -800 / 2 + 50,
        )
    )

    assert rect2.pos == pytest.approx(
        (
            -1000 / 2 + 2 * (1000 / 3),
            -800 / 2 + 50,
        )
    )

    assert exit_rect.pos == pytest.approx(
        (
            1000 / 2 - 60,
            800 / 2 - 60,
        )
    )

    rect1.draw.assert_called_once()
    text1.draw.assert_called_once()

    rect2.draw.assert_called_once()
    text2.draw.assert_called_once()

    exit_rect.draw.assert_called_once()
    exit_text.draw.assert_called_once()


def test_draw_window_uses_default_border_width(
    mocked_psychopy,
    monkeypatch,
):
    window = MagicMock()

    window.size = (
        800,
        600,
    )

    config = MagicMock()

    config.get_button_config.return_value = {
        "text": {
            "color": "white",
            "size": 30,
        }
    }

    config.get_fixation_dot_config.return_value = {
        "size": 5,
        "color": "white",
    }

    config.get_area_of_interest_size.return_value = (
        400,
        300,
    )

    dataset = MagicMock()
    dataset.draw_stimulus.return_value = []

    buttons = [
        (
            FakeRect(),
            FakeTextStim(),
            "a",
        ),
        (
            FakeRect(),
            FakeTextStim(),
            "functional_quit",
        ),
    ]

    monkeypatch.setattr(
        gui,
        "save_screenshot",
        MagicMock(
            return_value="/tmp/out.png"
        ),
    )

    gui.draw_window(
        config,
        window,
        {
            "id": "test"
        },
        dataset,
        buttons,
        0,
        "/tmp",
    )

    rect_kwargs = mocked_psychopy["Rect"].call_args.kwargs

    assert rect_kwargs["width"] == 405
    assert rect_kwargs["height"] == 305


def test_show_instructions_uses_custom_key(
    mocked_psychopy,
):
    window = MagicMock()

    window.size = (
        1280,
        720,
    )

    gui.show_instructions(
        window,
        ["Press ENTER"],
        key="return",
    )

    mocked_psychopy["waitKeys"].assert_called_once_with(
        keyList=["return"]
    )

    kwargs = mocked_psychopy["TextStim"].call_args.kwargs

    assert kwargs["height"] == 720 / 25
    assert kwargs["wrapWidth"] == 1280 * 0.8
    assert kwargs["text"] == "Press ENTER"
    assert kwargs["color"] == "white"
    assert kwargs["alignText"] == "center"
    assert kwargs["pos"] == (0, 0)

    mocked_psychopy["wait"].assert_called_once_with(
        0.2
    )


def test_show_instructions_accepts_empty_text_lines(
    mocked_psychopy,
):
    window = MagicMock()

    window.size = (
        1000,
        800,
    )

    message = FakeTextStim(
        win=window,
        text="",
    )

    mocked_psychopy["TextStim"].return_value = message

    gui.show_instructions(
        window,
        [],
    )

    kwargs = mocked_psychopy["TextStim"].call_args.kwargs

    assert kwargs["text"] == ""
    assert kwargs["height"] == 800 / 25
    assert kwargs["wrapWidth"] == 1000 * 0.8

    mocked_psychopy["waitKeys"].assert_called_once_with(
        keyList=["space"]
    )

    mocked_psychopy["wait"].assert_called_once_with(
        0.2
    )


def test_show_instructions_joins_multiple_lines_with_double_newlines(
    mocked_psychopy,
):
    window = MagicMock()

    window.size = (
        800,
        600,
    )

    message = FakeTextStim(
        win=window,
        text="",
    )

    mocked_psychopy["TextStim"].return_value = message

    gui.show_instructions(
        window,
        [
            "Line one",
            "Line two",
            "Line three",
        ],
    )

    kwargs = mocked_psychopy["TextStim"].call_args.kwargs

    assert kwargs["text"] == (
        "Line one\n\n"
        "Line two\n\n"
        "Line three"
    )
