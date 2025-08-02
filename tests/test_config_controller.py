from pathlib import Path
from unittest import mock

from redness_tracker.controller.main_controller import MainController
from redness_tracker.model.config import Settings


def test_settings_load_save(tmp_path: Path) -> None:
    path = tmp_path / 'settings.yaml'
    settings = Settings(roi=(1, 2, 3, 4), use_max_red=True)
    settings.save(path)
    loaded = Settings.load(path)
    assert loaded == settings


def test_controller_toggle(monkeypatch) -> None:
    class DummyVP:
        def __init__(self, *args, **kwargs):
            pass
        def start(self) -> None:
            pass
        def join(self, timeout: float | None = None) -> None:
            pass
    monkeypatch.setattr('redness_tracker.controller.main_controller.VideoProcessor', DummyVP)
    monkeypatch.setattr('redness_tracker.controller.main_controller.configure_logging', lambda path: None)
    monkeypatch.setattr('redness_tracker.controller.main_controller.MainView', lambda root, handler: None)
    root = mock.MagicMock()
    controller = MainController(root)
    controller.handle_event('toggle_max_red', {'state': True})
    assert controller.settings.use_max_red is True

def test_configure_logging(tmp_path: Path) -> None:
    from redness_tracker.model.config import configure_logging
    import logging

    configure_logging(tmp_path)
    logging.getLogger(__name__).info("hello")
    assert (tmp_path / "app.log").exists()
