import pytest

from CellClicker.tooltips import Tooltip, add_tooltip, reset_once_shown


class FakeWidget:
    def __init__(self):
        self.bindings = {}
        self.cancelled = []
        self.scheduled = []

    def bind(self, event, callback, add=None):
        self.bindings[event] = (callback, add)

    def after(self, delay, callback):
        self.scheduled.append((delay, callback))
        return "after-1"

    def after_cancel(self, after_id):
        self.cancelled.append(after_id)

    def winfo_exists(self):
        return True


@pytest.fixture(autouse=True)
def _forget_shown_hints():
    """One-shot state lives for the run, so tests must not inherit it."""
    reset_once_shown()
    yield
    reset_once_shown()


class FakeWindow:
    """Stand-in for the tooltip pop-up, which needs a real toolkit to build."""

    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def _hover(tooltip):
    """Drive one hover to completion without building a real pop-up window."""
    tooltip._create_window = FakeWindow
    tooltip._schedule()
    for _delay, callback in tooltip.widget.scheduled[-1:]:
        callback()
    tooltip._hide()


def test_add_tooltip_binds_hover_cleanup_and_returns_widget():
    widget = FakeWidget()

    returned = add_tooltip(widget, "Helpful text")

    assert returned is widget
    assert isinstance(widget._cellclicker_tooltip, Tooltip)
    assert set(widget.bindings) == {"<Enter>", "<Leave>", "<ButtonPress>", "<Destroy>"}
    assert all(add == "+" for _, add in widget.bindings.values())


def test_tooltip_schedules_and_cancels_delayed_display():
    widget = FakeWidget()
    tooltip = Tooltip(widget, "Helpful text", delay_ms=250)

    tooltip._schedule()
    tooltip._hide()

    assert widget.scheduled[0][0] == 250
    assert widget.cancelled == ["after-1"]


@pytest.mark.parametrize("text", ["", "   "])
def test_tooltip_rejects_empty_text(text):
    with pytest.raises(ValueError, match="must not be empty"):
        Tooltip(FakeWidget(), text)


def test_tooltip_rejects_negative_delay():
    with pytest.raises(ValueError, match="must not be negative"):
        Tooltip(FakeWidget(), "Helpful text", delay_ms=-1)


def test_a_one_shot_tooltip_is_shown_only_once_per_run():
    """The image canvases are worked over constantly; a repeating hint is noise."""
    tooltip = Tooltip(FakeWidget(), "Canvas guidance", once=True)

    _hover(tooltip)
    assert len(tooltip.widget.scheduled) == 1

    _hover(tooltip)
    assert len(tooltip.widget.scheduled) == 1, "the hint must not be scheduled again"


def test_a_one_shot_tooltip_stays_quiet_for_a_rebuilt_widget():
    """The mini-clicker is rebuilt for every track, so keying by widget is not enough."""
    first = Tooltip(FakeWidget(), "Canvas guidance", once=True)
    _hover(first)

    rebuilt = Tooltip(FakeWidget(), "Canvas guidance", once=True)
    rebuilt._schedule()

    assert rebuilt.widget.scheduled == []


def test_a_one_shot_tooltip_is_only_spent_when_it_is_actually_shown():
    """Passing the pointer over without resting must not use up the one showing."""
    tooltip = Tooltip(FakeWidget(), "Canvas guidance", once=True)

    tooltip._schedule()
    tooltip._hide()

    second = Tooltip(FakeWidget(), "Canvas guidance", once=True)
    second._schedule()
    assert len(second.widget.scheduled) == 1


def test_ordinary_tooltips_keep_repeating():
    """Buttons are visited briefly and deliberately, so their hints stay."""
    tooltip = Tooltip(FakeWidget(), "Button guidance")

    _hover(tooltip)
    _hover(tooltip)

    assert len(tooltip.widget.scheduled) == 2


def test_reset_once_shown_allows_the_hint_again():
    tooltip = Tooltip(FakeWidget(), "Canvas guidance", once=True)
    _hover(tooltip)

    reset_once_shown()
    tooltip._schedule()

    assert len(tooltip.widget.scheduled) == 2
