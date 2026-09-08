"""Small, reusable hover tooltips for the Tk interfaces."""

import tkinter as tk


#: Text of every one-shot tooltip already shown in this run of the application.
#: Keyed by text rather than by widget so that a hint does not reappear each
#: time a short-lived window (such as the mini-clicker) is rebuilt.
_ONCE_SHOWN = set()


def reset_once_shown():
    """Allow one-shot tooltips to be shown again."""
    _ONCE_SHOWN.clear()


class Tooltip:
    """Display explanatory text after the pointer rests over a widget."""

    def __init__(self, widget, text, delay_ms=500, wraplength=360, once=False):
        """Attach a tooltip to ``widget``.

        Set ``once`` for a hint on a working surface such as an image canvas,
        where the pointer rests constantly: it is shown the first time in a run
        and then stays out of the way. Such text belongs somewhere permanently
        visible as well, since a hint shown once is easily missed.
        """
        if not text or not text.strip():
            raise ValueError("Tooltip text must not be empty.")
        if delay_ms < 0:
            raise ValueError("Tooltip delay must not be negative.")

        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self.once = once
        self._after_id = None
        self._window = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._destroy, add="+")

    def _schedule(self, _event=None):
        if self.once and self.text in _ONCE_SHOWN:
            return
        self._cancel_scheduled()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_scheduled(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._window is not None or not self.widget.winfo_exists():
            return
        if self.once:
            _ONCE_SHOWN.add(self.text)
        self._window = self._create_window()

    def _create_window(self):
        """Build the pop-up holding this tooltip's text."""
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(
            f"+{self.widget.winfo_pointerx() + 14}+{self.widget.winfo_pointery() + 12}"
        )
        tk.Label(
            window,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            background="#ffffe0",
            foreground="#111111",
            padx=6,
            pady=4,
            wraplength=self.wraplength,
        ).pack()
        return window

    def _hide(self, _event=None):
        self._cancel_scheduled()
        if self._window is not None:
            self._window.destroy()
            self._window = None

    def _destroy(self, _event=None):
        self._cancel_scheduled()
        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
        self._window = None


def add_tooltip(widget, text, **options):
    """Attach a tooltip and return ``widget`` for concise UI construction."""
    widget._cellclicker_tooltip = Tooltip(widget, text, **options)
    return widget
