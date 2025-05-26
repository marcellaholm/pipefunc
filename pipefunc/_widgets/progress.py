from __future__ import annotations

import asyncio
import functools
import hashlib
import textwrap
import time
from typing import TYPE_CHECKING, Any

import IPython.display
import ipywidgets as widgets

from pipefunc._utils import at_least_tuple, clip

if TYPE_CHECKING:
    from collections.abc import Callable

    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._progress import Status


_IPYWIDGETS_MAJOR_VERSION = int(widgets.__version__.split(".")[0])


def _span(class_name: str, value: str) -> str:
    return f'<span class="{class_name}">{value}</span>'


def _create_button(
    description: str,
    button_style: str,
    icon: str,
    on_click: Callable[[Any], None],
) -> widgets.Button:
    button = widgets.Button(description=description, button_style=button_style, icon=icon)
    button.on_click(on_click)
    return button


def _create_progress_bar(name: OUTPUT_TYPE, progress: float) -> widgets.FloatProgress:
    description = ", ".join(at_least_tuple(name))
    styles = [
        "direction: rtl",  # Reverses text direction so ellipsis appears at the beginning
        "display: inline-block",  # Allows block-like behavior within the inline flow
        "width: 100%",  # Ensures span takes full available width
        "white-space: nowrap",  # Prevents text from wrapping to a new line
        "overflow: hidden",  # Hides text that extends beyond the container
        "text-overflow: ellipsis",  # Shows "..." when text is truncated
    ]
    style = "; ".join(styles)
    tooltip_key = "tooltip" if _IPYWIDGETS_MAJOR_VERSION >= 8 else "description_tooltip"  # noqa: PLR2004
    tooltip = {tooltip_key: description}
    return widgets.FloatProgress(
        value=progress,
        max=1.0,
        description=f"<span style='{style}'>{description}</span>",
        description_allow_html=True,
        layout={"width": "95%"},
        bar_style="info",
        style={"description_width": "150px"},
        **tooltip,
    )


def _create_html_label(class_name: str, initial_value: str) -> widgets.HTML:
    return widgets.HTML(value=_span(class_name, initial_value))


def _get_scope_hue(output_name: OUTPUT_TYPE) -> int | None:
    """Extract scope and calculate a consistent hue value from it."""
    output_name = at_least_tuple(output_name)
    all_have_scope = all("." in name for name in output_name)
    if not all_have_scope:
        return None

    scope = output_name[0].split(".")[0]
    # Convert string to int (0-255)
    hash_value = int(hashlib.md5(scope.encode()).hexdigest(), 16)  # noqa: S324
    return hash_value % 360


def _scope_border_color(hue: int | None) -> str:
    return f"hsl({hue}, 70%, 70%)" if hue is not None else "#999999"


def _scope_background_color_css(hue: int) -> str:
    return textwrap.dedent(
        f"""
        .scope-bg-{hue} {{
            background-color: hsla({hue}, 70%, 95%, 0.75);
        }}
        """,
    )


class ProgressTracker:
    """Class to track progress and display it with ipywidgets."""

    def __init__(
        self,
        progress_dict: dict[OUTPUT_TYPE, Status],
        task: asyncio.Task[Any] | None = None,
        *,
        target_progress_change: float = 0.05,
        auto_update: bool = True,
        display: bool = True,
        in_async: bool = True,
    ) -> None:
        self.task: asyncio.Task[None] | None = task
        self.progress_dict: dict[OUTPUT_TYPE, Status] = progress_dict
        self.target_progress_change: float = target_progress_change
        self.auto_update: bool = auto_update
        self.auto_update_task: asyncio.Task | None = None
        self.in_async: bool = in_async
        self.last_update_time: float = 0.0
        self._min_auto_update_interval: float = 0.1
        self._max_auto_update_interval: float = 10.0
        self._first_auto_update_interval: float = 1.0
        self._sync_update_interval: float = 0.01
        self.progress_bars: dict[OUTPUT_TYPE, widgets.FloatProgress] = {}
        self._progress_widgets: dict[
            OUTPUT_TYPE, widgets.Widget
        ] = {}  # Changed from _progress_vboxes
        self.labels: dict[OUTPUT_TYPE, dict[OUTPUT_TYPE, widgets.HTML]] = {}
        self.buttons: dict[OUTPUT_TYPE, widgets.Button] = {
            "update": _create_button(
                description="Update Progress",
                button_style="info",
                icon="refresh",
                on_click=self.update_progress,
            ),
            "toggle_auto_update": _create_button(
                description="Start Auto-Update",
                button_style="success",
                icon="refresh",
                on_click=self._toggle_auto_update,
            ),
            "cancel": _create_button(
                description="Cancel Calculation",
                button_style="danger",
                icon="stop",
                on_click=self._cancel_calculation,
            ),
        }
        for name, status in self.progress_dict.items():
            self.progress_bars[name] = _create_progress_bar(name, status.progress)
            self.labels[name] = {
                "percentage": _create_html_label("percent-label", f"{status.progress * 100:.1f}%"),
                "estimated_time": _create_html_label(
                    "estimate-label",
                    "Elapsed: 0.00 sec | ETA: Calculating...",
                ),
                "speed": _create_html_label("speed-label", "Speed: Calculating..."),
            }
            # Initially, store the VBox
            labels = self.labels[name]
            labels_box = widgets.HBox(
                [labels["percentage"], labels["estimated_time"], labels["speed"]],
                layout=widgets.Layout(justify_content="space-between"),
            )
            hue = _get_scope_hue(name)
            border_color = _scope_border_color(hue)
            border = f"1px solid {border_color}"
            container = widgets.VBox(
                [self.progress_bars[name], labels_box],
                layout=widgets.Layout(border=border, margin="2px 4px", padding="2px"),
            )
            container.add_class("progress-vbox")
            if hue is not None:
                container.add_class(f"scope-bg-{hue}")
            self._progress_widgets[name] = container

        self.auto_update_interval_label = _create_html_label(
            "interval-label",
            "Auto-update every: N/A",
        )
        self._initial_update_period: float = 30.0
        self._initial_max_update_interval: float = 1.0
        self.start_time: float = 0.0
        self._marked_completed: set[OUTPUT_TYPE] = (
            set()
        )  # Tracks items that have been *transformed*
        if display:
            self.display()
        if self.task is not None:
            self._set_auto_update(auto_update)

    def attach_task(self, task: asyncio.Task[Any]) -> None:
        """Attach a new task to the progress tracker."""
        self.task = task
        self._set_auto_update(self.auto_update)

    def _create_compact_widget(self, name: OUTPUT_TYPE, status: Status) -> widgets.HTML:
        output_name_str = ", ".join(at_least_tuple(name))
        icon = "✅" if status.n_failed == 0 else "❌"

        summary_text = f"{icon} {output_name_str} ({status.progress * 100:.0f}%) - Done: {status.elapsed_time():.1f}s"

        completed_str = f"{status.n_completed:,}"
        failed_str = f"{status.n_failed:,}"
        left_str = f"{status.n_left:,}"
        elapsed_s = status.elapsed_time()
        speed_val = status.n_attempted / elapsed_s if elapsed_s > 0 else float("inf")
        speed_str = f"{speed_val:,.2f}"

        tooltip_text = (
            f"Completed: {completed_str}, Failed: {failed_str}, Left: {left_str} | "
            f"Speed: {speed_str} iter/s | Elapsed: {elapsed_s:.2f}s"
        )

        hue = _get_scope_hue(name)
        border_color = _scope_border_color(hue)

        # Using a class that could be styled further by the user if needed
        html_value = _span("compact-summary-label", summary_text)

        compact_widget = widgets.HTML(value=html_value, tooltip=tooltip_text)
        compact_widget.layout = widgets.Layout(
            border=f"1px solid {border_color}",
            margin="2px 4px",
            padding="4px 8px",  # Slightly more padding for a single line
            width="auto",
            min_height="28px",  # Keep similar height to original progress bar
        )

        compact_widget.add_class("progress-vbox")  # For consistent general styling
        compact_widget.add_class("pulse-animation")
        if hue is not None:
            compact_widget.add_class(f"scope-bg-{hue}")

        return compact_widget

    def update_progress(self, _: Any = None, *, force: bool = False) -> None:
        """Update the progress values and labels."""
        now = time.monotonic()
        return_early = False
        if not self.in_async and not force:
            assert self.task is None
            if now - self.last_update_time < self._sync_update_interval:
                return_early = True

        for name, status in self.progress_dict.items():
            if status.progress == 0:  # Skip items that haven't started
                continue

            if name in self._marked_completed:  # If already compacted, skip
                continue

            if return_early and status.progress < 1.0:
                return

            if status.progress >= 1.0:
                # Item is complete, transform its widget
                compact_widget = self._create_compact_widget(name, status)

                # If _widgets (the main VBox) is already materialized, update its children
                if "_widgets" in self.__dict__ and hasattr(self._widgets, "children"):
                    children_list = list(self._widgets.children)
                    try:
                        # self._progress_widgets[name] should still hold the original VBox at this point
                        widget_index = children_list.index(self._progress_widgets[name])
                        children_list[widget_index] = compact_widget
                        self._widgets.children = tuple(children_list)
                    except ValueError:
                        # This might happen if the widget was already replaced due to rapid updates
                        # or if the structure of _widgets.children is not as expected.
                        # For this phase, we'll assume it's found or the _progress_widgets update handles it.
                        pass

                self._progress_widgets[name] = (
                    compact_widget  # This ensures future renders of _widgets use the compact form
                )
                self._marked_completed.add(name)  # Mark as transformed

                # Clean up no longer needed specific widgets for this item
                if name in self.progress_bars:
                    del self.progress_bars[name]
                if name in self.labels:
                    del self.labels[name]

            elif (
                name in self.progress_bars
            ):  # Check if widgets still exist (haven't been compacted)
                progress_bar = self.progress_bars[name]
                progress_bar.value = status.progress
                progress_bar.bar_style = (
                    "info"  # Reset to info, completion style handled by compact widget
                )
                progress_bar.remove_class("completed-progress")
                progress_bar.add_class("animated-progress")
                self._update_labels(name, status)

        if self._all_completed():
            self._mark_completed()
        self.last_update_time = time.monotonic()
        self._sync_update_interval = clip(50 * (self.last_update_time - now), 0.01, 1.0)

    def _update_labels(self, name: OUTPUT_TYPE, status: Status) -> None:
        assert status.progress > 0
        completed = f"✅ {status.n_completed:,}"
        failed = f"❌ {status.n_failed:,}"
        left = f"⏳ {status.n_left:,}"
        if status.n_failed == 0:
            iterations_label = f"{completed} | {left}"
        else:
            iterations_label = f"{completed} | {failed} | {left}"

        labels = self.labels[name]
        labels["percentage"].value = _span(
            "percent-label",
            f"{status.progress * 100:.1f}% | {iterations_label}",
        )

        elapsed_time = status.elapsed_time()
        if status.end_time is not None:
            eta = "Completed"
        else:
            estimated_time_left = (1.0 - status.progress) * (elapsed_time / status.progress)
            eta = f"ETA: {estimated_time_left:.2f} sec"
        speed = f"{status.n_attempted / elapsed_time:,.2f}" if elapsed_time > 0 else "∞"
        labels["speed"].value = _span("speed-label", f"Speed: {speed} iterations/sec")
        labels["estimated_time"].value = _span(
            "estimate-label",
            f"Elapsed: {elapsed_time:.2f} sec | {eta}",
        )

    def _calculate_adaptive_interval_with_previous(self) -> float:
        """Calculate a dynamic interval based on progress changes for all resources."""
        min_interval = self._min_auto_update_interval
        max_interval = self._max_auto_update_interval
        shortest_interval = max_interval
        current_time = time.monotonic()
        for status in self.progress_dict.values():
            if status.progress <= 0 or status.progress >= 1:
                continue
            assert status.start_time is not None
            elapsed_time = current_time - status.start_time
            progress_rate = status.progress / elapsed_time
            estimated_time_for_target = self.target_progress_change / progress_rate
            # Estimate time for target progress change
            shortest_interval = min(shortest_interval, estimated_time_for_target)
        return min(max(shortest_interval, min_interval), max_interval)

    async def _auto_update_progress(self) -> None:
        """Periodically update the progress."""
        self.start_time = time.monotonic()
        while self.auto_update:
            self.update_progress()
            current_time = time.monotonic()
            elapsed_since_start = current_time - self.start_time

            new_interval = self._calculate_adaptive_interval_with_previous()
            if elapsed_since_start <= self._initial_update_period:
                new_interval = min(new_interval, self._initial_max_update_interval)

            if self._all_completed():
                break

            self.auto_update_interval_label.value = _span(
                "interval-label",
                f"Auto-update every: {new_interval:.2f} sec",
            )
            await asyncio.sleep(new_interval)

    def _all_completed(self) -> bool:
        # Check if all tasks are *transformed* (marked_completed) or their progress is >= 1.0
        # This ensures that if a task completes but transformation hasn't run yet, it's still considered.
        return all(
            name in self._marked_completed or status.progress >= 1.0
            for name, status in self.progress_dict.items()
        )

    def _mark_completed(self) -> None:
        if self.auto_update:
            self._toggle_auto_update()
        if any(status.n_failed > 0 for status in self.progress_dict.values()):
            msg = "Completed with errors ❌"
        else:
            msg = "Completed all tasks 🎉"
        self.auto_update_interval_label.value = _span("interval-label", msg)
        for button in self.buttons.values():
            button.disabled = True

    def _toggle_auto_update(self, _: Any = None) -> None:
        """Toggle the auto-update feature on or off."""
        self._set_auto_update(not self.auto_update)

    def _set_auto_update(self, value: bool) -> None:  # noqa: FBT001
        """Set the auto-update feature to the given value."""
        self.auto_update = value
        self.buttons["toggle_auto_update"].description = (
            "Stop Auto-Update" if self.auto_update else "Start Auto-Update"
        )
        self.buttons["toggle_auto_update"].button_style = (
            "danger" if self.auto_update else "success"
        )
        if self.auto_update:
            self.auto_update_task = asyncio.create_task(self._auto_update_progress())
        elif self.auto_update_task is not None:
            self.auto_update_task.cancel()
            self.auto_update_task = None

    def _cancel_calculation(self, _: Any) -> None:
        """Cancel the ongoing calculation."""
        assert self.task is not None
        self.task.cancel()
        self.update_progress()  # Update progress one last time
        if self.auto_update:
            self._toggle_auto_update()
        for button in self.buttons.values():
            button.disabled = True

        # For items not yet compacted, mark their progress bars as cancelled/failed
        for name, progress_widget in self._progress_widgets.items():
            if name not in self._marked_completed and name in self.progress_bars:
                # Check if it's still a VBox with a progress bar
                if (
                    isinstance(progress_widget, widgets.VBox)
                    and self.progress_bars[name] in progress_widget.children
                ):
                    pb = self.progress_bars[name]
                    if pb.value < 1.0:
                        pb.bar_style = "danger"
                        pb.remove_class("animated-progress")
                        pb.add_class("completed-progress")  # Visually "stops" it

        self.auto_update_interval_label.value = _span("interval-label", "Calculation cancelled ❌")

    @functools.cached_property
    def _widgets(self) -> widgets.VBox:
        """Display the progress widgets with styles."""
        # Now iterates over _progress_widgets which holds either VBox or the compact HTML
        parts = list(self._progress_widgets.values())

        buttons = self.buttons
        if self.task:  # Only show buttons if there's an active task context
            button_box = widgets.HBox(
                [buttons["update"], buttons["toggle_auto_update"], buttons["cancel"]],
                layout=widgets.Layout(justify_content="center"),
            )
            parts.extend([button_box, self.auto_update_interval_label])
        return widgets.VBox(parts, layout=widgets.Layout(max_width="700px"))

    def display(self) -> None:
        style = textwrap.dedent(
            """
            <style>
            .progress {
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .progress-bar {
                border-radius: 5px;
                transition: width 0.4s ease;
                background-image: linear-gradient(
                    -45deg,
                    rgba(255, 255, 255, 0.15) 25%,
                    transparent 25%,
                    transparent 50%,
                    rgba(255, 255, 255, 0.15) 50%,
                    rgba(255, 255, 255, 0.15) 75%,
                    transparent 75%,
                    transparent
                );
                background-size: 40px 40px;
            }
            .animated-progress .progress-bar {
                animation: stripes 1s linear infinite;
            }
            .completed-progress .progress-bar {
                animation: none;
            }
            @keyframes stripes {
                0% {
                    background-position: 0 0;
                }
                100% {
                    background-position: 40px 0;
                }
            }

            /* Pulse animation for completed tasks */
            @keyframes balanced-pulse {
                0% {
                    box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.4);
                    border-color: rgba(46, 204, 113, 0.8);
                    transform: scale(1);
                }
                40% {
                    box-shadow: 0 0 0 5px rgba(46, 204, 113, 0.2);
                    border-color: rgba(46, 204, 113, 0.9);
                    transform: scale(1.005);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(46, 204, 113, 0);
                    border-color: inherit; /* Use parent's border or default */
                    transform: scale(1);
                }
            }
            .pulse-animation {
                animation: balanced-pulse 1.5s ease-out 1;
            }

            .percent-label {
                margin-left: 10px;
                font-weight: bold;
                color: #3366cc;
            }
            .estimate-label {
                font-style: italic;
                color: #666;
            }
            .speed-label {
                font-weight: bold;
                color: #009900;
            }
            .interval-label {
                font-weight: bold;
                color: #990000;
            }
            .widget-label {
                margin-bottom: 5px;
                color: #333;
                font-family: monospace;
            }
            .widget-button {
                margin-top: 5px;
            }
            .progress-vbox { /* Applied to both original VBox and new compact HTML */
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                transition: all 0.2s ease-in-out;
                position: relative;
                top: 0;
            }
            .progress-vbox:hover {
                transform: scale(1.005);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                top: -2px;
            }
            .compact-summary-label { /* Style for the text inside the compact HTML widget */
                font-weight: bold;
                color: #333; /* A neutral dark color */
                display: block; /* Ensures it takes up the space */
                text-align: left; /* Align text to the left */
            }
            </style>
            """,
        )
        hues = {hue for name in self.progress_dict if (hue := _get_scope_hue(name)) is not None}
        for hue in hues:
            style += _scope_background_color_css(hue)
        style += "</style>"
        IPython.display.display(IPython.display.HTML(style))
        IPython.display.display(self._widgets)
