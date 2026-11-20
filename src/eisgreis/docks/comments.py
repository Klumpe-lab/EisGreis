"""The Tomogram Comments dock: a comment per tomogram, saved to and loaded from a CSV."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from magicgui.widgets import Container, FileEdit, LineEdit, PushButton

if TYPE_CHECKING:
    from eisgreis.app import Controller


class CommentsDock:
    def __init__(self, controller: "Controller") -> None:
        self.c = controller
        self._updating = False
        self.comment = LineEdit(label="Comment", value="")
        self.load_comments = FileEdit(label="Load comments CSV", mode="r", filter="*.csv", value=None)
        self.save_comments = FileEdit(
            label="Save comments CSV", mode="w", filter="*.csv", value=str(controller.session.outputs.comments_csv)
        )
        self.save_now = PushButton(label="Save comments now")

        self.comment.changed.connect(self._on_comment_changed)
        self.load_comments.changed.connect(self._on_load_comments_changed)
        self.save_comments.changed.connect(self._on_save_path_changed)
        self.save_now.changed.connect(self._on_save_now)

        self.container = Container(
            widgets=[self.comment, self.load_comments, self.save_comments, self.save_now], labels=True
        )
        self.container.native.setWindowTitle("Tomogram Comments")

    def show_comment(self) -> None:
        """The comment of the tomogram on screen, in the field."""
        self._updating = True
        try:
            self.comment.value = self.c.session.comment()
        finally:
            self._updating = False

    def show_outputs(self) -> None:
        """The comments CSV the session saves to, in the field."""
        self._updating = True
        try:
            self.save_comments.value = str(self.c.session.outputs.comments_csv)
        finally:
            self._updating = False

    def load_csv(self, csv_path: Path) -> None:
        self.c.session.load_comments_csv(csv_path)
        if self.c.session.current is not None:
            self.show_comment()

    def _on_comment_changed(self, value: str) -> None:
        if self._updating or self.c.session.current is None:
            return
        self.c.session.set_comment(value)

    def _on_load_comments_changed(self, value: object) -> None:
        if self._updating or value in (None, ""):
            return
        self.load_csv(Path(value))
        self._updating = True
        try:
            self.load_comments.value = ""
        finally:
            self._updating = False

    def _on_save_path_changed(self, value: object) -> None:
        if self._updating:
            return
        self.c.session.set_comments_csv(None if value in (None, "") else str(value))

    def _on_save_now(self, _value: object) -> None:
        self.c.session.save_comments_csv()
