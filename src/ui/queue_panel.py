"""Batch queue panel for managing multiple videos."""

from pathlib import Path
from enum import Enum

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class QueueStatus(str, Enum):
    """Status of a queue item."""

    READY = "Ready"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    ERROR = "Error"


class QueuePanel(QWidget):
    """Panel for managing the video processing queue."""

    item_selected = pyqtSignal(str)
    export_requested = pyqtSignal()
    items_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._items: dict[str, QueueStatus] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the queue panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Queue")
        group_layout = QVBoxLayout(group)

        # Table for queue items
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["", "File", "Status"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Column sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 30)
        self.table.setColumnWidth(2, 100)

        # Hide vertical header
        self.table.verticalHeader().setVisible(False)

        # Connect selection change
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.cellDoubleClicked.connect(self._on_double_click)

        group_layout.addWidget(self.table)

        # Button row
        btn_layout = QHBoxLayout()

        self.add_btn = QPushButton("+ Add")
        self.add_btn.clicked.connect(self._on_add_clicked)
        btn_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("- Remove")
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        btn_layout.addWidget(self.remove_btn)

        self.up_btn = QPushButton("▲")
        self.up_btn.setFixedWidth(30)
        self.up_btn.clicked.connect(self._move_up)
        btn_layout.addWidget(self.up_btn)

        self.down_btn = QPushButton("▼")
        self.down_btn.setFixedWidth(30)
        self.down_btn.clicked.connect(self._move_down)
        btn_layout.addWidget(self.down_btn)

        btn_layout.addStretch()

        self.clear_btn = QPushButton("Clear Completed")
        self.clear_btn.clicked.connect(self._clear_completed)
        btn_layout.addWidget(self.clear_btn)

        group_layout.addLayout(btn_layout)
        layout.addWidget(group)

    def add_item(self, path: str, select: bool = False) -> None:
        """Add a video to the queue."""
        # Don't add duplicates
        if path in self._items:
            if select:
                self._select_path(path)
            return

        self._items[path] = QueueStatus.READY

        row = self.table.rowCount()
        self.table.insertRow(row)

        # Checkbox
        checkbox = QTableWidgetItem()
        checkbox.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        checkbox.setCheckState(Qt.CheckState.Checked)
        self.table.setItem(row, 0, checkbox)

        # Filename
        name_item = QTableWidgetItem(Path(path).name)
        name_item.setData(Qt.ItemDataRole.UserRole, path)
        name_item.setToolTip(path)
        self.table.setItem(row, 1, name_item)

        # Status
        status_item = QTableWidgetItem(QueueStatus.READY.value)
        self.table.setItem(row, 2, status_item)

        if select:
            self.table.selectRow(row)

        self.items_changed.emit()

    def _select_path(self, path: str) -> None:
        """Select the row with the given path."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                self.table.selectRow(row)
                break

    def remove_item(self, row: int) -> None:
        """Remove an item from the queue."""
        if 0 <= row < self.table.rowCount():
            item = self.table.item(row, 1)
            if item:
                path = item.data(Qt.ItemDataRole.UserRole)
                self._items.pop(path, None)
            self.table.removeRow(row)
            self.items_changed.emit()

    def set_item_status(self, path: str, status: QueueStatus) -> None:
        """Update the status of a queue item."""
        self._items[path] = status
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                self.table.item(row, 2).setText(status.value)
                break

    def get_selected_items(self) -> list[str]:
        """Get paths of selected and checked items."""
        selected_rows = set(index.row() for index in self.table.selectedIndexes())
        paths = []

        for row in selected_rows:
            checkbox = self.table.item(row, 0)
            if checkbox and checkbox.checkState() == Qt.CheckState.Checked:
                item = self.table.item(row, 1)
                if item:
                    paths.append(item.data(Qt.ItemDataRole.UserRole))

        return paths

    def get_checked_items(self) -> list[str]:
        """Get paths of all checked items."""
        paths = []
        for row in range(self.table.rowCount()):
            checkbox = self.table.item(row, 0)
            if checkbox and checkbox.checkState() == Qt.CheckState.Checked:
                item = self.table.item(row, 1)
                if item:
                    paths.append(item.data(Qt.ItemDataRole.UserRole))
        return paths

    def get_all_items(self) -> list[str]:
        """Get paths of all items."""
        paths = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            if item:
                paths.append(item.data(Qt.ItemDataRole.UserRole))
        return paths

    def _on_selection_changed(self) -> None:
        """Handle selection change."""
        selected = self.table.selectedItems()
        if selected:
            for item in selected:
                if item.column() == 1:
                    path = item.data(Qt.ItemDataRole.UserRole)
                    self.item_selected.emit(path)
                    break

    def _on_double_click(self, row: int, col: int) -> None:
        """Handle double-click on a row."""
        item = self.table.item(row, 1)
        if item:
            path = item.data(Qt.ItemDataRole.UserRole)
            self.item_selected.emit(path)

    def _on_add_clicked(self) -> None:
        """Handle add button click - emits signal for main window to handle."""
        # Main window will handle file dialog
        pass

    def _on_remove_clicked(self) -> None:
        """Remove selected items."""
        rows = sorted(set(index.row() for index in self.table.selectedIndexes()), reverse=True)
        for row in rows:
            self.remove_item(row)

    def _move_up(self) -> None:
        """Move selected item up in the queue."""
        rows = list(set(index.row() for index in self.table.selectedIndexes()))
        if len(rows) != 1 or rows[0] == 0:
            return

        row = rows[0]
        self._swap_rows(row, row - 1)
        self.table.selectRow(row - 1)

    def _move_down(self) -> None:
        """Move selected item down in the queue."""
        rows = list(set(index.row() for index in self.table.selectedIndexes()))
        if len(rows) != 1 or rows[0] >= self.table.rowCount() - 1:
            return

        row = rows[0]
        self._swap_rows(row, row + 1)
        self.table.selectRow(row + 1)

    def _swap_rows(self, row1: int, row2: int) -> None:
        """Swap two rows in the table."""
        for col in range(self.table.columnCount()):
            item1 = self.table.takeItem(row1, col)
            item2 = self.table.takeItem(row2, col)
            self.table.setItem(row1, col, item2)
            self.table.setItem(row2, col, item1)

    def _clear_completed(self) -> None:
        """Remove all completed items from the queue."""
        rows_to_remove = []
        for row in range(self.table.rowCount()):
            status_item = self.table.item(row, 2)
            if status_item and status_item.text() == QueueStatus.COMPLETED.value:
                rows_to_remove.append(row)

        for row in reversed(rows_to_remove):
            self.remove_item(row)
