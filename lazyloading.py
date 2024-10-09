from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt


class LazyLoadingTable(QTableWidget):
    def __init__(self, parent=None, data=None, chunk_size=100):
        super().__init__(parent)
        self.chunk_size = chunk_size
        self.data = data
        self.total_rows = len(data)
        self.loaded_rows = 0  # 현재 로드된 행 수
        
        # 테이블 설정
        self.setRowCount(0)
        self.setColumnCount(len(data.columns))
        self.setHorizontalHeaderLabels(data.columns)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 데이터 로드
        self.load_more_data()

        # 스크롤 시 추가 데이터 로드
        self.verticalScrollBar().valueChanged.connect(self.on_scroll)

    def load_more_data(self):
        if self.loaded_rows >= self.total_rows:
            return  # 더 이상 로드할 데이터가 없음

        # 데이터의 다음 청크 로드
        next_chunk = self.data.iloc[self.loaded_rows:self.loaded_rows + self.chunk_size]
        self.setRowCount(self.loaded_rows + len(next_chunk))  # 현재 행 수에 청크 크기만큼 증가

        for row_idx, (index, row) in enumerate(next_chunk.iterrows(), start=self.loaded_rows):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.setItem(row_idx, col_idx, item)

        self.loaded_rows += len(next_chunk)

    def on_scroll(self, value):
        # 스크롤이 끝에 도달했을 때 추가 데이터 로드
        if self.verticalScrollBar().value() == self.verticalScrollBar().maximum():
            self.load_more_data()