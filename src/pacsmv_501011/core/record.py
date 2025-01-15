from dataclasses import dataclass, field

from torch.utils.tensorboard.writer import SummaryWriter


@dataclass
class Record:
    best: float
    history: list[float] = field(default_factory=list)
    total: int = 0

    @property
    def is_best(self) -> bool:
        if not self.history:
            return False
        return self.best == self.history[-1]

    @property
    def avg(self) -> float:
        return sum(self.history) / self.total

    # api for tensorboard
    def write_avg_to_tensorboard(self, writer: SummaryWriter, global_step: int, title: str) -> None:
        writer.add_scalar(tag=title, scalar_value=self.avg, global_step=global_step)

    def update(self, item: float, count: int) -> None:
        self.history.append(item)
        self.total += count


@dataclass
class LossRecord(Record):
    best: float = float("inf")

    def update(self, item: float, count: int) -> None:
        super().update(item, count)
        self.best = min(self.best, self.avg)

    def __str__(self) -> str:
        return f"best loss = {self.best:.4f} | avg loss = {self.avg:.4f} | total = {self.total}"


@dataclass
class AccuracyRecord(Record):
    best: float = 0

    def update(self, item: float, count: int) -> None:
        super().update(item, count)
        self.best = max(self.best, self.avg)

    def __str__(self) -> str:
        return f"best acc = {self.best:.2%} | avg acc = {self.avg:.2%} | total = {self.total}"
