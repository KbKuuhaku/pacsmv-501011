from dataclasses import dataclass


@dataclass
class TrainState:
    lr: float
    epoch: int = 0
    step: int = 0
    global_step: int = 0

    def update(self) -> None:
        self.step += 1
        self.global_step += 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.step = 0

    def __str__(self) -> str:
        return f"epoch = {self.epoch} | " + f"step = {self.step} | " + f"lr = {self.lr:.3E}"
