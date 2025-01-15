# How to use it

### Prerequisite
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

- Python 3.12. Install it using [uv](https://docs.astral.sh/uv/guides/install-python/) 
or [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation). Changing python default version is not recommended

- Model and data will be transferred to specific device automatically (multi devices are not suppported)

```python
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### HW1

[HW1 report](src/pacsmv_501011/hw1/report.md)

First, download MNIST using `scripts/download_mnist.sh`, or Fashion-MNIST using `scripts/download_fmnist.sh`

Then run

Adam
```bash
uv run hw1 "hw1-adam" 
```

SGD
```bash
uv run hw1 "hw1-sgd" 
```
