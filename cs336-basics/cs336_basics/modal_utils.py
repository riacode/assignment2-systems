from pathlib import Path, PurePosixPath

import modal

SUNET_ID = "riagarg"

if SUNET_ID == "":
    raise NotImplementedError(f"Please set the SUNET_ID in {__file__}")

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"basics-{SUNET_ID}")
user_volume = modal.Volume.from_name(f"basics-{SUNET_ID}", create_if_missing=True, version=2)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = (
        modal.Image.debian_slim()
        .apt_install("wget", "gzip")
        .uv_pip_install(
            "einops>=0.8",
            "einx>=0.4",
            "jaxtyping>=0.3",
            "numpy>=2.4",
            "psutil>=7",
            "pytest>=9.0",
            "pytest-timeout",
            "regex>=2026.3.32",
            "tiktoken>=0.12.0",
            "torch~=2.11.0",
            "tqdm>=4.67",
            "wandb>=0.25",
        )
    )
    image = image.add_local_python_source("cs336_basics")
    repo_root = Path(__file__).resolve().parents[2]
    for filename in ("AGENTS.md", "CLAUDE.md"):
        local_path = repo_root / filename
        if local_path.is_file():
            image = image.add_local_file(str(local_path), f"/root/{filename}")
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    f"/root/{DATA_PATH}": user_volume,
}


def secrets(include_huggingface_secret: bool = False) -> list[modal.Secret]:
    secrets = [modal.Secret.from_dict({"SOME_ENV_VAR": "some-value"}), modal.Secret.from_name("my-secret")]
    return secrets


def _cs336_basics_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_nsys_profile_image(*, python_version: str = "3.12") -> modal.Image:
    """Modal image with Nsight CLI, PyTorch, and profiling code at `/profiling` (minimal copy)."""
    root = _cs336_basics_repo_root()
    return (
        modal.Image.debian_slim(python_version=python_version)
        .run_commands(
            "apt-get update && apt-get install -y wget",
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
        )
        .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
        .uv_pip_install("numpy", "torch")
        .uv_pip_install("einops", "einx", "jaxtyping")
        .add_local_python_source("cs336_basics")
        .add_local_file(str(root / "benchmark.py"), "/profiling/benchmark.py")
        .add_local_file(
            str(root / "cs336_basics" / "benchmarking_script.py"),
            "/profiling/benchmarking_script.py",
        )
    )


def build_nsys_profile_image_py313() -> modal.Image:
    """Same as `build_nsys_profile_image` with ``python_version=\"3.13\"``."""
    return build_nsys_profile_image(python_version="3.13")
