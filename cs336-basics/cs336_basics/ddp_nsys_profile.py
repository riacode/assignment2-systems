import shutil
import subprocess
from pathlib import Path
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_nsys_profile_image, user_volume

@app.function(image=build_nsys_profile_image(), volumes=VOLUME_MOUNTS, gpu="B200:2", timeout=32400)
def run_ddp_nsys_profile_remote(profile_type: str):
    report_path = Path("/tmp") / f"{profile_type}.nsys-rep"
    command = ["nsys", "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,nccl,osrt",
        "-o",
        str(Path("/tmp") / profile_type),
        "python",
        "-m",
        "cs336_basics.ddp_nsys_workload",
        "--profile-type",
        profile_type,
    ]
    subprocess.run(command, check=True)

    output_path = Path("/root") / DATA_PATH / "nsys_reports" / f"{profile_type}.nsys-rep"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(report_path, output_path)
    user_volume.commit()
    return str(output_path)

@app.local_entrypoint()
def profile_ddp_nsys(profile_type: str = "both"):
    profile_types = ["initial", "overlap"] if profile_type == "both" else [profile_type]
    for current_type in profile_types:
        path = run_ddp_nsys_profile_remote.remote(current_type)
        print(path)
