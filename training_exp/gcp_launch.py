import argparse
import os
import sys
import time
import uuid
import subprocess
import json

from typing import Dict, Any

from google.cloud import aiplatform, storage


def _upload_file(bucket_name: str, local_path: str, dest_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{dest_path}"


def _upload_text(bucket_name: str, dest_path: str, text: str, content_type: str = "text/yaml") -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_path)
    blob.upload_from_string(text, content_type=content_type)
    return f"gs://{bucket_name}/{dest_path}"


def _ensure_on_gcs(path: str, bucket_name: str, prefix: str) -> str:
    if path.startswith("gs://"):
        return path
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = f"{prefix.rstrip('/')}/{ts}-{os.path.basename(path)}"
    return _upload_file(bucket_name, path, dest)


def _build_and_upload_package(project_root: str, bucket_name: str, prefix: str) -> str:
    try:
        import build  # type: ignore
    except Exception:
        print("[INFO] Installing 'build' locally ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "build"]) 
    # Build sdist
    subprocess.check_call([sys.executable, "-m", "build", "--sdist"], cwd=project_root)
    dist_dir = os.path.join(project_root, "dist")
    candidates = sorted([f for f in os.listdir(dist_dir) if f.endswith(".tar.gz")], reverse=True)
    if not candidates:
        raise RuntimeError("No sdist found in dist/")
    pkg = os.path.join(dist_dir, candidates[0])
    dest = f"{prefix.rstrip('/')}/{os.path.basename(pkg)}"
    return _upload_file(bucket_name, pkg, dest)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML or JSON config from local path or gs:// URI."""
    if path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "rb") as f:
            data = f.read().decode("utf-8")
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(data) or {}
    return json.loads(data)


def main():
    ap = argparse.ArgumentParser(description="Launch Vertex AI Custom Training for LoRA SFT (config-based)")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config (local or gs://)")
    args = ap.parse_args()

    # Load config (contains both Vertex settings and training params)
    cfg = load_config(args.config)

    project = cfg.get("project")
    region = cfg.get("region", "us-central1")
    staging_bucket = cfg.get("staging_bucket")  # bucket name (no gs://)
    display_name = cfg.get("display_name", f"llama-sft-{uuid.uuid4().hex[:8]}")
    container_image = cfg.get("container_image", "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest")
    machine_type = cfg.get("machine_type", "a2-highgpu-1g")
    accelerator_type = cfg.get("accelerator_type", "NVIDIA_TESLA_A100")
    accelerator_count = int(cfg.get("accelerator_count", 1))

    if not project or not staging_bucket:
        raise ValueError("Config must include 'project' and 'staging_bucket'.")

    # Init Vertex
    aiplatform.init(project=project, location=region, staging_bucket=f"gs://{staging_bucket}")

    # Resolve training config: ensure dataset and output paths are on GCS
    ts = time.strftime("%Y%m%d-%H%M%S")
    train_cfg: Dict[str, Any] = dict(cfg)
    train_path = train_cfg.get("train_path")
    if not train_path:
        raise ValueError("Config must include 'train_path'.")
    dataset_gs = _ensure_on_gcs(train_path, staging_bucket, prefix="datasets")
    train_cfg["train_path"] = dataset_gs

    out_dir = train_cfg.get("output_dir")
    if not out_dir or not str(out_dir).startswith("gs://"):
        out_dir = f"gs://{staging_bucket}/outputs/{ts}-{uuid.uuid4().hex[:6]}/"
    train_cfg["output_dir"] = out_dir

    # Upload resolved training config to GCS
    try:
        import yaml
        config_text = yaml.safe_dump(train_cfg, sort_keys=False)
        config_gs = _upload_text(staging_bucket, f"configs/{ts}-{uuid.uuid4().hex[:6]}.yaml", config_text)
    except Exception as e:
        raise RuntimeError(f"Failed to serialize/upload config: {e}")

    # Build and upload package
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    package_gs = _build_and_upload_package(project_root, staging_bucket, prefix="packages")

    # Compose training args for module (just pass the config)
    train_args = ["--config", config_gs]

    env = {}
    if os.getenv("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=package_gs,
        python_module_name="training_exp.sft_llama",
        container_uri=container_image,
    )

    job.run(
        args=train_args,
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        environment_variables=env,
        base_output_dir=out_dir,
    )

    print(f"Launched job: {job.resource_name}")


if __name__ == "__main__":
    main()
