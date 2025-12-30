"""
Apple SHARP Model API - Modal Deployment

Deploys the Apple SHARP model as a serverless API endpoint on Modal.
SHARP converts single images into 3D Gaussian splats (PLY format).

Usage:
    # Deploy to Modal
    modal deploy sharp_api.py

    # Run locally for testing
    modal serve sharp_api.py
"""

import modal
import io
import base64
import os
from pathlib import Path

# Create the Modal app
app = modal.App("image-to-3d-sharp")

# Define the container image with all dependencies
sharp_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "ffmpeg",
        "libgomp1",
    )
    .pip_install(
        # Core ML packages
        "torch>=2.0.0",
        "torchvision",
        "numpy<2",
        "Pillow>=9.0",
        # Sharp dependencies
        "scipy>=1.11.0",
        "imageio>=2.31.0",
        "plyfile",
        "tqdm",
        "einops",
        "timm",
        "huggingface_hub",
        "pillow_heif",
        "matplotlib",
        "opencv-python-headless",
        "trimesh",
        "open3d",
        "safetensors",
        "moviepy==1.0.3",
        "e3nn",
        "omegaconf",
        "gsplat",
        "click",
        # API dependencies
        "fastapi",
        "requests>=2.31.0",
    )
    .run_commands(
        "git clone https://github.com/apple/ml-sharp.git /opt/ml-sharp",
        "cd /opt/ml-sharp && pip install -e . --no-deps",
        "python -c 'from sharp.models import create_predictor, PredictorParams; print(\"Sharp model imports OK\")'",
    )
)

# Volume to cache the model weights
model_cache = modal.Volume.from_name("sharp-model-cache", create_if_missing=True)

MODEL_CACHE_PATH = "/cache/models"
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def quaternions_from_rotation_matrices_gpu(matrices: "torch.Tensor") -> "torch.Tensor":
    """
    Pure PyTorch GPU implementation of rotation matrix to quaternion conversion.
    Based on the Shepperd method for numerical stability.
    """
    import torch

    batch_shape = matrices.shape[:-2]
    matrices = matrices.reshape(-1, 3, 3)

    m00, m01, m02 = matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 0, 2]
    m10, m11, m12 = matrices[:, 1, 0], matrices[:, 1, 1], matrices[:, 1, 2]
    m20, m21, m22 = matrices[:, 2, 0], matrices[:, 2, 1], matrices[:, 2, 2]

    trace = m00 + m11 + m22
    quaternions = torch.zeros(matrices.shape[0], 4, device=matrices.device, dtype=matrices.dtype)

    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        quaternions[mask1, 0] = 0.25 * s
        quaternions[mask1, 1] = (m21[mask1] - m12[mask1]) / s
        quaternions[mask1, 2] = (m02[mask1] - m20[mask1]) / s
        quaternions[mask1, 3] = (m10[mask1] - m01[mask1]) / s

    mask2 = (~mask1) & (m00 > m11) & (m00 > m22)
    if mask2.any():
        s = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2
        quaternions[mask2, 0] = (m21[mask2] - m12[mask2]) / s
        quaternions[mask2, 1] = 0.25 * s
        quaternions[mask2, 2] = (m01[mask2] + m10[mask2]) / s
        quaternions[mask2, 3] = (m02[mask2] + m20[mask2]) / s

    mask3 = (~mask1) & (~mask2) & (m11 > m22)
    if mask3.any():
        s = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2
        quaternions[mask3, 0] = (m02[mask3] - m20[mask3]) / s
        quaternions[mask3, 1] = (m01[mask3] + m10[mask3]) / s
        quaternions[mask3, 2] = 0.25 * s
        quaternions[mask3, 3] = (m12[mask3] + m21[mask3]) / s

    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2
        quaternions[mask4, 0] = (m10[mask4] - m01[mask4]) / s
        quaternions[mask4, 1] = (m02[mask4] + m20[mask4]) / s
        quaternions[mask4, 2] = (m12[mask4] + m21[mask4]) / s
        quaternions[mask4, 3] = 0.25 * s

    quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
    return quaternions.reshape(batch_shape + (4,))


def fast_decompose_covariance_matrices_gpu(covariance_matrices: "torch.Tensor"):
    """GPU-optimized SVD decomposition."""
    import torch

    device = covariance_matrices.device
    dtype = covariance_matrices.dtype

    rotations, singular_values_2, _ = torch.linalg.svd(covariance_matrices)

    det = torch.linalg.det(rotations)
    reflection_mask = det < 0
    if reflection_mask.any():
        rotations[reflection_mask, :, -1] *= -1

    quaternions = quaternions_from_rotation_matrices_gpu(rotations)
    singular_values = singular_values_2.sqrt()

    return quaternions.to(dtype=dtype), singular_values.to(dtype=dtype)


def fast_apply_transform_gpu(gaussians: "Gaussians3D", transform: "torch.Tensor"):
    """GPU-optimized transform."""
    import torch
    from sharp.utils.gaussians import Gaussians3D, compose_covariance_matrices

    transform_linear = transform[..., :3, :3]
    transform_offset = transform[..., :3, 3]

    mean_vectors = gaussians.mean_vectors @ transform_linear.T + transform_offset
    covariance_matrices = compose_covariance_matrices(
        gaussians.quaternions, gaussians.singular_values
    )
    covariance_matrices = (
        transform_linear @ covariance_matrices @ transform_linear.transpose(-1, -2)
    )

    quaternions, singular_values = fast_decompose_covariance_matrices_gpu(covariance_matrices)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def fast_unproject_gaussians_gpu(gaussians_ndc, extrinsics, intrinsics, image_shape):
    """GPU-optimized unprojection."""
    from sharp.utils.gaussians import get_unprojection_matrix

    unprojection_matrix = get_unprojection_matrix(extrinsics, intrinsics, image_shape)
    gaussians = fast_apply_transform_gpu(gaussians_ndc, unprojection_matrix[:3])
    return gaussians


def fast_save_ply_bytes(gaussians, f_px: float, image_shape: tuple) -> bytes:
    """Optimized PLY export that returns bytes directly."""
    import torch
    import numpy as np
    import io
    from plyfile import PlyData, PlyElement
    from sharp.utils import color_space as cs_utils
    from sharp.utils.gaussians import convert_rgb_to_spherical_harmonics

    with torch.no_grad():
        xyz = gaussians.mean_vectors.flatten(0, 1).cpu()
        scale_logits = torch.log(gaussians.singular_values).flatten(0, 1).cpu()
        quaternions = gaussians.quaternions.flatten(0, 1).cpu()
        colors_linear = gaussians.colors.flatten(0, 1).cpu()
        opacities = gaussians.opacities.flatten(0, 1).cpu()

        colors_srgb = cs_utils.linearRGB2sRGB(colors_linear)
        colors = convert_rgb_to_spherical_harmonics(colors_srgb)

        opacity_logits = torch.log(opacities / (1.0 - opacities)).unsqueeze(-1)

        disparity = 1.0 / gaussians.mean_vectors[0, ..., -1].cpu()
        quantiles = torch.quantile(disparity, q=torch.tensor([0.1, 0.9])).numpy()

    xyz_np = xyz.numpy()
    colors_np = colors.numpy()
    opacity_np = opacity_logits.numpy()
    scale_np = scale_logits.numpy()
    quat_np = quaternions.numpy()

    num_gaussians = len(xyz_np)
    image_height, image_width = image_shape

    dtype_full = np.dtype([
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])

    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements['x'] = xyz_np[:, 0]
    elements['y'] = xyz_np[:, 1]
    elements['z'] = xyz_np[:, 2]
    elements['f_dc_0'] = colors_np[:, 0]
    elements['f_dc_1'] = colors_np[:, 1]
    elements['f_dc_2'] = colors_np[:, 2]
    elements['opacity'] = opacity_np[:, 0]
    elements['scale_0'] = scale_np[:, 0]
    elements['scale_1'] = scale_np[:, 1]
    elements['scale_2'] = scale_np[:, 2]
    elements['rot_0'] = quat_np[:, 0]
    elements['rot_1'] = quat_np[:, 1]
    elements['rot_2'] = quat_np[:, 2]
    elements['rot_3'] = quat_np[:, 3]

    vertex_elements = PlyElement.describe(elements, 'vertex')

    image_size_arr = np.array([(image_width,), (image_height,)], dtype=[('image_size', 'u4')])
    intrinsic_arr = np.array(
        [(f_px,), (0,), (image_width * 0.5,), (0,), (f_px,), (image_height * 0.5,), (0,), (0,), (1,)],
        dtype=[('intrinsic', 'f4')]
    )
    extrinsic_arr = np.array([(v,) for v in np.eye(4).flatten()], dtype=[('extrinsic', 'f4')])
    frame_arr = np.array([(1,), (num_gaussians,)], dtype=[('frame', 'i4')])
    disparity_arr = np.array([(quantiles[0],), (quantiles[1],)], dtype=[('disparity', 'f4')])
    color_space_arr = np.array([(cs_utils.encode_color_space('sRGB'),)], dtype=[('color_space', 'u1')])
    version_arr = np.array([(1,), (5,), (0,)], dtype=[('version', 'u1')])

    plydata = PlyData([
        vertex_elements,
        PlyElement.describe(extrinsic_arr, 'extrinsic'),
        PlyElement.describe(intrinsic_arr, 'intrinsic'),
        PlyElement.describe(image_size_arr, 'image_size'),
        PlyElement.describe(frame_arr, 'frame'),
        PlyElement.describe(disparity_arr, 'disparity'),
        PlyElement.describe(color_space_arr, 'color_space'),
        PlyElement.describe(version_arr, 'version'),
    ])

    buffer = io.BytesIO()
    plydata.write(buffer)
    return buffer.getvalue()


@app.cls(
    image=sharp_image,
    gpu="A10G",
    timeout=300,
    volumes={MODEL_CACHE_PATH: model_cache},
    scaledown_window=300,
)
class SharpModel:
    """Sharp model class for image-to-3D Gaussian splat conversion."""

    @modal.enter()
    def load_model(self):
        """Load the Sharp model into GPU memory when the container starts."""
        import torch
        import subprocess
        import time

        start_time = time.time()

        os.environ["TORCH_HOME"] = MODEL_CACHE_PATH

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Warning: CUDA not available, using CPU")

        checkpoint_path = Path(MODEL_CACHE_PATH) / "sharp_2572gikvuh.pt"
        if not checkpoint_path.exists():
            print(f"Downloading Sharp model checkpoint from {DEFAULT_MODEL_URL}...")
            subprocess.run([
                "wget", "-q",
                DEFAULT_MODEL_URL,
                "-O", str(checkpoint_path)
            ], check=True)
            model_cache.commit()
            print("Model checkpoint downloaded and cached.")
        else:
            print("Using cached model checkpoint.")

        from sharp.models import create_predictor, PredictorParams

        print("Loading model weights...")
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=self.device)

        print("Creating predictor model...")
        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(self.device)

        print("Warming up model with dummy inference...")
        with torch.no_grad():
            import torch.nn.functional as F
            dummy_image = torch.randn(1, 3, 1536, 1536, device=self.device)
            dummy_disparity = torch.tensor([1.0], device=self.device)
            _ = self.predictor(dummy_image, dummy_disparity)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        print(f"Sharp model loaded and ready in {elapsed:.2f}s!")

    @modal.method()
    def predict(self, image_bytes: bytes) -> bytes:
        """Convert an image to 3D Gaussian splats."""
        import time
        import torch
        import torch.nn.functional as F
        import numpy as np
        from PIL import Image
        from sharp.utils.io import convert_focallength

        start_time = time.time()

        img_pil = Image.open(io.BytesIO(image_bytes))

        if img_pil.mode in ('RGBA', 'LA', 'P'):
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        image = np.array(img_pil)
        height, width = image.shape[:2]

        f_35mm = 30.0
        f_px = convert_focallength(width, height, f_35mm)

        print(f"Processing image: {width}x{height}, focal length: {f_px:.2f}px")

        internal_shape = (1536, 1536)

        image_pt = torch.from_numpy(image.copy()).float().to(self.device).permute(2, 0, 1) / 255.0
        disparity_factor = torch.tensor([f_px / width], device=self.device).float()

        image_resized = F.interpolate(
            image_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        print("Running inference...")
        inference_start = time.time()
        with torch.no_grad():
            gaussians_ndc = self.predictor(image_resized, disparity_factor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.time() - inference_start
        print(f"Inference completed in {inference_time:.3f}s")

        print("Running postprocessing...")
        postprocess_start = time.time()

        intrinsics = torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device
        )
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height

        gaussians = fast_unproject_gaussians_gpu(
            gaussians_ndc,
            torch.eye(4, device=self.device),
            intrinsics_resized,
            internal_shape
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ply_bytes = fast_save_ply_bytes(gaussians, f_px, (height, width))

        postprocess_time = time.time() - postprocess_start
        print(f"Postprocessing completed in {postprocess_time:.3f}s")

        elapsed = time.time() - start_time
        print(f"Total processing time: {elapsed:.3f}s")

        return ply_bytes

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        """Web endpoint for generating 3D Gaussian splats from an image."""
        try:
            image_bytes = None

            if "image" in request and request["image"]:
                image_bytes = base64.b64decode(request["image"])
            elif "image_url" in request and request["image_url"]:
                import requests
                response = requests.get(request["image_url"], timeout=30)
                response.raise_for_status()
                image_bytes = response.content
            else:
                return {
                    "success": False,
                    "error": "No image provided. Send 'image' (base64) or 'image_url'."
                }

            ply_bytes = self.predict.local(image_bytes)
            ply_base64 = base64.b64encode(ply_bytes).decode("utf-8")

            return {
                "success": True,
                "ply_base64": ply_base64,
                "message": "3D Gaussian splats generated successfully"
            }

        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }


@app.local_entrypoint()
def main():
    """Test the Sharp model locally."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: modal run sharp_api.py -- <image_path>")
        return

    image_path = sys.argv[1]
    print(f"Processing image: {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    model = SharpModel()
    ply_bytes = model.predict.remote(image_bytes)

    output_path = Path(image_path).stem + "_gaussian.ply"
    with open(output_path, "wb") as f:
        f.write(ply_bytes)

    print(f"Saved 3D Gaussian splats to: {output_path}")
