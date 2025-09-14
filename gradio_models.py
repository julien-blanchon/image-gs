import logging
import math
import os
import threading
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim
from torchvision.transforms.functional import gaussian_blur
from PIL import Image

from gsplat import (
    project_gaussians_2d_scale_rot,
    rasterize_gaussians_no_tiles,
    rasterize_gaussians_sum,
)
from utils.image_utils import (
    compute_image_gradients,
    get_grid,
    get_psnr,
    load_images,
    to_output_format,
)
from utils.misc_utils import set_random_seed
from utils.quantization_utils import ste_quantize
from utils.saliency_utils import get_smap


class StreamingResults:
    """Container for streaming training results"""

    def __init__(self):
        self.step = 0
        self.total_steps = 0
        self.current_render = None
        self.current_gaussian_id = None
        self.initialization_map = None  # Single map for current initialization mode
        self.final_render = None
        self.final_checkpoint_path = None
        self.training_logs = []
        self.metrics = {
            "psnr": 0.0,
            "ssim": 0.0,
            "loss": 0.0,
            "render_time": 0.0,
            "total_time": 0.0,
        }
        self.is_complete = False
        # Store all step results for interactive browsing
        self.step_renders = {}  # {step: PIL_Image}
        self.step_gaussian_ids = {}  # {step: PIL_Image}
        # For async visualization generation
        self.vis_lock = threading.Lock()


class GradioStreamingHandler(logging.Handler):
    """Custom logging handler that captures logs for Gradio streaming"""

    def __init__(self, results_container: StreamingResults):
        super().__init__()
        self.results = results_container

    def emit(self, record):
        log_entry = self.format(record)
        self.results.training_logs.append(log_entry)
        # Keep only last 100 log entries to avoid memory issues
        if len(self.results.training_logs) > 100:
            self.results.training_logs = self.results.training_logs[-100:]


class GradioGaussianSplatting2D(nn.Module):
    """Gradio-optimized version of GaussianSplatting2D with streaming capabilities"""

    def __init__(self, args, results_container: StreamingResults):
        super(GradioGaussianSplatting2D, self).__init__()
        self.results = results_container
        self.evaluate = args.eval
        set_random_seed(seed=args.seed)

        # Device setup
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float32

        # Initialize components
        self._init_logging(args)
        self._init_target(args)
        self._init_bit_precision(args)
        self._init_gaussians(args)
        self._init_loss(args)
        self._init_optimization(args)

        # Initialization
        if self.evaluate:
            self.ckpt_file = args.ckpt_file
            self._load_model()
        else:
            self._init_pos_scale_feat(args)

    def _init_logging(self, args):
        self.log_dir = getattr(args, "log_dir", "temp_gradio_logs")
        self.vis_gaussians = args.vis_gaussians
        self.save_image_steps = args.save_image_steps
        self.eval_steps = args.eval_steps

        # Set up streaming logger
        self.worklog = logging.getLogger("GradioImageGS")
        self.worklog.handlers.clear()  # Remove existing handlers

        # Add our streaming handler
        stream_handler = GradioStreamingHandler(self.results)
        stream_handler.setFormatter(
            logging.Formatter(fmt="[{asctime}] {message}", style="{")
        )
        self.worklog.addHandler(stream_handler)
        self.worklog.setLevel(logging.INFO)

        self.worklog.info(
            f"Start optimizing {args.num_gaussians:d} Gaussians for '{args.input_path}'"
        )

    def _init_target(self, args):
        self.gamma = args.gamma
        self.downsample = args.downsample
        if self.downsample:
            self.downsample_ratio = float(args.downsample_ratio)

        self.block_h, self.block_w = 16, 16
        self._load_target_images(path=os.path.join(args.data_root, args.input_path))

        if self.downsample:
            self.gt_images_upsampled = self.gt_images
            self.img_h_upsampled, self.img_w_upsampled = self.img_h, self.img_w
            self.tile_bounds_upsampled = self.tile_bounds
            self._load_target_images(
                path=os.path.join(args.data_root, args.input_path),
                downsample_ratio=self.downsample_ratio,
            )

        self.num_pixels = self.img_h * self.img_w

    def _load_target_images(self, path, downsample_ratio=None):
        self.gt_images, self.input_channels, self.image_fnames = load_images(
            load_path=path, downsample_ratio=downsample_ratio, gamma=self.gamma
        )
        self.gt_images = torch.from_numpy(self.gt_images).to(
            dtype=self.dtype, device=self.device
        )
        self.img_h, self.img_w = self.gt_images.shape[1:]
        self.tile_bounds = (
            (self.img_w + self.block_w - 1) // self.block_w,
            (self.img_h + self.block_h - 1) // self.block_h,
            1,
        )

    def _init_bit_precision(self, args):
        self.quantize = args.quantize
        self.pos_bits = args.pos_bits
        self.scale_bits = args.scale_bits
        self.rot_bits = args.rot_bits
        self.feat_bits = args.feat_bits

    def _init_gaussians(self, args):
        self.num_gaussians = args.num_gaussians
        self.total_num_gaussians = args.num_gaussians
        self.disable_prog_optim = args.disable_prog_optim

        if not self.disable_prog_optim and not self.evaluate:
            self.initial_ratio = args.initial_ratio
            self.add_times = args.add_times
            self.add_steps = args.add_steps
            self.num_gaussians = math.ceil(
                self.initial_ratio * self.total_num_gaussians
            )
            self.max_add_num = math.ceil(
                float(self.total_num_gaussians - self.num_gaussians) / self.add_times
            )
            min_steps = self.add_steps * self.add_times + args.post_min_steps
            if args.max_steps < min_steps:
                self.worklog.info(
                    f"Max steps ({args.max_steps:d}) is too small for progressive optimization. Resetting to {min_steps:d}"
                )
                args.max_steps = min_steps

        self.topk = args.topk
        self.eps = 1e-7 if args.disable_tiles else 1e-4
        self.init_scale = args.init_scale
        self.disable_topk_norm = args.disable_topk_norm
        self.disable_inverse_scale = args.disable_inverse_scale
        self.disable_color_init = args.disable_color_init

        # Initialize parameters
        self.xy = nn.Parameter(
            torch.rand(self.num_gaussians, 2, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
        self.scale = nn.Parameter(
            torch.ones(self.num_gaussians, 2, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
        self.rot = nn.Parameter(
            torch.zeros(self.num_gaussians, 1, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
        self.feat_dim = sum(self.input_channels)
        self.feat = nn.Parameter(
            torch.rand(
                self.num_gaussians, self.feat_dim, dtype=self.dtype, device=self.device
            ),
            requires_grad=True,
        )
        self.vis_feat = nn.Parameter(torch.rand_like(self.feat), requires_grad=False)

        self._log_compression_rate()

    def _log_compression_rate(self):
        bytes_uncompressed = float(self.gt_images.numel())
        bpp_uncompressed = float(8 * self.feat_dim)
        self.worklog.info(
            f"Uncompressed: {bytes_uncompressed / 1e3:.2f} KB | {bpp_uncompressed:.3f} bpp | 8.0 bppc"
        )
        bits_compressed = (
            2 * self.pos_bits
            + 2 * self.scale_bits
            + self.rot_bits
            + self.feat_dim * self.feat_bits
        ) * self.total_num_gaussians
        bytes_compressed = bits_compressed / 8.0
        bpp_compressed = float(bits_compressed) / self.num_pixels
        bppc_compressed = bpp_compressed / self.feat_dim
        self.num_bytes = bytes_compressed
        self.worklog.info(
            f"Compressed: {bytes_compressed / 1e3:.2f} KB | {bpp_compressed:.3f} bpp | {bppc_compressed:.3f} bppc"
        )
        self.worklog.info(
            f"Compression rate: {bpp_uncompressed / bpp_compressed:.2f}x | {100.0 * bpp_compressed / bpp_uncompressed:.2f}%"
        )

    def _init_loss(self, args):
        self.l1_loss_ratio = args.l1_loss_ratio
        self.l2_loss_ratio = args.l2_loss_ratio
        self.ssim_loss_ratio = args.ssim_loss_ratio

    def _init_optimization(self, args):
        self.disable_tiles = args.disable_tiles
        self.start_step = 1
        self.max_steps = args.max_steps
        self.results.total_steps = (
            args.max_steps
        )  # Set total steps for streaming progress
        self.pos_lr = args.pos_lr
        self.scale_lr = args.scale_lr
        self.rot_lr = args.rot_lr
        self.feat_lr = args.feat_lr

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.xy, "lr": self.pos_lr},
                {"params": self.scale, "lr": self.scale_lr},
                {"params": self.rot, "lr": self.rot_lr},
                {"params": self.feat, "lr": self.feat_lr},
            ]
        )

        self.disable_lr_schedule = args.disable_lr_schedule
        if not self.disable_lr_schedule:
            self.decay_ratio = args.decay_ratio
            self.check_decay_steps = args.check_decay_steps
            self.max_decay_times = args.max_decay_times
            self.decay_threshold = args.decay_threshold

    def _init_pos_scale_feat(self, args):
        self.init_mode = args.init_mode
        self.init_random_ratio = args.init_random_ratio
        self.pixel_xy = (
            get_grid(h=self.img_h, w=self.img_w)
            .to(dtype=self.dtype, device=self.device)
            .reshape(-1, 2)
        )

        with torch.no_grad():
            # Position initialization
            if self.init_mode == "gradient":
                self._compute_gmap()
                self.xy.copy_(self._sample_pos(prob=self.image_gradients))
            elif self.init_mode == "saliency":
                self.smap_filter_size = args.smap_filter_size
                self._compute_smap()
                self.xy.copy_(self._sample_pos(prob=self.saliency))
            else:  # random mode
                selected = np.random.choice(
                    self.num_pixels, self.num_gaussians, replace=False, p=None
                )
                self.xy.copy_(self.pixel_xy.detach().clone()[selected])
                # For random mode, create a simple random noise pattern
                if self.init_mode == "random":
                    random_pattern = np.random.rand(self.img_h, self.img_w)
                    self.results.initialization_map = Image.fromarray(
                        (random_pattern * 255).astype(np.uint8)
                    )

            # Scale initialization
            self.scale.fill_(
                self.init_scale if self.disable_inverse_scale else 1.0 / self.init_scale
            )

            # Feature initialization
            if not self.disable_color_init:
                self.feat.copy_(
                    self._get_target_features(positions=self.xy).detach().clone()
                )

    def _sample_pos(self, prob):
        num_random = round(self.init_random_ratio * self.num_gaussians)
        selected_random = np.random.choice(
            self.num_pixels, num_random, replace=False, p=None
        )
        selected_other = np.random.choice(
            self.num_pixels, self.num_gaussians - num_random, replace=False, p=prob
        )
        return torch.cat(
            [
                self.pixel_xy.detach().clone()[selected_random],
                self.pixel_xy.detach().clone()[selected_other],
            ],
            dim=0,
        )

    def _compute_gmap(self):
        gy, gx = compute_image_gradients(
            np.power(self.gt_images.detach().cpu().clone().numpy(), 1.0 / self.gamma)
        )
        g_norm = np.hypot(gy, gx).astype(np.float32)
        g_norm = g_norm / g_norm.max()

        # Store gradient map for streaming (only if this is the selected initialization mode)
        if self.init_mode == "gradient":
            self.results.initialization_map = Image.fromarray(
                (g_norm * 255).astype(np.uint8)
            )

        g_norm = np.power(g_norm.reshape(-1), 2.0)
        self.image_gradients = g_norm / g_norm.sum()
        self.worklog.info("Image gradient map computed")

    def _compute_smap(self):
        smap = get_smap(
            torch.pow(self.gt_images.detach().clone(), 1.0 / self.gamma),
            "models",
            self.smap_filter_size,
        )

        # Store saliency map for streaming (only if this is the selected initialization mode)
        if self.init_mode == "saliency":
            self.results.initialization_map = Image.fromarray(
                (smap * 255).astype(np.uint8)
            )

        self.saliency = (smap / smap.sum()).reshape(-1)
        self.worklog.info("Saliency map computed")

    def _get_target_features(self, positions):
        with torch.no_grad():
            target_features = F.grid_sample(
                self.gt_images.unsqueeze(0),
                positions[None, None, ...] * 2.0 - 1.0,
                align_corners=False,
            )
            target_features = target_features[0, :, 0, :].permute(1, 0)
        return target_features

    def forward(self, img_h, img_w, tile_bounds, upsample_ratio=None, benchmark=False):
        scale = self._get_scale(upsample_ratio=upsample_ratio)
        xy, rot, feat = self.xy, self.rot, self.feat

        if self.quantize:
            xy, scale, rot, feat = (
                ste_quantize(xy, self.pos_bits),
                ste_quantize(scale, self.scale_bits),
                ste_quantize(rot, self.rot_bits),
                ste_quantize(feat, self.feat_bits),
            )

        begin = perf_counter()
        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp

        if not self.disable_tiles:
            enable_topk_norm = not self.disable_topk_norm
            tmp = (
                xy,
                radii,
                conics,
                num_tiles_hit,
                feat,
                img_h,
                img_w,
                self.block_h,
                self.block_w,
                enable_topk_norm,
            )
            out_image = rasterize_gaussians_sum(*tmp)
        else:
            tmp = xy, conics, feat, img_h, img_w
            out_image = rasterize_gaussians_no_tiles(*tmp)

        render_time = perf_counter() - begin

        if benchmark:
            return render_time

        out_image = (
            out_image.view(-1, img_h, img_w, self.feat_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return out_image.squeeze(dim=0), render_time

    def _get_scale(self, upsample_ratio=None):
        scale = self.scale
        if not self.disable_inverse_scale:
            scale = 1.0 / scale
        if upsample_ratio is not None:
            scale = upsample_ratio * scale
        return scale

    def _tensor_to_pil_image(self, tensor_image):
        """Convert tensor image to PIL Image for streaming"""
        if tensor_image is None:
            return None

        # Convert to numpy and apply gamma correction
        image_np = (
            torch.pow(torch.clamp(tensor_image, 0.0, 1.0), 1.0 / self.gamma)
            .detach()
            .cpu()
            .numpy()
        )

        # Convert to uint8 format
        image_formatted = to_output_format(image_np, gamma=None)
        return Image.fromarray(image_formatted)

    def _create_gaussian_id_visualization(self):
        """Create Gaussian ID visualization as PIL Image using rasterization with vis_feat"""
        if not self.vis_gaussians:
            return None

        try:
            # Use vis_feat for ID visualization (this creates unique colors per Gaussian)
            feat = self.vis_feat * self.feat.norm(dim=-1, keepdim=True)

            # Render with ID features
            scale = self._get_scale()
            xy, rot = self.xy, self.rot

            if self.quantize:
                xy, scale, rot, feat = (
                    ste_quantize(xy, self.pos_bits),
                    ste_quantize(scale, self.scale_bits),
                    ste_quantize(rot, self.rot_bits),
                    ste_quantize(feat, self.feat_bits),
                )

            tmp = project_gaussians_2d_scale_rot(
                xy, scale, rot, self.img_h, self.img_w, self.tile_bounds
            )
            xy, radii, conics, num_tiles_hit = tmp

            if not self.disable_tiles:
                enable_topk_norm = not self.disable_topk_norm
                tmp = (
                    xy,
                    radii,
                    conics,
                    num_tiles_hit,
                    feat,
                    self.img_h,
                    self.img_w,
                    self.block_h,
                    self.block_w,
                    enable_topk_norm,
                )
                out_image = rasterize_gaussians_sum(*tmp)
            else:
                tmp = xy, conics, feat, self.img_h, self.img_w
                out_image = rasterize_gaussians_no_tiles(*tmp)

            out_image = (
                out_image.view(-1, self.img_h, self.img_w, self.feat_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            ).squeeze(dim=0)

            return self._tensor_to_pil_image(out_image)

        except Exception as e:
            self.worklog.error(f"Error creating Gaussian ID visualization: {e}")
            return None

    def optimize(self):
        """Main optimization loop with streaming updates"""
        self.psnr_curr, self.ssim_curr = 0.0, 0.0
        self.best_psnr, self.best_ssim = 0.0, 0.0
        self.decay_times, self.no_improvement_steps = 0, 0
        self.render_time_accum, self.total_time_accum = 0.0, 0.0

        # Initialize attributes needed for evaluation
        self.l1_loss = None
        self.l2_loss = None
        self.ssim_loss = None
        self.stop_requested = False

        # Initial render and update
        with torch.no_grad():
            images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds)
            self.results.current_render = self._tensor_to_pil_image(images)
            if self.vis_gaussians:
                try:
                    self.results.current_gaussian_id = (
                        self._create_gaussian_id_visualization()
                    )
                    self.worklog.info(
                        f"Initial visualizations created - Render: {'✓' if self.results.current_render else '✗'}, ID: {'✓' if self.results.current_gaussian_id else '✗'}"
                    )
                except Exception as e:
                    self.worklog.error(f"Error creating initial visualizations: {e}")
                    self.results.current_gaussian_id = None

            # Store initial results (step 0)
            self.results.step_renders[0] = self.results.current_render
            if self.vis_gaussians:
                self.results.step_gaussian_ids[0] = self.results.current_gaussian_id

        for step in range(self.start_step, self.max_steps + 1):
            self.step = step
            self.results.step = step

            self.optimizer.zero_grad()

            # Forward pass
            images, render_time = self.forward(self.img_h, self.img_w, self.tile_bounds)
            self.render_time_accum += render_time

            # Compute loss
            begin = perf_counter()
            self._get_total_loss(images)
            self.total_loss.backward()
            self.optimizer.step()
            self.total_time_accum += perf_counter() - begin + render_time

            # Update streaming results
            with torch.no_grad():
                if step % self.eval_steps == 0:
                    self._evaluate_and_update_stream(images)

                # Update render image more frequently, but visualizations less frequently
                render_update_freq = max(
                    50, self.save_image_steps // 2
                )  # Render updates every 50 steps
                vis_update_freq = max(
                    200, self.save_image_steps
                )  # Visualizations every 200 steps

                if step % render_update_freq == 0:
                    render_img = self._tensor_to_pil_image(images)
                    self.results.current_render = render_img

                # Only update Gaussian ID visualization less frequently
                if step % vis_update_freq == 0 and self.vis_gaussians:
                    # Generate Gaussian ID visualization asynchronously
                    def generate_gaussian_id_async():
                        try:
                            with self.results.vis_lock:
                                gaussian_id_vis = (
                                    self._create_gaussian_id_visualization()
                                )
                                self.results.current_gaussian_id = gaussian_id_vis

                        except Exception as e:
                            self.worklog.error(
                                f"Error creating Gaussian ID visualization at step {step}: {e}"
                            )
                            with self.results.vis_lock:
                                self.results.current_gaussian_id = None

                    # Start async visualization generation
                    vis_thread = threading.Thread(target=generate_gaussian_id_async)
                    vis_thread.daemon = True
                    vis_thread.start()

                # Store results for interactive browsing only at save_image_steps intervals
                if step % self.save_image_steps == 0:
                    # Store the current render for browsing
                    if self.results.current_render:
                        self.results.step_renders[step] = self.results.current_render

                    # Store Gaussian ID visualization for browsing
                    if self.vis_gaussians and self.results.current_gaussian_id:
                        self.results.step_gaussian_ids[step] = (
                            self.results.current_gaussian_id
                        )

                # Progressive optimization
                if (
                    not self.disable_prog_optim
                    and step % self.add_steps == 0
                    and self.num_gaussians < self.total_num_gaussians
                ):
                    self._add_gaussians(self.max_add_num)

                # Learning rate schedule
                terminate = False
                if (
                    not self.disable_lr_schedule
                    and self.num_gaussians == self.total_num_gaussians
                    and step % self.eval_steps == 0
                ):
                    terminate = self._lr_schedule()

                if terminate or self.stop_requested:
                    if self.stop_requested:
                        self.worklog.info("Training stopped by user request")
                    break

        # Final updates
        with torch.no_grad():
            images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds)
            self.results.final_render = self._tensor_to_pil_image(images)

            # Save final checkpoint and store path
            self._save_final_checkpoint()

            self.results.is_complete = True
            self.worklog.info("Optimization completed")

    def _get_total_loss(self, images):
        self.total_loss = 0

        if self.l1_loss_ratio > 1e-7:
            self.l1_loss = self.l1_loss_ratio * F.l1_loss(images, self.gt_images)
            self.total_loss += self.l1_loss
        else:
            self.l1_loss = None

        if self.l2_loss_ratio > 1e-7:
            self.l2_loss = self.l2_loss_ratio * F.mse_loss(images, self.gt_images)
            self.total_loss += self.l2_loss
        else:
            self.l2_loss = None

        if self.ssim_loss_ratio > 1e-7:
            self.ssim_loss = self.ssim_loss_ratio * (
                1 - fused_ssim(images.unsqueeze(0), self.gt_images.unsqueeze(0))
            )
            self.total_loss += self.ssim_loss
        else:
            self.ssim_loss = None

    def _evaluate_and_update_stream(self, images):
        """Evaluate current state and update streaming results"""
        gamma_corrected_images = torch.pow(
            torch.clamp(images, 0.0, 1.0), 1.0 / self.gamma
        )
        gamma_corrected_gt = torch.pow(self.gt_images, 1.0 / self.gamma)

        psnr = get_psnr(gamma_corrected_images, gamma_corrected_gt).item()
        ssim = fused_ssim(
            gamma_corrected_images.unsqueeze(0), gamma_corrected_gt.unsqueeze(0)
        ).item()

        self.psnr_curr, self.ssim_curr = psnr, ssim

        # Update metrics
        self.results.metrics.update(
            {
                "psnr": psnr,
                "ssim": ssim,
                "loss": self.total_loss.item(),
                "render_time": self.render_time_accum,
                "total_time": self.total_time_accum,
            }
        )

        # Log progress
        loss_results = f"Loss: {self.total_loss.item():.4f}"
        if self.l1_loss is not None:
            loss_results += f", L1: {self.l1_loss.item():.4f}"
        if self.l2_loss is not None:
            loss_results += f", L2: {self.l2_loss.item():.4f}"
        if self.ssim_loss is not None:
            loss_results += f", SSIM: {self.ssim_loss.item():.4f}"

        time_results = f"Total: {self.total_time_accum:.2f} s | Render: {self.render_time_accum:.2f} s"

        self.worklog.info(
            f"Step: {self.step:d} | {time_results} | {loss_results} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}"
        )

    def _save_final_checkpoint(self):
        """Save final checkpoint and store the path"""
        if self.quantize:
            with torch.no_grad():
                self.xy.copy_(ste_quantize(self.xy, self.pos_bits))
                self.scale.copy_(ste_quantize(self.scale, self.scale_bits))
                self.rot.copy_(ste_quantize(self.rot, self.rot_bits))
                self.feat.copy_(ste_quantize(self.feat, self.feat_bits))

        # Create checkpoint directory
        ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        psnr = self.results.metrics.get("psnr", 0.0)
        ssim = self.results.metrics.get("ssim", 0.0)

        ckpt_data = {
            "step": self.step,
            "psnr": psnr,
            "ssim": ssim,
            "bytes": getattr(self, "num_bytes", 0),
            "time": self.total_time_accum,
            "state_dict": self.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
        }

        ckpt_path = os.path.join(ckpt_dir, f"ckpt_step-{self.step:d}.pt")
        torch.save(ckpt_data, ckpt_path)
        self.results.final_checkpoint_path = ckpt_path

        self.worklog.info(f"Final checkpoint saved: {ckpt_path}")

    def _lr_schedule(self):
        """Learning rate scheduling logic"""
        if (
            self.psnr_curr <= self.best_psnr + 100 * self.decay_threshold
            or self.ssim_curr <= self.best_ssim + self.decay_threshold
        ):
            self.no_improvement_steps += self.eval_steps
            if self.no_improvement_steps >= self.check_decay_steps:
                self.no_improvement_steps = 0
                self.decay_times += 1
                if self.decay_times > self.max_decay_times:
                    return True
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] /= self.decay_ratio
                self.worklog.info(f"Learning rate decayed by {self.decay_ratio:.1f}")
            return False
        else:
            self.best_psnr = self.psnr_curr
            self.best_ssim = self.ssim_curr
            self.no_improvement_steps = 0
            return False

    def _add_gaussians(self, add_num):
        """Add Gaussians during progressive optimization"""
        add_num = min(
            add_num, self.max_add_num, self.total_num_gaussians - self.num_gaussians
        )
        if add_num <= 0:
            return

        # Compute error map for new Gaussian placement
        raw_images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds)
        images = torch.pow(torch.clamp(raw_images, 0.0, 1.0), 1.0 / self.gamma)
        gt_images = torch.pow(self.gt_images, 1.0 / self.gamma)

        kernel_size = round(np.sqrt(self.img_h * self.img_w) // 400)
        if kernel_size >= 1:
            kernel_size = max(3, kernel_size)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            gt_images = gaussian_blur(img=gt_images, kernel_size=kernel_size)

        diff_map = (gt_images - images).detach().clone()
        error_map = torch.pow(torch.abs(diff_map).mean(dim=0).reshape(-1), 2.0)
        sample_prob = (error_map / error_map.sum()).cpu().numpy()
        selected = np.random.choice(
            self.num_pixels, add_num, replace=False, p=sample_prob
        )

        # Create new Gaussians
        new_xy = self.pixel_xy.detach().clone()[selected]
        new_scale = torch.ones(add_num, 2, dtype=self.dtype, device=self.device)
        init_scale = self.init_scale
        new_scale.fill_(init_scale if self.disable_inverse_scale else 1.0 / init_scale)
        new_rot = torch.zeros(add_num, 1, dtype=self.dtype, device=self.device)
        new_feat = diff_map.permute(1, 2, 0).reshape(-1, self.feat_dim)[selected]
        new_vis_feat = torch.rand_like(new_feat)

        # Update parameters
        old_xy = self.xy.detach().clone()
        old_scale = self.scale.detach().clone()
        old_rot = self.rot.detach().clone()
        old_feat = self.feat.detach().clone()
        old_vis_feat = self.vis_feat.detach().clone()

        self.num_gaussians += add_num
        all_xy = torch.cat([old_xy, new_xy], dim=0)
        all_scale = torch.cat([old_scale, new_scale], dim=0)
        all_rot = torch.cat([old_rot, new_rot], dim=0)
        all_feat = torch.cat([old_feat, new_feat], dim=0)
        all_vis_feat = torch.cat([old_vis_feat, new_vis_feat], dim=0)

        self.xy = nn.Parameter(all_xy, requires_grad=True)
        self.scale = nn.Parameter(all_scale, requires_grad=True)
        self.rot = nn.Parameter(all_rot, requires_grad=True)
        self.feat = nn.Parameter(all_feat, requires_grad=True)
        self.vis_feat = nn.Parameter(all_vis_feat, requires_grad=False)

        # Update optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.xy, "lr": self.pos_lr},
                {"params": self.scale, "lr": self.scale_lr},
                {"params": self.rot, "lr": self.rot_lr},
                {"params": self.feat, "lr": self.feat_lr},
            ]
        )

        self.worklog.info(
            f"Step: {self.step:d} | Adding {add_num:d} Gaussians ({self.num_gaussians - add_num:d} -> {self.num_gaussians:d})"
        )
