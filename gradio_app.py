import os
import sys
import time
import threading
import argparse
import tempfile
import shutil
from typing import Generator, Optional, Tuple
import logging

try:
    import gradio as gr
except ImportError:
    print("❌ Gradio not found. Please install it with: pip install gradio>=4.0.0")
    sys.exit(1)

import torch
from PIL import Image

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradio_models import GradioGaussianSplatting2D, StreamingResults
from utils.misc_utils import load_cfg
from main import get_log_dir


class TrainingState:
    """Manages the state of training sessions"""

    def __init__(self):
        self.is_training = False
        self.training_thread = None
        self.model = None
        self.temp_dir = None
        self.results = StreamingResults()

    def reset(self):
        self.is_training = False
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.temp_dir = None
        self.results = StreamingResults()


# Global training state
training_state = TrainingState()


def create_args_from_config(
    image_path: str,
    exp_name: str,
    num_gaussians: int,
    quantize: bool,
    pos_bits: int,
    scale_bits: int,
    rot_bits: int,
    feat_bits: int,
    init_mode: str,
    init_random_ratio: float,
    max_steps: int,
    vis_gaussians: bool,
    save_image_steps: int,
    l1_loss_ratio: float,
    l2_loss_ratio: float,
    ssim_loss_ratio: float,
    pos_lr: float,
    scale_lr: float,
    rot_lr: float,
    feat_lr: float,
    disable_lr_schedule: bool,
    disable_prog_optim: bool,
) -> argparse.Namespace:
    """Create arguments object from Gradio inputs"""

    # Load default config
    parser = argparse.ArgumentParser()
    parser = load_cfg(cfg_path="cfgs/default.yaml", parser=parser)
    args = parser.parse_args([])  # Parse empty args to get defaults

    # Override with user inputs
    args.input_path = image_path
    args.exp_name = exp_name
    args.num_gaussians = num_gaussians
    args.quantize = quantize
    args.pos_bits = pos_bits
    args.scale_bits = scale_bits
    args.rot_bits = rot_bits
    args.feat_bits = feat_bits
    args.init_mode = init_mode
    args.init_random_ratio = init_random_ratio
    args.max_steps = max_steps
    args.vis_gaussians = vis_gaussians
    args.save_image_steps = save_image_steps
    args.l1_loss_ratio = l1_loss_ratio
    args.l2_loss_ratio = l2_loss_ratio
    args.ssim_loss_ratio = ssim_loss_ratio
    args.pos_lr = pos_lr
    args.scale_lr = scale_lr
    args.rot_lr = rot_lr
    args.feat_lr = feat_lr
    args.disable_lr_schedule = disable_lr_schedule
    args.disable_prog_optim = disable_prog_optim
    args.eval = False

    # Set up logging directory
    args.log_dir = get_log_dir(args)

    return args


def train_model(args: argparse.Namespace) -> None:
    """Training function that runs in a separate thread"""
    try:
        # Create and train model with streaming results
        training_state.model = GradioGaussianSplatting2D(args, training_state.results)

        # Start training
        training_state.model.optimize()

    except Exception as e:
        training_state.results.training_logs.append(f"ERROR: {str(e)}")
        logging.error(f"Training failed: {str(e)}")
    finally:
        training_state.is_training = False


def start_training_and_stream(
    image_file,
    exp_name: str,
    num_gaussians: int,
    quantize: bool,
    pos_bits: int,
    scale_bits: int,
    rot_bits: int,
    feat_bits: int,
    init_mode: str,
    init_random_ratio: float,
    max_steps: int,
    vis_gaussians: bool,
    save_image_steps: int,
    l1_loss_ratio: float,
    l2_loss_ratio: float,
    ssim_loss_ratio: float,
    pos_lr: float,
    scale_lr: float,
    rot_lr: float,
    feat_lr: float,
    disable_lr_schedule: bool,
    disable_prog_optim: bool,
) -> Generator[
    Tuple[
        str,
        str,
        Optional[Image.Image],  # initialization_map
        Optional[Image.Image],  # current_render
        Optional[Image.Image],  # current_gaussian_id
        bool,  # start_btn_interactive
        bool,  # stop_btn_interactive
    ],
    None,
    None,
]:
    """Start training and stream progress with images"""

    if training_state.is_training:
        yield (
            "Training is already in progress!",
            "",
            None,
            None,
            None,
            False,  # start_btn disabled
            True,  # stop_btn enabled
        )
        return

    if image_file is None:
        yield (
            "Please upload an image first!",
            "",
            None,
            None,
            None,
            True,  # start_btn enabled
            False,  # stop_btn disabled
        )
        return

    try:
        # Reset training state
        training_state.reset()

        # Create temporary directory for the uploaded image
        training_state.temp_dir = tempfile.mkdtemp()

        # Save uploaded image
        image_path = os.path.join(training_state.temp_dir, "input_image.png")
        image_file.save(image_path)

        # Create args
        args = create_args_from_config(
            image_path=image_path,
            exp_name=exp_name,
            num_gaussians=num_gaussians,
            quantize=quantize,
            pos_bits=pos_bits,
            scale_bits=scale_bits,
            rot_bits=rot_bits,
            feat_bits=feat_bits,
            init_mode=init_mode,
            init_random_ratio=init_random_ratio,
            max_steps=max_steps,
            vis_gaussians=vis_gaussians,
            save_image_steps=save_image_steps,
            l1_loss_ratio=l1_loss_ratio,
            l2_loss_ratio=l2_loss_ratio,
            ssim_loss_ratio=ssim_loss_ratio,
            pos_lr=pos_lr,
            scale_lr=scale_lr,
            rot_lr=rot_lr,
            feat_lr=feat_lr,
            disable_lr_schedule=disable_lr_schedule,
            disable_prog_optim=disable_prog_optim,
        )

        # Update data_root to use temp directory
        args.data_root = training_state.temp_dir
        args.input_path = "input_image.png"

        # Start training in separate thread
        training_state.is_training = True
        training_state.training_thread = threading.Thread(
            target=train_model, args=(args,)
        )
        training_state.training_thread.start()

        # Initial yield
        yield (
            "Training started! Check the progress below.",
            "Initializing training...",
            None,  # initialization_map
            None,  # current_render
            None,  # current_gaussian_id
            False,  # start_btn disabled
            True,  # stop_btn enabled
        )

        # Stream training progress
        while training_state.is_training or not training_state.results.is_complete:
            # Check if stop was requested
            if (
                not training_state.is_training
                and training_state.training_thread
                and training_state.training_thread.is_alive()
            ):
                # Force stop the training thread if needed
                training_state.results.training_logs.append(
                    "🛑 Training stopped by user request"
                )
                break

            # Get training logs
            if training_state.results.training_logs:
                logs_text = "\n".join(training_state.results.training_logs)

                # Add current metrics if available
                if training_state.results.step > 0:
                    # Break if step is greater than total steps
                    if training_state.results.step > training_state.results.total_steps:
                        break

                    metrics = training_state.results.metrics
                    status_line = (
                        f"\nCurrent: Step {training_state.results.step}/{training_state.results.total_steps} | "
                        f"PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | "
                        f"Loss: {metrics['loss']:.4f}"
                    )
                    logs_text += status_line

                    # Add image status info for debugging
                    if training_state.results.current_render is not None:
                        logs_text += f"\n📸 Current render: {training_state.results.current_render.size}"
                    else:
                        logs_text += "\n📸 Current render: None"

                    if training_state.results.current_gaussian_id is not None:
                        logs_text += f"\n🆔 Gaussian ID: {training_state.results.current_gaussian_id.size}"
                    else:
                        logs_text += "\n🆔 Gaussian ID: None"

                    logs_text += (
                        f"\n💾 Stored steps: {len(training_state.results.step_renders)}"
                    )
            else:
                logs_text = "Waiting for training to start..."

            # Get current images
            initialization_map = training_state.results.initialization_map
            current_render = training_state.results.current_render
            current_gaussian_id = training_state.results.current_gaussian_id

            # Simple status based on training state
            current_step = training_state.results.step
            if training_state.results.is_complete:
                status = "✅ Training completed successfully!"
                start_btn_interactive = True
                stop_btn_interactive = False
            elif not training_state.is_training:
                status = "⏹️ Training stopped."
                start_btn_interactive = True
                stop_btn_interactive = False
            else:
                status = f"🔄 Training in progress... Step {current_step}/{training_state.results.total_steps}"
                start_btn_interactive = False
                stop_btn_interactive = True

            # Always yield, even if images haven't changed
            yield (
                status,
                logs_text,
                initialization_map,
                current_render,
                current_gaussian_id,
                start_btn_interactive,
                stop_btn_interactive,
            )

            # Stop if training is complete
            if training_state.results.is_complete or not training_state.is_training:
                break
            if current_step > training_state.results.total_steps:
                break

            time.sleep(0.5)  # Update more frequently for better responsiveness

    except Exception as e:
        training_state.reset()
        yield (
            f"Failed to start training: {str(e)}",
            "",
            None,
            None,
            None,
            True,  # start_btn enabled
            False,  # stop_btn disabled
        )


def stop_training() -> str:
    """Stop the current training"""
    if not training_state.is_training:
        return "No training in progress."

    training_state.is_training = False
    training_state.results.training_logs.append(
        "🛑 STOP: Training stop requested by user..."
    )

    # Set a flag in the model to stop training
    if training_state.model:
        training_state.model.stop_requested = True

    return "Training stop requested. Will complete current step and stop."


def get_final_results() -> Tuple[Optional[Image.Image], Optional[str]]:
    """Get final training results"""
    final_render = training_state.results.final_render
    checkpoint_path = training_state.results.final_checkpoint_path
    return final_render, checkpoint_path


def browse_step_results(
    step: int,
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Browse results from a specific training step"""
    if not training_state.results.is_complete:
        return None, None

    # Find the closest available step
    available_steps = list(training_state.results.step_renders.keys())
    if not available_steps:
        return None, None

    closest_step = min(available_steps, key=lambda x: abs(x - step))

    render_img = training_state.results.step_renders.get(closest_step)
    gaussian_id_img = training_state.results.step_gaussian_ids.get(closest_step)

    return render_img, gaussian_id_img


def update_step_slider_after_training() -> gr.Slider:
    """Update step slider range and enable it after training completes"""
    if not training_state.results.is_complete:
        return gr.Slider(
            minimum=0,
            maximum=10000,
            value=0,
            step=100,
            label="Browse Training Steps",
            info="Training not complete yet",
            interactive=False,
        )

    available_steps = list(training_state.results.step_renders.keys())
    if not available_steps:
        return gr.Slider(
            minimum=0,
            maximum=10000,
            value=0,
            step=100,
            label="Browse Training Steps",
            info="No training steps available",
            interactive=False,
        )

    max_step = max(available_steps)
    min_step = min(available_steps)
    # Use the step size from save_image_steps if available, otherwise use difference between steps
    if len(available_steps) > 1:
        step_size = available_steps[1] - available_steps[0]
    else:
        step_size = 100

    return gr.Slider(
        minimum=min_step,
        maximum=max_step,
        value=max_step,
        step=step_size,
        label="Browse Training Steps",
        info=f"Browse results from steps {min_step}-{max_step} (interactive)",
        interactive=True,
    )


def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(
        title="Image-GS: 2D Gaussian Splatting", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # Image-GS: Content-Adaptive Image Representation via 2D Gaussians
        
        Upload an image and configure parameters to train a 2D Gaussian Splatting representation.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Configuration")

                # Image upload
                image_input = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=300,
                    sources=["upload"],
                    show_label=True,
                )

                # Basic parameters
                with gr.Group():
                    gr.Markdown("### Basic Parameters")
                    exp_name = gr.Textbox(
                        label="Experiment Name",
                        value="gradio_experiment",
                        info="Name for this training run",
                    )
                    num_gaussians = gr.Slider(
                        minimum=1000,
                        maximum=50000,
                        value=10000,
                        step=1000,
                        label="Number of Gaussians",
                        info="Number of Gaussians (for compression rate control). More = higher quality but slower training",
                    )
                    max_steps = gr.Slider(
                        minimum=100,
                        maximum=20000,
                        value=10000,
                        step=100,
                        label="Maximum Training Steps",
                        info="Maximum number of optimization steps. Default: 10000",
                    )

                # Quantization parameters
                with gr.Group():
                    gr.Markdown("### Quantization")
                    quantize = gr.Checkbox(
                        label="Enable Quantization",
                        value=False,
                        info="Enable bit precision control of Gaussian parameters. Reduces memory usage.",
                    )
                    with gr.Row():
                        pos_bits = gr.Slider(
                            4,
                            32,
                            16,
                            step=1,
                            label="Position Bits",
                            info="Bit precision of individual coordinate dimension",
                        )
                        scale_bits = gr.Slider(
                            4,
                            32,
                            16,
                            step=1,
                            label="Scale Bits",
                            info="Bit precision of individual scale dimension",
                        )
                    with gr.Row():
                        rot_bits = gr.Slider(
                            4,
                            32,
                            16,
                            step=1,
                            label="Rotation Bits",
                            info="Bit precision of Gaussian orientation angle",
                        )
                        feat_bits = gr.Slider(
                            4,
                            32,
                            16,
                            step=1,
                            label="Feature Bits",
                            info="Bit precision of individual feature dimension",
                        )

                # Initialization parameters
                with gr.Group():
                    gr.Markdown("### Initialization")
                    init_mode = gr.Radio(
                        choices=["gradient", "saliency", "random"],
                        value="saliency",
                        label="Initialization Mode",
                        info="Gaussian position initialization mode. Gradient uses image gradients, saliency uses attention maps.",
                    )
                    init_random_ratio = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Random Ratio",
                        info="Ratio of Gaussians with randomly initialized position (default: 0.3)",
                    )

                # Advanced parameters (collapsible)
                with gr.Accordion("Advanced Parameters", open=False):
                    # Loss parameters
                    gr.Markdown("#### Loss Weights")
                    with gr.Row():
                        l1_loss_ratio = gr.Slider(
                            0.0, 2.0, 1.0, step=0.1, label="L1 Loss"
                        )
                        l2_loss_ratio = gr.Slider(
                            0.0, 2.0, 0.0, step=0.1, label="L2 Loss"
                        )
                        ssim_loss_ratio = gr.Slider(
                            0.0, 1.0, 0.1, step=0.01, label="SSIM Loss"
                        )

                    # Learning rates
                    gr.Markdown("#### Learning Rates")
                    with gr.Row():
                        pos_lr = gr.Number(value=5e-4, label="Position LR", precision=6)
                        scale_lr = gr.Number(value=2e-3, label="Scale LR", precision=6)
                    with gr.Row():
                        rot_lr = gr.Number(value=2e-3, label="Rotation LR", precision=6)
                        feat_lr = gr.Number(value=5e-3, label="Feature LR", precision=6)

                    # Optimization options
                    gr.Markdown("#### Optimization")
                    disable_lr_schedule = gr.Checkbox(
                        label="Disable LR Schedule",
                        value=False,
                        info="Keep learning rate constant",
                    )
                    disable_prog_optim = gr.Checkbox(
                        label="Disable Progressive Optimization",
                        value=False,
                        info="Don't add Gaussians during training",
                    )

                # Visualization parameters
                with gr.Group():
                    gr.Markdown("### Visualization")
                    vis_gaussians = gr.Checkbox(
                        label="Visualize Gaussians",
                        value=True,
                        info="Visualize Gaussians during optimization (default: True)",
                    )
                    save_image_steps = gr.Slider(
                        minimum=200,
                        maximum=10000,
                        value=200,
                        step=100,
                        label="Save Image Every N Steps",
                        info="Frequency of rendering intermediate results during optimization (default: 100)",
                    )

                # Control buttons
                with gr.Row():
                    start_btn = gr.Button(
                        "Start Training", variant="primary", size="lg"
                    )
                    stop_btn = gr.Button("Stop Training", variant="stop", size="lg")

                status_text = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Column(scale=2):
                gr.Markdown("## Training Progress")

                # Progress logs (streaming)
                progress_logs = gr.Textbox(
                    label="Training Logs",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    autoscroll=True,
                )

                # Initial map (computed at start based on initialization mode)
                gr.Markdown("### Initialization Map")
                initialization_map = gr.Image(
                    label="Initialization Map",
                    type="pil",
                    height=200,
                )

                # Training images (streaming)
                gr.Markdown("### Current Training Results")
                with gr.Row():
                    current_render = gr.Image(
                        label="Current Render",
                        type="pil",
                        height=300,
                        show_label=True,
                        show_download_button=True,
                    )
                    current_gaussian_id = gr.Image(
                        label="Gaussian ID",
                        type="pil",
                        height=300,
                        show_label=True,
                        show_download_button=True,
                    )

                # Step slider for interactive browsing (will be updated dynamically)
                step_slider = gr.Slider(
                    minimum=0,
                    maximum=10000,
                    value=0,
                    step=100,
                    label="Browse Training Steps",
                    info="Slide to view results from different training steps (disabled during training)",
                    interactive=False,
                )

                gr.Markdown("## Final Results")
                with gr.Row():
                    final_render = gr.Image(
                        label="Final Render", type="pil", height=300
                    )
                    final_checkpoint = gr.File(label="Download Final Checkpoint (.pt)")

                # Results buttons
                with gr.Row():
                    results_btn = gr.Button("Load Final Results", size="lg")
                    enable_slider_btn = gr.Button(
                        "Enable Step Browsing", size="lg", variant="secondary"
                    )

        # Event handlers
        start_btn.click(
            fn=start_training_and_stream,
            inputs=[
                image_input,
                exp_name,
                num_gaussians,
                quantize,
                pos_bits,
                scale_bits,
                rot_bits,
                feat_bits,
                init_mode,
                init_random_ratio,
                max_steps,
                vis_gaussians,
                save_image_steps,
                l1_loss_ratio,
                l2_loss_ratio,
                ssim_loss_ratio,
                pos_lr,
                scale_lr,
                rot_lr,
                feat_lr,
                disable_lr_schedule,
                disable_prog_optim,
            ],
            outputs=[
                status_text,
                progress_logs,
                initialization_map,
                current_render,
                current_gaussian_id,
                start_btn,
                stop_btn,
            ],
        )

        stop_btn.click(fn=stop_training, outputs=status_text)

        results_btn.click(
            fn=get_final_results, outputs=[final_render, final_checkpoint]
        )

        enable_slider_btn.click(
            fn=update_step_slider_after_training, outputs=[step_slider]
        )

        step_slider.change(
            fn=browse_step_results,
            inputs=[step_slider],
            outputs=[current_render, current_gaussian_id],
        )

    return demo


if __name__ == "__main__":
    # Set torch hub directory
    torch.hub.set_dir("models/torch")

    # Create and launch the interface
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
