from typing import Callable, List, Optional, Union

import torch

from ...models import UNet2DModel
from ...schedulers import CMStochasticIterativeScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """


class ConsistencyModelPipeline(DiffusionPipeline):


    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet2DModel, scheduler: CMStochasticIterativeScheduler) -> None:
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

        self.safety_checker = None

    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Follows diffusers.VaeImageProcessor.postprocess
    def postprocess_image(self, sample: torch.Tensor, output_type: str = "pil"):
        if output_type not in ["pt", "np", "pil"]:
            raise ValueError(
                f"output_type={output_type} is not supported. Make sure to choose one of ['pt', 'np', or 'pil']"
            )

        # Equivalent to diffusers.VaeImageProcessor.denormalize
        sample = (sample / 2 + 0.5).clamp(0, 1)
        if output_type == "pt":
            return sample

        # Equivalent to diffusers.VaeImageProcessor.pt_to_numpy
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "np":
            return sample

        # Output_type must be 'pil'
        sample = self.numpy_to_pil(sample)
        return sample

    def prepare_class_labels(self, batch_size, device, class_labels=None):
        if self.unet.config.num_class_embeds is not None:
            if isinstance(class_labels, list):
                class_labels = torch.tensor(class_labels, dtype=torch.int)
            elif isinstance(class_labels, int):
                assert batch_size == 1, "Batch size must be 1 if classes is an int"
                class_labels = torch.tensor([class_labels], dtype=torch.int)
            elif class_labels is None:
                # Randomly generate batch_size class labels
                # TODO: should use generator here? int analogue of randn_tensor is not exposed in ...utils
                class_labels = torch.randint(0, self.unet.config.num_class_embeds, size=(batch_size,))
            class_labels = class_labels.to(device)
        else:
            class_labels = None
        return class_labels

    def check_inputs(self, num_inference_steps, timesteps, latents, batch_size, img_size, callback_steps):
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Exactly one of `num_inference_steps` or `timesteps` must be supplied.")

        if num_inference_steps is not None and timesteps is not None:
            logger.warning(
                f"Both `num_inference_steps`: {num_inference_steps} and `timesteps`: {timesteps} are supplied;"
                " `timesteps` will be used over `num_inference_steps`."
            )

        if latents is not None:
            expected_shape = (batch_size, 3, img_size, img_size)
            if latents.shape != expected_shape:
                raise ValueError(f"The shape of latents is {latents.shape} but is expected to be {expected_shape}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        batch_size: int = 1,
        class_labels: Optional[Union[torch.Tensor, List[int], int]] = None,
        num_inference_steps: int = 1,
        timesteps: List[int] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        class labels
                # TODO: should use generator here? int analogue of randn_tensor is not exposed in ...utils
                class_labels = torch.randint(0, self.unet.config.num_class_embeds, size=(batch_size,))
            class_labels = class_labels.to(device)
        else:
            class_labels = None
        return class_labels

    def check_inputs(self, num_inference_steps, timesteps, latents, batch_size, img_size, callback_steps):
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Exactly one of `num_inference_steps` or `timesteps` must be supplied.")

        if num_inference_steps is not None and timesteps is not None:
            logger.warning(
                f"Both `num_inference_steps`: {num_inference_steps} and `timesteps`: {timesteps} are supplied;"
                " `timesteps` will be used over `num_inference_steps`."
            )

        if latents is not None:
            expected_shape = (batch_size, 3, img_size, img_size)
            if latents.shape != expected_shape:
                raise ValueError(f"The shape of latents is {latents.shape} but is expected to be {expected_shape}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        batch_size: int = 1,
        class_labels: Optional[Union[torch.Tensor, List[int], int]] = None,
        num_inference_steps: int = 1,
        timesteps: List[int] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):

        # 0. Prepare call parameters
        img_size = self.unet.config.sample_size
        device = self._execution_device

        # 1. Check inputs
        self.check_inputs(num_inference_steps, timesteps, latents, batch_size, img_size, callback_steps)

        # 2. Prepare image latents
        # Sample image latents x_0 ~ N(0, sigma_0^2 * I)
        sample = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.unet.config.in_channels,
            height=img_size,
            width=img_size,
            dtype=self.unet.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 3. Handle class_labels for class-conditional models
        class_labels = self.prepare_class_labels(batch_size, device, class_labels=class_labels)

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

        # 5. Denoising loop
        # Multistep sampling: implements Algorithm 1 in the paper
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                scaled_sample = self.scheduler.scale_model_input(sample, t)
                model_output = self.unet(scaled_sample, t, class_labels=class_labels, return_dict=False)[0]

                sample = self.scheduler.step(model_output, t, sample, generator=generator)[0]

                # call the callback, if provided
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, sample)

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 6. Post-process image sample
        image = self.postprocess_image(sample, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
