from PIL import Image
import numpy as np
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


class ImageToImageDiffuser:
    '''
    Diffuser class for text-guided image-to-image generation using Stable Diffusion

    '''
    def __init__(self, scheduler, model_name='CompVis/stable-diffusion-v1-4',
                 image_height=256, image_width=256, model=None, tokenizer=None,
                 text_encoder=None, unet=None, device='cuda'):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.image_height = image_height
        self.image_width = image_width
        self.model = model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.device = device
        self.models_available = ['runwayml/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-2',
                                 'stabilityai/stable-diffusion-2-1', 'CompVis/stable-diffusion-v1-4']
        if self.model is None:
            self.load_model()
        if self.tokenizer is None or self.text_encoder is None:
            self.load_tokenizer_and_text_encoder()
        if self.unet is None:
            self.load_unet()

    def load_model(self, subfolder='vae') -> None:
        '''
        Load a pre-trained model which will be used to encoder the conditioned images,
        and decode the latents back into image space.
        :return: None
        '''
        if self.model_name in self.models_available:
            self.model = AutoencoderKL.from_pretrained(self.model_name, subfolder=subfolder, use_auth_token=True)
            self.model = self.model.to(self.device)
        else:
            raise ValueError('Model not available. Please choose from: {}'.format(self.models_available))

    def load_tokenizer_and_text_encoder(self, model_id: str = 'openai/clip-vit-large-patch14') -> None:
        '''
        Load the tokenizer and text encoder to tokenize and encode prompts.
        :return: None
        '''
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id)
        self.text_encoder = self.text_encoder.to(self.device)

    def load_unet(self, subfolder='unet') -> None:
        '''
        Load a pretrained UNet for transforming the text and image latents.
        :return: None
        '''
        self.unet = UNet2DConditionModel.from_pretrained(self.model_name, subfolder=subfolder, use_auth_token=True)
        self.unet = self.unet.to(self.device)

    def _get_text_embeds(self, prompt: str) -> torch.Tensor:
        '''
        Get the text embeddings for the prompt.
        (text input -> text encoder -> text embeddings)
        :param prompt: str
        :return: torch.Tensor
        '''

        text_input = self.tokenizer(prompt, padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # prepare negative conditioning for classifier-free guidance later
        uncond_input = self.tokenizer([''] * len(prompt), padding='max_length',
                                      max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # concatenate into a batch for a single forward pass
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def _encode_img_latents(self, imgs: list, use_latent_mean: bool) -> torch.Tensor:
        '''
        Encode the images into latents using a pretrained model encoder.
        (image input -> image encoder -> image latents)
        :param imgs: list
        :return: latent features extracted from images: torch.Tensor
        '''

        if not isinstance(imgs, list):
            imgs = [imgs]

        # prepare images for encoding
        img_arr = np.stack([np.array(img) for img in imgs], axis=0)
        img_arr = img_arr / 255.0
        img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
        img_arr = 2.0 * img_arr - 1.0

        # encode images into latents
        if use_latent_mean:
            latent_samples = self.model.encode(img_arr.to(self.device)).latent_dist.mean
        else:
            latent_samples = self.model.encode(img_arr.to(self.device)).latent_dist.sample()

        return latent_samples * self.model.config.scaling_factor

    def _decode_img_latents(self, latents: torch.Tensor) -> list:
        '''
        Decode the latents into images using a pretrained model decoder.
        (latent output from a pretrained unet -> image decoder -> image output)
        :param latents: torch.Tensor
        :return: list of images generated
        '''

        latents /= self.model.config.scaling_factor

        with torch.no_grad():
            imgs = self.model.decode(latents)['sample']

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()  # NCHW -> NHWC
        imgs = (imgs * 255).round().astype('uint8')
        return imgs

    def _prepare_latents(self, text_embeddings: torch.Tensor, image_latents: torch.Tensor,
                        num_inference_steps: int = 50, guidance_scale: float = 7.5,
                        return_all_latents: bool = False, start_step: int = 10) -> torch.Tensor:
        '''
        Produce latent output from text embeddings and image latents.
        (text and image latent input -> latent generative model (e.g., a denoising diffusion unet) -> latent output)
        :param text_embeddings: from a text encoder, torch.Tensor
        :param image_latents: from image model encoder, torch.Tensor
        :param num_inference_steps: number of inference diffusion steps to run, int
        :param guidance_scale: hyperparameter for the classifier-free guidance, float
        :param return_all_latents: whether to return all latents or just the final one, bool
        :param start_step: starting diffusion step for inference
        :return: latents: all latent conditions for generation, torch.Tensor
        '''

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        # add correct level of noise to the latents based on the starting step
        start_timestep = self.scheduler.timesteps[start_step]
        start_timesteps = start_timestep.repeat(image_latents.shape[0])
        noise = torch.randn_like(image_latents)
        image_latents = self.scheduler.add_noise(image_latents, noise, start_timesteps)

        latent_hist = [image_latents]
        with autocast(device_type=self.device):
            for i, t in tqdm(enumerate(self.scheduler.timesteps[start_step:])):
                # repeat the image latents for obtaining results with and without conditioning (guidance)
                latent_model_input = torch.cat([image_latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                image_latents = self.scheduler.step(noise_pred, t, image_latents)['prev_sample']
                latent_hist.append(image_latents)

        if not return_all_latents:
            return image_latents
        else:
            all_latents = torch.cat(latent_hist, dim=0)
            return all_latents

    def prompt_to_img(self, prompts: str or list, images: list, num_inference_steps: int = 50,
                      guidance_scale: float = 7.5, return_all_latents: bool = False,
                      batch_size: int = 2, start_step: int = 0, use_latent_mean: bool = True) -> list:
        '''
        Generate images from prompts with or without image conditions.
        :param prompts: text prompts, str or list
        :param images: image conditions, list
        :param num_inference_steps: number of inference diffusion steps to run, int
        :param guidance_scale: hyperparameter for the classifier-free guidance, float
        :param return_all_latents: whether to return all latents or just the final one, bool
        :param start_step: starting diffusion step for inference
        :param batch_size: batch size for generation, int
        :return: list of images generated
        '''

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self._get_text_embeds(prompts)

        # Image -> img latents
        image_latents = self._encode_img_latents(images, use_latent_mean=use_latent_mean)

        # Text embeds + img latents -> denoised image latents
        latents = self._prepare_latents(text_embeds, image_latents, num_inference_steps=num_inference_steps,
                                        guidance_scale=guidance_scale, return_all_latents=return_all_latents,
                                        start_step=start_step)

        # denoised img latents -> imgs
        # microbatch latent decoding to avoid running out of memory
        all_imgs = []
        for i in tqdm(range(0, len(latents), batch_size)):
            imgs = self._decode_img_latents(latents[i:i+batch_size])
            all_imgs.extend(imgs)

        return all_imgs


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule='scaled_linear', num_train_timesteps=1000)

    diffuser = ImageToImageDiffuser(scheduler=scheduler, model_name='CompVis/stable-diffusion-v1-4',
                                    image_height=512, image_width=512, device=device)

    images = Image.open("/path_to_data/sample.png")
    prompt = 'a oil painting in the style of Picasso'

    # sample images from prompt and conditioning image
    generated_img = diffuser.prompt_to_img(prompt, [images], num_inference_steps=30, start_step=20,
                                           use_latent_mean=True)

    # prepare to plot images
    pil_images = [Image.fromarray(image) for image in generated_img]
    # show the first generated image as an example
    pil_images[0].show()


if __name__ == '__main__':
    main()


