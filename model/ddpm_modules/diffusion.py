import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), t))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def pc_sampler(self,score_model, marginal_prob_std, diffusion_coeff, batch_size=64, num_steps=500,
                   snr=0.16, device='cuda', eps=1e-3):
        # Step1??????????????????1??????????????????????????????
        t = torch.ones(batch_size, device=device)
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
        # Step2????????????????????????????????????????????????????????????
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
            for time_step in tqdm.tqdm(time_steps):
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                # Corrector step (Langevin MCMC)
                grad = score_model(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevinstep_size = 2 * (snr * noise_norm / grad_norm) ** 2
                print(f" {langevinstep_size=}")
                for _ in range(10):
                    x = x + langevinstep_size * grad + torch.sqrt(2 * langevinstep_size) * torch.randn_like(x)
                    grad = score_model(x, batch_time_step)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
                    print(f" {langevin_step_size=}")
                # Predictor step (Euler-Maruyama )
                g = diffusion_coeff(batch_time_step)
                x_mean = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
                x = x_mean + torch.sqrt(g ** 2 * step_size)[:, None, None, None] * torch.randn_like(x)
                # Step4???????????????????????????????????????????????????????????? ???????????????
            return x_mean
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    def marginal_prob_std(self,t, sigma, device):
        "????????????t????????????????????????????????????????????????"
        t = torch.tensor(t, device=device)
        return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

    def diffusion_coeff(self,t, sigma, device):
        "????????????t???????????????????????????????????????SDE?????????????????? "
        return torch.tensor(sigma ** t, device=device)
    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        sigma = 25.0
        marginal_prob_std_fn = partial(self.marginal_prob_std, sigma=sigma)  # ??????????????????
        diffusion_coeff_fn = partial(self.diffusion_coeff, sigma=sigma)  # ??????????????????
        device = self.betas.device
        if not self.conditional:
            shape = x_in
            b = shape[0]
            # self.pc_sampler(self.denoise_fn, marginal_prob_std_fn,
            #                 diffusion_coeff_fn, sample_batch_size,
            #                 device=device,num_steps= self.num_timesteps)
            snr = 0.16
            eps = 1e-3
            t = torch.ones(b, device=device)
            init_x = torch.randn(shape, device=device) *marginal_prob_std_fn(t)[:, None, None, None]
            # Step2????????????????????????????????????????????????????????????
            time_steps = np.linspace(1., eps, self.num_timesteps)
            step_size = time_steps[0] - time_steps[1]
            ret_img = init_x
            sample_inter = (1 | (self.num_timesteps // 10))
            with torch.no_grad():
                for time_step in tqdm.tqdm(time_steps):
                    batch_time_step = torch.ones(b, device=device) * time_step
                    # Corrector step (Langevin MCMC)
                    grad =self.denoise_fn(torch.cat([x_in, init_x], dim=1), batch_time_step)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = np.sqrt(np.prod(init_x.shape[1:]))
                    langevinstep_size = 2 * (snr * noise_norm / grad_norm) ** 2
                    print(f" {langevinstep_size=}")
                    for _ in range(10):
                        init_x = init_x + langevinstep_size * grad + torch.sqrt(2 * langevinstep_size) * torch.randn_like(init_x)
                        grad = self.denoise_fn(torch.cat([x_in, init_x], dim=1), batch_time_step)
                        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                        noise_norm = np.sqrt(np.prod(x.shape[1:]))
                        langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
                        print(f" {langevin_step_size=}")
                    # Predictor step (Euler-Maruyama )
                    g = diffusion_coeff_fn(batch_time_step)
                    x_mean = init_x + (g ** 2)[:, None, None, None] * self.denoise_fn(torch.cat([x_in, init_x], dim=1), batch_time_step) * step_size
                    x = x_mean + torch.sqrt(g ** 2 * step_size)[:, None, None, None] * torch.randn_like(init_x)
                    # Step4???????????????????????????????????????????????????????????? ???????????????
                    if time_step % sample_inter == 0:
                        ret_img = torch.cat([ret_img,  x_mean], dim=0)
            if continous:
                return ret_img
            else:
                return ret_img[-1]
    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )
        # random gama
        # x_shape = x_start.shape
        # l = self.alphas_cumprod .gather(-1, t)
        # r = self.alphas_cumprod .gather(-1, t+1)
        # gama = (r - l) * torch.rand(0, 1) + l
        # gama = gama.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
        # return (
        #     nq.sqrt(gama) * x_start + nq.sqrt(1-gama)* noise
        # )



    def loss_fn(self, x, eps=1e-5):
        sigma = 25.0
        x_start = x['HR']
        device = x_start.device
        random_t = torch.rand(x_start.shape[0], device=x_start.device) * (1. - eps) + eps
        # step2????????????????????????????????????p_t(x)?????????????????????perturbed_x
        random_t=random_t.veiw(-1)
        z = torch.randn_like(x_start)
        std = self.marginal_prob_std(random_t, sigma,device)
        perturbed_x = x_start + z * std[:, None, None, None]
        # step3??????????????????????????????????????????score Network??????????????????score
        perturbed_x =torch.cat([x['SR'], perturbed_x ], dim=1)
        score= self.denoise_fn(perturbed_x,random_t)

        # step4 ??????score matching loss
        loss = torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3))
        return loss


    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, t)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), t)
        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.loss_fn(x, *args, **kwargs)
