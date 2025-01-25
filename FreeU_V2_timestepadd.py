import torch
import logging
import comfy.utils

def Fourier_filter(x, threshold, scale):
    # フーリエ変換フィルタリング
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # 逆フーリエ変換
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


class FreeU_V2_timestepadd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "b1": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
                "b2": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 10.0, "step": 0.01}),
                "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.01}),
                "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    def patch(self, model, b1, b2, s1, s2, start_percent, end_percent):
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        # モデルのスケーリング情報
        model_channels = model.model.model_config.unet_config["model_channels"]
        scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
        on_cpu_devices = {}

        # 出力ブロックのパッチ
        def output_block_patch(h, hsp, transformer_options):
            sigma = transformer_options["sigmas"][0].item()

            # 指定した範囲内のsigmaに基づいて処理
            if sigma_start >= sigma >= sigma_end:
                scale = scale_dict.get(int(h.shape[1]), None)
                if scale is not None:
                    hidden_mean = h.mean(1).unsqueeze(1)
                    B = hidden_mean.shape[0]
                    hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                    hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                    hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                    h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)

                    if hsp.device not in on_cpu_devices:
                        try:
                            hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                        except:
                            logging.warning("Device {} does not support the torch.fft functions used in the FreeU node, switching to CPU.".format(hsp.device))
                            on_cpu_devices[hsp.device] = True
                            hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
                    else:
                        hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

            return h, hsp

        # モデルのクローン作成とパッチ適用
        m = model.clone()
        m.set_model_output_block_patch(output_block_patch)
        return (m, )


# ノードクラスマッピング
NODE_CLASS_MAPPINGS = {
    "FreeU_V2_timestepadd": FreeU_V2_timestepadd,
}
