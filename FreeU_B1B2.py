# [ファイル名: FreeU_B1B2.py]

import torch
from comfy_api.latest import io
import logging

class FreeU_B_Scaling(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FreeU_B_Scaling",
            display_name="FreeU B (Backbone)",
            category="model_patches/unet",
            inputs=[
                io.Model.Input(id="model"),
                # B (Backbone)
                io.Float.Input(id="b1", default=1.3, min=0.0, max=10.0, step=0.01, display_name="B1 (1280ch)"),
                io.Float.Input(id="b2", default=1.4, min=0.0, max=10.0, step=0.01, display_name="B2 (640ch)"),
                
                # Timesteps
                io.Float.Input(id="b_start_percent", default=0.0, min=0.0, max=1.0, step=0.001, display_name="Start %"),
                io.Float.Input(id="b_end_percent", default=0.35, min=0.0, max=1.0, step=0.001, display_name="End %"),

                # --- ▼ 汎用性・デバッグUI (V5) ▼ ---
                io.Int.Input(id="channel_threshold", default=96, min=0, max=256, 
                              display_name="Channel Match Threshold (±)"),
                io.Boolean.Input(id="verbose_logging", default=False, 
                                 display_name="Enable Verbose Logging"),
            ],
            outputs=[
                io.Model.Output(id="patched_model", display_name="MODEL"),
            ]
        )

    @classmethod
    def execute(cls, model: 'ModelPatcher', 
                b1: float, b2: float, 
                b_start_percent: float, b_end_percent: float,
                channel_threshold: int, verbose_logging: bool
                ) -> io.NodeOutput:
        
        model_sampling = model.get_model_object("model_sampling")
        b_sigma_start = model_sampling.percent_to_sigma(b_start_percent)
        # ▼【バグ修正】 b_end_percent から b_sigma_end を計算
        b_sigma_end = model_sampling.percent_to_sigma(b_end_percent)

        model_channels = model.model.model_config.unet_config["model_channels"]
        
        # ▼【汎用性】 ターゲットチャンネルを定義
        target_ch_1 = model_channels * 4 # 1280
        target_ch_2 = model_channels * 2 # 640

        def output_block_patch(h, hsp, transformer_options):
            sigma = transformer_options["sigmas"][0].item()
            
            # ▼【汎用性】 近傍マッチ
            ch = int(h.shape[1])
            b_scale = None
            if abs(ch - target_ch_1) <= channel_threshold:
                b_scale = b1
            elif abs(ch - target_ch_2) <= channel_threshold:
                b_scale = b2

            if b_scale is not None:
                # ▼【バグ修正】 比較対象を b_sigma_end に修正
                if b_sigma_start >= sigma >= b_sigma_end:
                    if verbose_logging:
                        logging.info(f"[FreeU_B] Applying B-Scale (b={b_scale}) at Ch:{ch}, Sigma:{sigma:.4f}")
                        
                    try:
                        hidden_mean = h.mean(1).unsqueeze(1)
                        B = hidden_mean.shape[0]
                        hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                        hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                        
                        # ゼロ除算を防止 (V4のまま)
                        denominator = (hidden_max - hidden_min).clamp_min(1e-6)
                        hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / denominator.unsqueeze(2).unsqueeze(3)
                        
                        h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((b_scale - 1) * hidden_mean + 1)
                    except Exception as e:
                        logging.warning(f"[FreeU_B] B-Scaling failed at Ch:{ch} with error: {e}")
            
            # S-Scaling (hsp) は変更せず、そのまま次のパッチ（Sノード）へ渡す
            return h, hsp

        m = model.clone()
        m.set_model_output_block_patch(output_block_patch)
        return io.NodeOutput(m)