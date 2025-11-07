# [ファイル名: FreeU_S1S2_Refactored.py]
# 変更点:
# 1. UIに表示しない詳細設定をクラス定数として先頭に移動
# 2. define_schema (UI) を簡素化
# 3. execute のシグネチャを変更し、クラス定数を参照するように修正

import torch
import logging
import comfy.utils
from comfy_api.latest import io
import math

# ユーティリティ関数をインポート
from .utils import Fourier_filter_gauss, get_band_energy_stats

class FreeU_S_Scaling_AdaptiveCap(io.ComfyNode):

    # --- ▼▼▼ UIから削除された詳細設定 ▼▼▼ ---
    # これらの値はUIに表示されません。
    # 変更する場合は、このスクリプトファイルを直接編集してください。
    
    # S1 (1280ch) 用の固定設定
    RADIUS_RATIO_1 = 0.08   # S1 (1280ch) Radius Ratio
    HF_BOOST_S1 = 1.2       # S1 HF Boost (1.0 = Off)
    CAP_THRESHOLD_S1 = 0.35 # S1 Cap Threshold (35%)
    CAP_FACTOR_S1 = 0.6     # S1 Cap Factor (0.6)
    
    # S2 (640ch) 用の固定設定
    RADIUS_RATIO_2 = 0.06   # S2 (640ch) Radius Ratio
    HF_BOOST_S2 = 1.0       # S2 HF Boost (1.0 = Off)
    CAP_THRESHOLD_S2 = 0.70 # S2 Cap Threshold (70%)
    CAP_FACTOR_S2 = 0.6     # S2 Cap Factor (0.6)
    
    # 汎用設定
    CHANNEL_THRESHOLD = 96  # Channel Match Threshold (±)
    
    # --- ▲▲▲ UIから削除された詳細設定 ▲▲▲ ---


    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FreeU_S_Scaling_AdaptiveCap",
            # (分かりやすいように表示名を変更。元のままでも構いません)
            display_name="FreeU S (Fourier) Adaptive Cap (Simple)", 
            category="model_patches/unet",
            inputs=[
                io.Model.Input(id="model"),
                
                # S (Skip/Fourier) - UIに残す
                io.Float.Input(id="s1", default=1.2, min=0.0, max=10.0, step=0.01, display_name="S1 (1280ch)"),
                io.Float.Input(id="s2", default=0.70, min=0.0, max=5.0, step=0.01, display_name="S2 (640ch)"),
                
                # Timesteps - UIに残す
                io.Float.Input(id="s_start_percent", default=0.6, min=0.0, max=1.0, step=0.001, display_name="S (Fourier) Start %"),
                io.Float.Input(id="s_end_percent", default=1.0, min=0.0, max=1.0, step=0.001, display_name="S (Fourier) End %"),
                
                # --- UIから削除された項目 ---
                # radius_ratio_1, radius_ratio_2
                # hf_boost_s1, hf_boost_s2
                # cap_threshold_s1, cap_factor_s1
                # cap_threshold_s2, cap_factor_s2
                # channel_threshold
                # ------------------------------

                # Adaptive Cap - UIに残す
                io.Boolean.Input(id="enable_cap", default=True, 
                                 display_name="Enable Over-attenuation Cap"),
                io.Boolean.Input(id="adaptive_cap", default=True, 
                                 display_name="Enable Adaptive Cap Factor"),
                
                # Debug - UIに残す
                io.Boolean.Input(id="verbose_logging", default=False, 
                                 display_name="Enable Verbose Logging"),
            ],
            outputs=[
                io.Model.Output(id="patched_model", display_name="MODEL"),
            ]
        )

    @classmethod
    def execute(cls, model: 'ModelPatcher', 
                s1: float, s2: float, 
                s_start_percent: float, s_end_percent: float,
                # --- UIから削除された引数をここからも削除 ---
                # radius_ratio_1: float, radius_ratio_2: float,
                # hf_boost_s1: float, hf_boost_s2: float,
                enable_cap: bool, adaptive_cap: bool,
                # cap_threshold_s1: float, cap_factor_s1: float,
                # cap_threshold_s2: float, cap_factor_s2: float,
                # channel_threshold: int, 
                verbose_logging: bool
                ) -> io.NodeOutput:
        
        model_sampling = model.get_model_object("model_sampling")
        s_sigma_start = model_sampling.percent_to_sigma(s_start_percent)
        s_sigma_end = model_sampling.percent_to_sigma(s_end_percent)

        model_channels = model.model.model_config.unet_config["model_channels"]
        
        target_ch_1 = model_channels * 4
        # --- ▼ ハードコードされたクラス定数を参照するように変更 ▼ ---
        s_tuple1 = (s1, cls.RADIUS_RATIO_1, cls.CAP_THRESHOLD_S1, cls.CAP_FACTOR_S1, cls.HF_BOOST_S1)
        
        target_ch_2 = model_channels * 2
        # --- ▼ ハードコードされたクラス定数を参照するように変更 ▼ ---
        s_tuple2 = (s2, cls.RADIUS_RATIO_2, cls.CAP_THRESHOLD_S2, cls.CAP_FACTOR_S2, cls.HF_BOOST_S2)

        on_cpu_devices = {} 
        MAX_CAP_ITER = 3 # ▼【安定性】 キャップの最大反復回数

        def output_block_patch(h, hsp, transformer_options):
            sigma = transformer_options["sigmas"][0].item()
            
            # ▼【汎用性】 近傍マッチ
            ch = int(h.shape[1])
            scale_tuple = None
            # --- ▼ ハードコードされたクラス定数を参照するように変更 ▼ ---
            if abs(ch - target_ch_1) <= cls.CHANNEL_THRESHOLD:
                scale_tuple = s_tuple1
            elif abs(ch - target_ch_2) <= cls.CHANNEL_THRESHOLD:
                scale_tuple = s_tuple2

            if scale_tuple is not None:
                # --- ▼ s_scale以外はクラス定数由来の値がここに入る ---
                s_scale, radius_ratio, cap_thresh, cap_fact, hf_boost = scale_tuple

                # --- B-Scaling (h) は変更せず、パススルー ---

                # --- S-Scaling (Fourier) Logic ---
                if s_sigma_start >= sigma >= s_sigma_end:
                    
                    H, W = hsp.shape[-2:]
                    R_eff = max(1, int(min(H, W) * radius_ratio))
                    
                    lf_e_before, hf_e_before, cover_rate = get_band_energy_stats(hsp, R_eff)
                    ratio_before = lf_e_before / hf_e_before if hf_e_before > 1e-6 else float('inf')

                    # 'verbose_logging' は execute の引数からスコープキャプチャされる
                    if verbose_logging:
                        logging.info(f"[FreeU_S-Cap] Mask | Sigma:{sigma:.4f}, Ch:{ch}, HxW:{H}x{W}, R_Ratio:{radius_ratio:.3f}, Eff_R:{R_eff}px, Cover:{cover_rate:.2f}%")
                        logging.info(f"    -> Energy Before | LF: {lf_e_before:.4e}, HF: {hf_e_before:.4e}, Ratio(LF/HF): {ratio_before:.4f}")

                    original_hsp = hsp
                    current_s_scale = s_scale
                    original_device = hsp.device
                    
                    # --- 1st Pass (hf_boost を渡す) ---
                    hsp_filtered = None
                    try:
                        hsp_filtered = Fourier_filter_gauss(
                            hsp.cpu() if on_cpu_devices.get(original_device) else hsp, 
                            radius_ratio, current_s_scale, hf_boost
                        )
                        if on_cpu_devices.get(original_device):
                            hsp_filtered = hsp_filtered.to(original_device)
                    except Exception as e:
                        if original_device not in on_cpu_devices:
                            logging.warning(f"[FreeU_S-Cap] Device {original_device} does not support torch.fft, switching to CPU (First time only log).")
                            on_cpu_devices[original_device] = True
                            hsp_filtered = Fourier_filter_gauss(hsp.cpu(), radius_ratio, current_s_scale, hf_boost).to(original_device)
                        else:
                            raise e 
                    
                    hsp = hsp_filtered

                    lf_e_after, hf_e_after, _ = get_band_energy_stats(hsp, R_eff)
                    ratio_after = lf_e_after / hf_e_after if hf_e_after > 1e-6 else float('inf')
                    drop = 1.0 - (ratio_after / ratio_before) if ratio_before > 1e-6 else 0.0
                    
                    original_drop = drop # ログ用に最初のドロップ率を保持
                    final_drop = drop
                    iters = 0

                    # --- ▼【安定性】 2nd Pass (反復型キャップ処理) ▼ ---
                    # 'enable_cap' と 'adaptive_cap' は execute の引数からスコープキャプチャされる
                    while (enable_cap and 
                           drop > cap_thresh and         # 閾値を超えている
                           current_s_scale < 0.999 and   # スケールが1.0未満（補正可能）
                           iters < MAX_CAP_ITER):      # 最大反復回数未満

                        if iters == 0:
                            logging.warning(f"[FreeU_S-Cap] Over-attenuation detected! Drop: {drop*100:.1f}% > {cap_thresh*100:.1f}%.")
                        elif verbose_logging:
                            logging.info(f"    -> [Iter {iters}] Re-capping. Drop: {drop*100:.1f}% > {cap_thresh*100:.1f}%")

                        effective_factor = cap_fact
                        if adaptive_cap:
                            effective_factor = cap_fact * (cap_thresh / drop)
                            if verbose_logging:
                                logging.info(f"    -> Adaptive factor calc: {cap_fact:.3f} (base) * ({cap_thresh:.3f} (target) / {drop:.3f} (drop)) = {effective_factor:.3f} (eff)")
                        
                        # 補間: s_eff = lerp(1.0, s, factor) = 1 - factor * (1 - s)
                        #【安定性】補間は常に *元* のs_scaleから計算し、安定させる
                        capped_s_scale = 1.0 - effective_factor * (1.0 - s_scale)
                        # ただし、前回の反復よりスケールが弱まる（1.0に近づく）ことだけを許可する
                        capped_s_scale = max(capped_s_scale, current_s_scale * (1.0 + 1e-4)) # 数値誤差を考慮
                        
                        # 変化がほぼ無ければ終了
                        if abs(capped_s_scale - current_s_scale) < 1e-4:
                            if verbose_logging: logging.info("    -> Cap converged.")
                            break

                        logging.warning(f"    -> Re-applying filter (Iter {iters+1}/{MAX_CAP_ITER}). S-Scale: {current_s_scale:.3f} -> {capped_s_scale:.3f}")

                        # フィルターを*元の*テンソル(original_hsp)から再実行
                        hsp_filtered_capped = None
                        try:
                            hsp_filtered_capped = Fourier_filter_gauss(
                                original_hsp.cpu() if on_cpu_devices.get(original_device) else original_hsp, 
                                radius_ratio, capped_s_scale, hf_boost
                            )
                            if on_cpu_devices.get(original_device):
                                hsp_filtered_capped = hsp_filtered_capped.to(original_device)
                        except Exception as e:
                            logging.error(f"[FreeU_S-Cap] Error during capped re-application: {e}")
                            hsp = original_hsp # エラー時は元に戻す
                            break
                        
                        if hsp_filtered_capped is not None:
                            hsp = hsp_filtered_capped
                            # 統計を再取得し、dropを更新（ループ条件のため）
                            lf_e_after, hf_e_after, _ = get_band_energy_stats(hsp, R_eff)
                            ratio_after = lf_e_after / hf_e_after if hf_e_after > 1e-6 else float('inf')
                            drop = 1.0 - (ratio_after / ratio_before) if ratio_before > 1e-6 else 0.0
                            
                            final_drop = drop # 最終ログ用に更新
                            current_s_scale = capped_s_scale
                        
                        iters += 1
                    # --- ▲ 反復型キャップ処理 終了 ▲ ---

                    # 最終的なログ文字列を生成
                    if iters > 0:
                        final_drop_str = f"Final Drop: {(final_drop*100):.1f}% (Capped from {(original_drop*100):.1f}% in {iters} iter(s))"
                    else:
                        final_drop_str = f"Final Drop: {(final_drop*100):.1f}%"

                    if verbose_logging:
                        logging.info(f"    -> Energy After  | LF: {lf_e_after:.4e}, HF: {hf_e_after:.4e}, Ratio(LF/HF): {ratio_after:.4f} | {final_drop_str}")
                    elif iters > 0: # キャップが作動した場合は、Verbose=OFFでも最終結果をログに出す
                        logging.info(f"[FreeU_S-Cap] {final_drop_str} (Sigma: {sigma:.4f}, Ch: {ch})")

            
            # B-Scaling (h) は変更せず、そのままパススルー
            return h, hsp

        m = model.clone()
        m.set_model_output_block_patch(output_block_patch)

        return io.NodeOutput(m)
