import torch
import torch.nn as nn
import torch.nn.functional as F


class RSAdapter(nn.Module):
    """
    경량 병목(Bottleneck) 어댑터.
    SAM3 ViT 트랜스포머 블록 출력에 삽입하여 RS 도메인 지식을 주입한다.

    파라미터 수:
        d_model=1024, bottleneck=64  →  1024×64 + 64×1024 ≈ 131K
        32블록 전체 삽입 시 ~4.2M 파라미터 (SAM3 전체 840M의 ~0.5%)
    """

    def __init__(self, d_model: int = 1024, bottleneck: int = 64):
        super().__init__()
        self.down  = nn.Linear(d_model, bottleneck)
        self.act   = nn.GELU()
        self.up    = nn.Linear(bottleneck, d_model)
        # scale=0으로 초기화 → 삽입 직후에는 원래 ViT 동작 완전 유지 (안정적 PEFT 학습 시작)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) 또는 (B, N, d_model)
               ViT Block 출력 (vitdet.py Block은 (B, H, W, C) 형식으로 출력)
        Returns:
            동일 shape, 잔차 연결 포함
        """
        shape_4d = (x.dim() == 4)
        if shape_4d:
            B, H, W, C = x.shape
            x_flat = x.view(B, H * W, C)
        else:
            x_flat = x

        adapted = self.up(self.act(self.down(x_flat)))
        adapted = adapted.to(x.dtype)   # [FIX] bfloat16 autocast 환경에서 scale(float32)과의 dtype 불일치 방지

        if shape_4d:
            adapted = adapted.view(B, H, W, C)

        return x + self.scale * adapted


class RSMultiscaleFPN(nn.Module):
    """
    ViT 중간 블록 출력(F7, F15, F23, F31)을 받아
    FPN 스타일의 다중 스케일 피처 피라미드를 구성한다.

    입력: dict { "f7": (B, H, W, C), "f15": ..., "f23": ..., "f31": ... }
    출력: dict { "p2": (B, 256, 4H, 4W), "p3": ..., "p4": ..., "p5": ... }
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 256):
        super().__init__()
        # 각 스케일별 채널 축소 (1×1 Conv)
        self.lateral_f7  = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral_f15 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral_f23 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral_f31 = nn.Conv2d(in_channels, out_channels, 1)

        # FPN 탑다운 퓨전용 3×3 Conv (aliasing 감소)
        self.fpn_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def _to_spatial(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, C, H, W)"""
        return feat.permute(0, 3, 1, 2).contiguous()

    def forward(self, feats: dict) -> dict:
        """
        Args:
            feats: {"f7": ..., "f15": ..., "f23": ..., "f31": ...}
                   각 텐서 shape: (B, 72, 72, 1024)  [vitdet.py 출력 형식]
        Returns:
            {"p2": (B,256,288,288), "p3": (B,256,144,144),
             "p4": (B,256,72,72),  "p5": (B,256,36,36)}
        """
        # ── Lateral 연결 (채널 축소 + 공간 변환) ────────────────────────
        c7  = self.lateral_f7 (self._to_spatial(feats["f7"]))   # (B,256,72,72)
        c15 = self.lateral_f15(self._to_spatial(feats["f15"]))
        c23 = self.lateral_f23(self._to_spatial(feats["f23"]))
        c31 = self.lateral_f31(self._to_spatial(feats["f31"]))

        # ── 업/다운샘플로 스케일 조정 ─────────────────────────────────
        # P5: 다운샘플 (광역 맥락, 36×36)
        p5_lat = F.max_pool2d(c31, kernel_size=2, stride=2)       # (B,256,36,36)
        # P4: 그대로 (기존 vision_features 해상도, 72×72)
        p4_lat = c23                                               # (B,256,72,72)
        # P3: 업샘플 ×2 (144×144)
        p3_lat = F.interpolate(c15, scale_factor=2,
                               mode="bilinear", align_corners=False)
        # P2: 업샘플 ×4 (소형 객체, 288×288)
        p2_lat = F.interpolate(c7,  scale_factor=4,
                               mode="bilinear", align_corners=False)

        # ── FPN 탑다운 퓨전 ───────────────────────────────────────────
        p5 = self.fpn_p5(p5_lat)
        p4 = self.fpn_p4(
            p4_lat + F.interpolate(p5, size=p4_lat.shape[-2:],
                                   mode="bilinear", align_corners=False)
        )
        p3 = self.fpn_p3(
            p3_lat + F.interpolate(p4, size=p3_lat.shape[-2:],
                                   mode="bilinear", align_corners=False)
        )
        p2 = self.fpn_p2(
            p2_lat + F.interpolate(p3, size=p2_lat.shape[-2:],
                                   mode="bilinear", align_corners=False)
        )

        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
