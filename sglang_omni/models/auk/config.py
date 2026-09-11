# SPDX-License-Identifier: Apache-2.0
"""AuK: preprocessing, conditioning, batched DiT sampling, and VAE decode."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.reference_cache import (
    DEFAULT_AUDIO_CACHE_MAX_BYTES,
    DEFAULT_AUDIO_CACHE_MAX_ITEMS,
    DEFAULT_POSTERIOR_CACHE_MAX_BYTES,
    DEFAULT_POSTERIOR_CACHE_MAX_ITEMS,
)
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.auk"
PREPROCESSING_STAGE = "preprocessing"
CONDITIONING_STAGE = "conditioning"
ENGINE_STAGE = "auk_engine"
DECODE_STAGE = "decode"


class AuKReferenceCacheFactoryArgs(FactoryArgs):
    """Reference cache knobs shared by the preprocessing and conditioning stages."""

    ref_audio_cache: bool | None = None
    ref_audio_cache_max_items: int | None = Field(default=None, ge=1)
    ref_audio_cache_max_bytes: int | None = Field(default=None, ge=1)


class AuKReferenceCacheStageConfig(StageConfig):
    factory: AuKReferenceCacheFactoryArgs = Field(
        default_factory=AuKReferenceCacheFactoryArgs
    )


class AuKPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "AuKForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("AuK", "AuK-Flash")
    required_speech_reference_count: ClassVar[int | None] = None

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        PREPROCESSING_STAGE: AuKReferenceCacheStageConfig,
        CONDITIONING_STAGE: AuKReferenceCacheStageConfig,
    }

    stages: list[StageConfig] = [
        AuKReferenceCacheStageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=AuKReferenceCacheFactoryArgs(
                max_concurrency=8,
                ref_audio_cache=True,
                ref_audio_cache_max_items=DEFAULT_AUDIO_CACHE_MAX_ITEMS,
                ref_audio_cache_max_bytes=DEFAULT_AUDIO_CACHE_MAX_BYTES,
            ),
            next=CONDITIONING_STAGE,
        ),
        AuKReferenceCacheStageConfig(
            name=CONDITIONING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_conditioning_executor",
            factory=AuKReferenceCacheFactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                text_encoder_path=C.DEFAULT_TEXT_ENCODER,
                max_batch_size=8,
                max_batch_wait_ms=10,
                ref_audio_cache=True,
                ref_audio_cache_max_items=DEFAULT_POSTERIOR_CACHE_MAX_ITEMS,
                ref_audio_cache_max_bytes=DEFAULT_POSTERIOR_CACHE_MAX_BYTES,
            ),
            gpu=0,
            next=ENGINE_STAGE,
        ),
        StageConfig(
            name=ENGINE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_auk_engine_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                nfe=C.DEFAULT_NFE,
                cfg_strength=C.DEFAULT_CFG_STRENGTH,
                sway_sampling_coef=C.DEFAULT_SWAY_SAMPLING_COEF,
                max_seconds=C.MAX_SECONDS,
                max_batch_size=16,
                max_batch_wait_ms=10,
                weight_dtype="bfloat16",
            ),
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            factory=FactoryArgs(device=current_platform.device_type, max_batch_size=4),
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = AuKPipelineConfig
