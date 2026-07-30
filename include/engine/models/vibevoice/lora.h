#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/vibevoice/assets.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::vibevoice {

struct VibeVoiceLoraTensorDelta {
    std::string base_weight_name;
    int64_t rank = 0;
    int64_t in_features = 0;
    int64_t out_features = 0;
    float scale = 1.0F;
    std::vector<float> a;
    std::vector<float> b;
};

struct VibeVoiceLoraAdapterData {
    int64_t rank = 0;
    float scale = 1.0F;
    std::vector<VibeVoiceLoraTensorDelta> tensors;
};

// Loads and validates decoder LoRA tensors without merging them into the base.
// Optional full-weight diffusion/connector overrides are deliberately rejected:
// runtime adapter switching currently supports decoder PEFT LoRA only.
VibeVoiceLoraAdapterData load_vibevoice_runtime_lora(
    const assets::TensorSource & base,
    const std::filesystem::path & adapter_path,
    float scale_override = -1.0F);

// Overlays a fine-tune adapter directory onto a base weight source: the language-model LoRA is
// delta-merged into the decoder linears, and the diffusion head and acoustic/semantic connectors
// (when present) fully replace their base tensors. Mirrors infer.py's apply_lora.
std::shared_ptr<const assets::TensorSource> make_vibevoice_finetune_source(
    std::shared_ptr<const assets::TensorSource> base,
    const std::filesystem::path & adapter_path,
    float scale_override);

// Applies the fine-tune adapter selected by the vibevoice.lora / vibevoice.lora_scale options,
// returning assets whose model_weights carry the overlay (and fine_tune_applied set). Returns the
// input unchanged when no vibevoice.lora option is present; throws if an adapter was already
// applied (guards against passing it via both --load-option and --session-option).
std::shared_ptr<const VibeVoiceAssets> apply_vibevoice_finetune_options(
    std::shared_ptr<const VibeVoiceAssets> assets,
    const std::unordered_map<std::string, std::string> & options);

}  // namespace engine::models::vibevoice
