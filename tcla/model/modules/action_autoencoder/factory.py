from .models_decoder_w_nonScalarPhase import Transformer_SequenceLang_w_nonScalarPhase
from .models_encoder_concise import Transformer_CNN
from .models_progress_perceiver import FrequencyProgressPerceiver
from .models_transport import LatentActionFlowNet_w_nonScalarPhase


def _require_type(actual, expected):
    if actual != expected:
        raise ValueError(f"Unsupported model type: {actual!r}; expected {expected!r}.")


def build_encoder(
    model_type,
    action_dim,
    chunk_size,
    lang_len,
    lang_dim,
    latent_action_dim,
    downsample_rate,
    is_relative_action,
    qpos_dim_for_relative_action,
):
    _require_type(model_type, "tf_seq_cnn")
    return Transformer_CNN(
        action_dim=action_dim,
        chunk_size=chunk_size,
        lang_len=lang_len,
        lang_dim=lang_dim,
        latent_action_dim=latent_action_dim,
        latent_action_len=chunk_size // downsample_rate,
        is_relative_action=is_relative_action,
        qpos_dim_for_relative_action=qpos_dim_for_relative_action,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )


def build_decoder(
    model_type,
    action_dim,
    chunk_size,
    lang_len,
    lang_dim,
    latent_action_dim,
    downsample_rate,
    is_relative_action,
    qpos_dim_for_relative_action,
    num_freqs,
):
    _require_type(model_type, "tf_attn_w_nonScalarPhase")
    return Transformer_SequenceLang_w_nonScalarPhase(
        action_dim=action_dim,
        chunk_size=chunk_size,
        lang_len=lang_len,
        lang_dim=lang_dim,
        latent_action_dim=latent_action_dim,
        latent_action_len=chunk_size // downsample_rate,
        is_relative_action=is_relative_action,
        qpos_dim_for_relative_action=qpos_dim_for_relative_action,
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
        dropout=0.0,
        use_memory_type_emb=True,
        use_learned_latent_pos=True,
        use_learned_lang_pos=True,
        causal_tgt_mask=False,
    )


def build_dynamics_model(model_type, latent_action_dim, latent_action_len=None):
    _require_type(model_type, "direct_flow_w_nonScalarPhase")
    return LatentActionFlowNet_w_nonScalarPhase(
        latent_action_dim=latent_action_dim,
        hidden_dim=256,
        num_layers=4,
        phase_noise_std=0.02,
        phase_drop_prob=0.05,
    )


def build_progress_model(
    model_type, num_freqs, action_dim, latent_action_dim, lang_dim
):
    _require_type(model_type, "perceiver")
    return FrequencyProgressPerceiver(
        num_freqs=num_freqs,
        action_dim=action_dim,
        latent_action_dim=latent_action_dim,
        lang_dim=lang_dim,
        hidden_dim=128,
    )
