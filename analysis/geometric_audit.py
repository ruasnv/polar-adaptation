def get_OV_circuits(model: PAFTModel) -> List[torch.Tensor]:
    """
    OV circuit matrix for each layer: W_V @ W_O  shape [d_model, d_model].
    The OV circuit is what directly maps token values to outputs.
    """
    layers = model.paft_layers()
    ovs = []
    for layer in layers:
        W_V = get_reconstructed_WV(layer)   # [d_model, d_model]
        W_O = get_reconstructed_WO(layer)   # [d_model, d_model]
        ovs.append(W_V @ W_O)
    return ovs


def get_per_head_WV(layer: PAFTAttentionLayer) -> List[torch.Tensor]:
    """Per-head W_V_h: list of [d_model, d_head] tensors."""
    d_head = layer.d_head
    W_V_flat = get_reconstructed_WV(layer)    # [d_model, d_model]
    return [W_V_flat[:, h*d_head:(h+1)*d_head] for h in range(layer.n_heads)]


def get_per_head_WO(layer: PAFTAttentionLayer) -> List[torch.Tensor]:
    """Per-head W_O_h: list of [d_head, d_model] tensors."""
    d_head = layer.d_head
    W_O_flat = get_reconstructed_WO(layer)    # [d_model, d_model]
    return [W_O_flat[h*d_head:(h+1)*d_head, :] for h in range(layer.n_heads)]