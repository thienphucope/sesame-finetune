def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor):
    """
    Forward pass cho Sesame CSM với delay pattern đúng theo tài liệu chính thức
   [](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice),
    tương thích hoàn toàn với kiến trúc knottwill/sesame-finetune gốc.
    """
    dtype = next(self.parameters()).dtype
    bsz, seq_len, _ = tokens.size()
    device = tokens.device

    # Embed tất cả tokens (text + audio codebooks)
    embeds = self._embed_tokens(tokens)  # [bsz, seq_len, n_codebooks+1, dim]

    # Xác định vị trí audio frames
    audio_mask = tokens_mask[:, :, 0]  # [bsz, seq_len] – True tại các frame có audio

    # Targets và ground-truth embeddings cho audio codebooks
    target_tokens = tokens[audio_mask][:, :-1]          # [total_audio_frames, n_codebooks]
    c_embeds = embeds[:, :, :-1, :][audio_mask]         # [total_audio_frames, n_codebooks, dim]

    # Backbone input: sum embeddings theo chiều codebook
    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)  # [bsz, seq_len, dim]

    # Backbone attention mask
    padding_mask = tokens_mask[:, :, 0] | tokens_mask[:, :, -1]  # text hoặc padding
    backbone_attn_mask = _create_causal_mask(seq_len, device)
    padding_3d = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)
    backbone_attn_mask = backbone_attn_mask.unsqueeze(0) * padding_3d
    backbone_attn_mask = backbone_attn_mask | torch.eye(seq_len, device=device).bool().unsqueeze(0).expand(bsz, -1, -1)

    input_pos = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
    h = self.backbone(h, input_pos=input_pos, mask=backbone_attn_mask).to(dtype=dtype)

    # Hidden state để predict codebook 0 của frame tiếp theo
    predict_mask = torch.roll(audio_mask, -1, dims=1)
    audio_h = h[predict_mask]  # [total_audio_frames, dim]

    # ================
    # Codebook 0 Loss
    # ================
    c0_logits = self.codebook0_head(audio_h)  # [total_audio_frames, vocab_size]
    c0_target = target_tokens[:, 0]
    c0_loss = F.cross_entropy(c0_logits, c0_target)

    # =================================================
    # Decoder với Delay Pattern (within-frame autoregressive)
    # =================================================

    # Compute amortization: chỉ train decoder trên 1/16 frames ngẫu nhiên
    num_frames = audio_h.size(0)
    subset_size = max(num_frames // 16, 1)
    indices = torch.randperm(num_frames, device=device)[:subset_size]

    audio_h_sub = audio_h[indices]                    # [subset, dim]
    c_embeds_sub = c_embeds[indices]                  # [subset, n_codebooks, dim]
    target_acoustic = target_tokens[indices][:, 1:]    # [subset, n_codebooks-1] – targets cho cb1..cbN-1

    # Xây input decoder autoregressively (teacher-forcing trong frame)
    # Bắt đầu bằng semantic hidden state
    decoder_inputs = [audio_h_sub.unsqueeze(1)]  # list chứa các step, bắt đầu với [subset, 1, dim]

    for k in range(1, c_embeds_sub.size(1)):  # từ codebook 1 đến N-1
        prev_emb = c_embeds_sub[:, k-1, :].unsqueeze(1)  # [subset, 1, dim] – embed của codebook trước (ground-truth)
        next_input = torch.cat([decoder_inputs[-1], prev_emb], dim=1)  # nối thêm vào sequence
        decoder_inputs.append(next_input)

    # Ghép tất cả các step sau semantic thành input đầy đủ cho decoder
    decoder_embeds = torch.cat(decoder_inputs[1:], dim=1)  # [subset, n_codebooks-1, dim]

    N, n_acoustic_cb, _ = decoder_embeds.size()

    # Position và causal mask cho decoder sequence
    c_pos = torch.arange(1, n_acoustic_cb + 1, device=device).unsqueeze(0).expand(N, -1).long()
    decoder_causal_mask = _create_causal_mask(n_acoustic_cb, device).expand(N, -1, -1)

    # Forward decoder
    decoder_h = self.decoder(
        self.projection(decoder_embeds),
        input_pos=c_pos,
        mask=decoder_causal_mask
    ).to(dtype=dtype)  # [subset, n_acoustic_cb, dim]

    # Dự đoán acoustic codebooks (cb1 đến cbN-1)
    # Fix shape mismatch: chỉ dùng weight từ index 1 trở đi trong audio_head
    acoustic_head = self.audio_head[1:, :, :]  # [n_codebooks-1, dim, vocab_size]

    c_logits = torch.einsum("bsd,sdv->bsv", decoder_h, acoustic_head)
    # Hoặc: c_logits = decoder_h @ acoustic_head.transpose(1, 2)

    # Decoder loss
    c_loss = F.cross_entropy(c_logits.reshape(-1, c_logits.size(-1)), target_acoustic.reshape(-1))

    # Total loss (giữ nguyên công thức gốc)
    loss = 2 * ((1 - self.decoder_loss_weight) * c0_loss + self.decoder_loss_weight * c_loss)

    return loss
