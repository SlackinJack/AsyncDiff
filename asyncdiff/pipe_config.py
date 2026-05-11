def splite_model(pipe, pipe_id, n):
    if pipe_id in ["flux", "sd3", "wani2v", "want2v", "zimage"]:
        transformer = pipe.transformer
    else:
        unet = pipe.unet

    if pipe_id == "svd":
        if n == 1:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 2:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ), (
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 3:
            return [(
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.down_blocks[2],
            ), (
                unet.down_blocks[3],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
            ), (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 4:
            return [(
                unet.down_blocks[1].resnets[0],
                unet.down_blocks[1].attentions[0],
                unet.conv_in,
                unet.down_blocks[0],
            ), (
                unet.down_blocks[1].resnets[1],
                unet.down_blocks[1].attentions[1],
                *unet.down_blocks[1].downsamplers,
                *unet.down_blocks[2:4],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ), (
                unet.up_blocks[2],
                unet.up_blocks[3].resnets[0],
                unet.up_blocks[3].attentions[0],
            ), (
                unet.up_blocks[3].resnets[1],
                unet.up_blocks[3].attentions[1],
                unet.up_blocks[3].resnets[2],
                unet.up_blocks[3].attentions[2],
                unet.conv_norm_out,
                unet.conv_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "sd2":
        if n == 1:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                *unet.up_blocks[:1],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[1],
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[2],
                unet.up_blocks[1].attentions[2],
                *unet.up_blocks[1].upsamplers,
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 2:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                *unet.up_blocks[:1],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[1],
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[2],
            ), (
                unet.up_blocks[1].attentions[2],
                *unet.up_blocks[1].upsamplers,
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 3:
            return [(
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.down_blocks[2],
                unet.down_blocks[3],
            ), (
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
            ), (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 4:
            return [(
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1]
            ), (
                *unet.down_blocks[2:4],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ), (
                unet.up_blocks[2],
                unet.up_blocks[3].resnets[0],
            ), (
                unet.up_blocks[3].attentions[0],
                unet.up_blocks[3].resnets[1],
                unet.up_blocks[3].attentions[1],
                unet.up_blocks[3].resnets[2],
                unet.up_blocks[3].attentions[2],
                unet.conv_norm_out,
                unet.conv_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "sd1":
        if n == 1:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                *unet.up_blocks[:1],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[1],
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[2],
                unet.up_blocks[1].attentions[2],
                *unet.up_blocks[1].upsamplers,
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 2:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                *unet.up_blocks[:1],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[1],
            ), (
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[2],
                unet.up_blocks[1].attentions[2],
                *unet.up_blocks[1].upsamplers,
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 3:
            return [(
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.down_blocks[2],
                unet.down_blocks[3],
            ), (
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2].resnets[0],
                unet.up_blocks[2].attentions[0],
                unet.up_blocks[2].resnets[1],
                unet.up_blocks[2].attentions[1],
                unet.up_blocks[2].resnets[2],
            ), (
                unet.up_blocks[2].attentions[2],
                *unet.up_blocks[2].upsamplers,
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 4:
            return [(
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1]
            ), (
                *unet.down_blocks[2:4],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[1],
            ), (
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[2],
                unet.up_blocks[1].attentions[2],
                *unet.up_blocks[1].upsamplers,
                unet.up_blocks[2],
            ), (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "sdxl":
        if n == 1:
            return [(
                unet.down_blocks[2],
                unet.mid_block,
                unet.up_blocks[0].resnets[0],
                unet.up_blocks[0].attentions[0],
                unet.up_blocks[0].resnets[1],
                unet.up_blocks[0].attentions[1],
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.up_blocks[1],
                unet.up_blocks[2],
                unet.conv_norm_out,
                unet.conv_out,
                unet.up_blocks[0].resnets[2],
                unet.up_blocks[0].attentions[2],
                *unet.up_blocks[0].upsamplers,
            )]
        elif n == 2:
            return [(
                unet.down_blocks[2],
                unet.mid_block,
                unet.up_blocks[0].resnets[0],
                unet.up_blocks[0].attentions[0],
                unet.up_blocks[0].resnets[1],
                unet.up_blocks[0].attentions[1],
            ), (
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.up_blocks[1],
                unet.up_blocks[2],
                unet.conv_norm_out,
                unet.conv_out,
                unet.up_blocks[0].resnets[2],
                unet.up_blocks[0].attentions[2],
                *unet.up_blocks[0].upsamplers,
            )]
        elif n == 3:
            return [(
                unet.down_blocks[2],
                unet.mid_block,
                unet.up_blocks[0].resnets[0],
            ), (
                unet.up_blocks[0].attentions[0],
                unet.up_blocks[0].resnets[1],
                unet.up_blocks[0].attentions[1],
                unet.up_blocks[0].resnets[2],
                unet.up_blocks[0].attentions[2],
                *unet.up_blocks[0].upsamplers,
            ), (
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.up_blocks[1],
                unet.up_blocks[2],
                unet.conv_norm_out,
                unet.conv_out,
            )]
        elif n == 4:
            return [(
                unet.down_blocks[1].attentions[0],
                unet.down_blocks[1].resnets[1],
                unet.down_blocks[1].attentions[1],
                *unet.down_blocks[1].downsamplers,
                unet.down_blocks[2]
            ), (
                unet.mid_block,
                unet.up_blocks[0].resnets[0],
                unet.up_blocks[0].attentions[0],
            ), (
                unet.up_blocks[0].resnets[1],
                unet.up_blocks[0].attentions[1],
                unet.up_blocks[0].resnets[2],
                unet.up_blocks[0].attentions[2],
                *unet.up_blocks[0].upsamplers,
            ), (
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1].resnets[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
                unet.conv_norm_out,
                unet.conv_out,
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "ad":
        if n == 1:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 2:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ), (
                unet.up_blocks[2],
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 3:
            return [(
                unet.conv_in,
                *unet.down_blocks,
            ), (
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
            ), (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "sdup":
        if n == 1:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[1],
                unet.up_blocks[1].attentions[2],
                unet.up_blocks[1].resnets[2],
                *unet.up_blocks[0].upsamplers,
                unet.up_blocks[2],
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 2:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[1],
            ), (
                unet.up_blocks[1].attentions[2],
                unet.up_blocks[1].resnets[2],
                *unet.up_blocks[0].upsamplers,
                unet.up_blocks[2],
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 3:
            return [(
                unet.conv_in,
                *unet.down_blocks,
            ), (
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2].attentions[0],
                unet.up_blocks[2].resnets[0],
                unet.up_blocks[2].attentions[1],
            ), (
                unet.up_blocks[2].resnets[1],
                unet.up_blocks[2].attentions[2],
                unet.up_blocks[2].resnets[2],
                *unet.up_blocks[2].upsamplers,
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 4:
            return [(
                unet.conv_in,
                *unet.down_blocks[0:3],
            ), (
                unet.down_blocks[3],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ), (
                unet.up_blocks[2],
            ), (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "sd3":
        if n == 1:
            return [(
                *transformer.transformer_blocks[0:12],
                *transformer.transformer_blocks[12:24],
                transformer.norm_out,
                transformer.proj_out
            )]
        elif n == 2:
            return [(
                *transformer.transformer_blocks[0:12],
            ), (
                *transformer.transformer_blocks[12:24],
                transformer.norm_out,
                transformer.proj_out
            )]
        elif n == 3:
            return [(
                *transformer.transformer_blocks[0:8],
            ), (
                *transformer.transformer_blocks[8:16],
            ), (
                *transformer.transformer_blocks[16:24],
                transformer.norm_out,
                transformer.proj_out
            )]
        elif n == 4:
            return [(
                *transformer.transformer_blocks[0:6],
            ), (
                *transformer.transformer_blocks[6:12],
            ), (
                *transformer.transformer_blocks[12:18],
            ), (
                *transformer.transformer_blocks[18:24],
                transformer.norm_out,
                transformer.proj_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id in ["flux"]:
        """
FluxTransformer2DModel(
  (pos_embed): FluxPosEmbed()
  (time_text_embed): CombinedTimestepGuidanceTextProjEmbeddings(
    (time_proj): Timesteps()
    (timestep_embedder): TimestepEmbedding(
      (linear_1): Linear(in_features=256, out_features=3072, bias=True)
      (act): SiLU()
      (linear_2): Linear(in_features=3072, out_features=3072, bias=True)
    )
    (guidance_embedder): TimestepEmbedding(
      (linear_1): Linear(in_features=256, out_features=3072, bias=True)
      (act): SiLU()
      (linear_2): Linear(in_features=3072, out_features=3072, bias=True)
    )
    (text_embedder): PixArtAlphaTextProjection(
      (linear_1): Linear(in_features=768, out_features=3072, bias=True)
      (act_1): SiLU()
      (linear_2): Linear(in_features=3072, out_features=3072, bias=True)
    )
  )
  (context_embedder): Linear(in_features=4096, out_features=3072, bias=True)
  (x_embedder): Linear(in_features=64, out_features=3072, bias=True)
  (transformer_blocks): ModuleList(
    (0-18): 19 x FluxTransformerBlock(
      (norm1): AdaLayerNormZero(
        (silu): SiLU()
        (linear): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([18432, 3072]), original_shape=torch.Size([18432, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([18432, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      )
      (norm1_context): AdaLayerNormZero(
        (silu): SiLU()
        (linear): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([18432, 3072]), original_shape=torch.Size([18432, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([18432, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      )
      (attn): FluxAttention(
        (norm_q): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (norm_k): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (to_q): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (to_k): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (to_v): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (to_out): ModuleList(
          (0): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
          (1): Dropout(p=0.0, inplace=False)
        )
        (norm_added_q): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (norm_added_k): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (add_q_proj): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (add_k_proj): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (add_v_proj): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (to_add_out): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
      )
      (norm2): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      (ff): FeedForward(
        (net): ModuleList(
          (0): GELU(
            (proj): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([12288, 3072]), original_shape=torch.Size([12288, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([12288, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 12288]), original_shape=torch.Size([3072, 12288]), original_stride=(12288, 1), quantized_weight_shape=torch.Size([3072, 192, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        )
      )
      (norm2_context): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      (ff_context): FeedForward(
        (net): ModuleList(
          (0): GELU(
            (proj): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([12288, 3072]), original_shape=torch.Size([12288, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([12288, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 12288]), original_shape=torch.Size([3072, 12288]), original_stride=(12288, 1), quantized_weight_shape=torch.Size([3072, 192, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        )
      )
    )
  )
  (single_transformer_blocks): ModuleList(
    (0): FluxSingleTransformerBlock(
      (norm): AdaLayerNormZeroSingle(
        (silu): SiLU()
        (linear): Linear(in_features=3072, out_features=9216, bias=True)
        (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      )
      (proj_mlp): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([12288, 3072]), original_shape=torch.Size([12288, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([12288, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
      (act_mlp): GELU(approximate='tanh')
      (proj_out): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 15360]), original_shape=torch.Size([3072, 15360]), original_stride=(15360, 1), quantized_weight_shape=torch.Size([3072, 240, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
      (attn): FluxAttention(
        (norm_q): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (norm_k): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (to_q): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (to_k): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (to_v): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
      )
    )
    (1-37): 37 x FluxSingleTransformerBlock(
      (norm): AdaLayerNormZeroSingle(
        (silu): SiLU()
        (linear): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([9216, 3072]), original_shape=torch.Size([9216, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([9216, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      )
      (proj_mlp): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([12288, 3072]), original_shape=torch.Size([12288, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([12288, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
      (act_mlp): GELU(approximate='tanh')
      (proj_out): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 15360]), original_shape=torch.Size([3072, 15360]), original_stride=(15360, 1), quantized_weight_shape=torch.Size([3072, 240, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
      (attn): FluxAttention(
        (norm_q): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (norm_k): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (to_q): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (to_k): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
        (to_v): SDNQLinear(original_class=Linear forward_func=<function quantized_linear_forward_int8_matmul at 0x7a4d82a33920> sdnq_dequantizer=SDNQDequantizer(result_dtype=torch.float32, result_shape=torch.Size([3072, 3072]), original_shape=torch.Size([3072, 3072]), original_stride=(3072, 1), quantized_weight_shape=torch.Size([3072, 48, 64]), weights_dtype='int4', quantized_matmul_dtype='int8', group_size=64, svd_rank=16, svd_steps=4, use_quantized_matmul=True, re_quantize_for_matmul=True, use_stochastic_rounding=False, layer_class_name='Linear', is_packed=True, is_unsigned=False, is_integer=True, is_integer_matmul=True))
      )
    )
  )
  (norm_out): AdaLayerNormContinuous(
    (silu): SiLU()
    (linear): Linear(in_features=3072, out_features=6144, bias=True)
    (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
  )
  (proj_out): Linear(in_features=3072, out_features=64, bias=True)
)
"""
        if n == 1:
            return [(
                *transformer.transformer_blocks[0:19],
                *transformer.single_transformer_blocks[0:38],
                transformer.norm_out,
                transformer.proj_out
            )]
        elif n == 2:
            return [(
                *transformer.transformer_blocks[0:19],
                *transformer.single_transformer_blocks[0:11]
            ), (
                *transformer.single_transformer_blocks[11:38],
                transformer.norm_out,
                transformer.proj_out
            )]
        elif n == 3:
            return [(
                *transformer.transformer_blocks[0:19],
            ), (
                *transformer.single_transformer_blocks[0:19],
            ), (
                *transformer.transformer_blocks[19:38],
                transformer.norm_out,
                transformer.proj_out
            )]
        elif n == 4:
            return [(
                *transformer.transformer_blocks[0:19],
            ), (
                *transformer.single_transformer_blocks[0:16],
            ), (
                *transformer.single_transformer_blocks[16:32],
            ), (
                *transformer.single_transformer_blocks[32:38],
                transformer.norm_out,
                transformer.proj_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "zimage":
        """
ZImageTransformer2DModel(
  (all_x_embedder): ModuleDict(
    (2-1): Linear(in_features=64, out_features=3840, bias=True)
  )
  (all_final_layer): ModuleDict(
    (2-1): FinalLayer(
      (norm_final): LayerNorm((3840,), eps=1e-06, elementwise_affine=False)
      (linear): Linear(in_features=3840, out_features=64, bias=True)
      (adaLN_modulation): Sequential(
        (0): SiLU()
        (1): Linear(in_features=256, out_features=3840, bias=True)
      )
    )
  )
  (noise_refiner): ModuleList(
    (0-1): 2 x ZImageTransformerBlock(
      (attention): Attention(
        (norm_q): RMSNorm()
        (norm_k): RMSNorm()
        (to_q): Linear(in_features=3840, out_features=3840, bias=False)
        (to_k): Linear(in_features=3840, out_features=3840, bias=False)
        (to_v): Linear(in_features=3840, out_features=3840, bias=False)
        (to_out): ModuleList(
          (0): Linear(in_features=3840, out_features=3840, bias=False)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=3840, out_features=10240, bias=False)
        (w2): Linear(in_features=10240, out_features=3840, bias=False)
        (w3): Linear(in_features=3840, out_features=10240, bias=False)
      )
      (attention_norm1): RMSNorm()
      (ffn_norm1): RMSNorm()
      (attention_norm2): RMSNorm()
      (ffn_norm2): RMSNorm()
      (adaLN_modulation): Sequential(
        (0): Linear(in_features=256, out_features=15360, bias=True)
      )
    )
  )
  (context_refiner): ModuleList(
    (0-1): 2 x ZImageTransformerBlock(
      (attention): Attention(
        (norm_q): RMSNorm()
        (norm_k): RMSNorm()
        (to_q): Linear(in_features=3840, out_features=3840, bias=False)
        (to_k): Linear(in_features=3840, out_features=3840, bias=False)
        (to_v): Linear(in_features=3840, out_features=3840, bias=False)
        (to_out): ModuleList(
          (0): Linear(in_features=3840, out_features=3840, bias=False)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=3840, out_features=10240, bias=False)
        (w2): Linear(in_features=10240, out_features=3840, bias=False)
        (w3): Linear(in_features=3840, out_features=10240, bias=False)
      )
      (attention_norm1): RMSNorm()
      (ffn_norm1): RMSNorm()
      (attention_norm2): RMSNorm()
      (ffn_norm2): RMSNorm()
    )
  )
  (t_embedder): TimestepEmbedder(
    (mlp): Sequential(
      (0): Linear(in_features=256, out_features=1024, bias=True)
      (1): SiLU()
      (2): Linear(in_features=1024, out_features=256, bias=True)
    )
  )
  (cap_embedder): Sequential(
    (0): RMSNorm()
    (1): Linear(in_features=2560, out_features=3840, bias=True)
  )
  (layers): ModuleList(
    (0-29): 30 x ZImageTransformerBlock(
      (attention): Attention(
        (norm_q): RMSNorm()
        (norm_k): RMSNorm()
        (to_q): Linear(in_features=3840, out_features=3840, bias=False)
        (to_k): Linear(in_features=3840, out_features=3840, bias=False)
        (to_v): Linear(in_features=3840, out_features=3840, bias=False)
        (to_out): ModuleList(
          (0): Linear(in_features=3840, out_features=3840, bias=False)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=3840, out_features=10240, bias=False)
        (w2): Linear(in_features=10240, out_features=3840, bias=False)
        (w3): Linear(in_features=3840, out_features=10240, bias=False)
      )
      (attention_norm1): RMSNorm()
      (ffn_norm1): RMSNorm()
      (attention_norm2): RMSNorm()
      (ffn_norm2): RMSNorm()
      (adaLN_modulation): Sequential(
        (0): Linear(in_features=256, out_features=15360, bias=True)
      )
    )
  )
)
"""
        # NOTE: "feed_forward" target will severely degrade quality
        # NOTE: "adaLN_modulation" can also be a target, but slightly degrades quality
        # NOTE: probably can add attention/ffn norm1/norm2 as targets, but OOM on 16gb
        def get(start, end):
            targets = ["attention"]
            out = []
            for i in range(start, end):
                for t in targets:
                    out.append(getattr(transformer.layers[i], t))
            return tuple(out)

        if n == 1:
            return [
                get(0, 30) + (transformer.all_final_layer,)
            ]
        elif n == 2:
            return [
                get(0, 15),
                get(15, 30) + (transformer.all_final_layer,)
            ]
        elif n == 3:
            return [
                get(0, 10),
                get(10, 20),
                get(20, 30) + (transformer.all_final_layer,)
            ]
        elif n == 4:
            return [
                get(0, 8),
                get(8, 16),
                get(16, 24),
                get(24, 30) + (transformer.all_final_layer,)
            ]
        else:
            raise NotImplementedError
    elif pipe_id in ["wani2v", "want2v"]: # TODO: might have to differentiate between 5b/14b, wan2.1/wan2.2
        if n == 1:
            return [(
                *transformer.blocks[0:39],
                transformer.norm_out,
                transformer.proj_out,
            )]
        elif n == 2:
            return [(
                *transformer.blocks[0:20],
            ), (
                *transformer.blocks[20:39],
                transformer.norm_out,
                transformer.proj_out,
            )]
        elif n == 3:
            return [(
                *transformer.blocks[0:14],
            ), (
                *transformer.blocks[14:28],
            ), (
                *transformer.blocks[28:39],
                transformer.norm_out,
                transformer.proj_out,
            )]
        elif n == 4:
            return [(
                *transformer.blocks[0:10],
            ), (
                *transformer.blocks[10:20],
            ), (
                *transformer.blocks[20:30],
            ), (
                *transformer.blocks[30:39],
                transformer.norm_out,
                transformer.proj_out,
            )]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

