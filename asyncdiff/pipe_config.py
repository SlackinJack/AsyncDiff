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
        if n == 1:
            return [(
                *transformer.layers[0:30],
                transformer.all_final_layer,
                transformer.noise_refiner,
                transformer.context_refiner,
            )]
        elif n == 2:
            return [(
                *transformer.layers[0:17],
            ), (
                *transformer.layers[17:30],
                transformer.all_final_layer,
                transformer.noise_refiner,
                transformer.context_refiner,
            )]
        elif n == 3:
            return [(
                *transformer.layers[0:12],
            ), (
                *transformer.layers[12:24],
            ), (
                *transformer.layers[24:30],
                transformer.all_final_layer,
                transformer.noise_refiner,
                transformer.context_refiner,
            )]
        elif n == 4:
            return [(
                *transformer.layers[0:9],
            ), (
                *transformer.layers[9:18],
            ), (
                *transformer.layers[18:27],
            ), (
                *transformer.layers[27:30],
                transformer.all_final_layer,
                transformer.noise_refiner,
                transformer.context_refiner,
            )]
            """return [(
                *tuple(
                    module
                    for i in range(0, 8)
                    for module in (
                        transformer.layers[i].attention,
                    )
                ),
            ), (
                *tuple(
                    module
                    for i in range(8, 16)
                    for module in (
                        transformer.layers[i].attention,
                    )
                ),
            ), (
                *tuple(
                    module
                    for i in range(16, 24)
                    for module in (
                        transformer.layers[i].attention,
                    )
                ),
            ), (
                transformer.all_final_layer,
                *tuple(
                    module
                    for i in range(24, 30)
                    for module in (
                        transformer.layers[i].attention,
                    )
                ),
            )]"""
        else:
            raise NotImplementedError
    elif pipe_id in ["wani2v", "want2v"]: # TODO: might have to differentiate between 5b/14b, wan2.1/wan2.2
        """
WanTransformer3DModel(                                                                                       
  (rope): WanRotaryPosEmbed()                                                                                
  (patch_embedding): Conv3d(36, 5120, kernel_size=(1, 2, 2), stride=(1, 2, 2))                               
  (condition_embedder): WanTimeTextImageEmbedding(                                                           
    (timesteps_proj): Timesteps()                                                                            
    (time_embedder): TimestepEmbedding(                                                                      
      (linear_1): Linear(in_features=256, out_features=5120, bias=True)                                        
      (act): SiLU()                                                                                          
      (linear_2): Linear(in_features=5120, out_features=5120, bias=True)                                     
    )                                                                                                        
    (act_fn): SiLU()                                  
    (time_proj): Linear(in_features=5120, out_features=30720, bias=True)                                       
    (text_embedder): PixArtAlphaTextProjection(                                                              
      (linear_1): Linear(in_features=4096, out_features=5120, bias=True)                                     
      (act_1): GELU(approximate='tanh')                                                                      
      (linear_2): Linear(in_features=5120, out_features=5120, bias=True)                                     
    )
  (image_embedder): WanImageEmbedding(              
      (norm1): FP32LayerNorm((1280,), eps=1e-05, elementwise_affine=True)                                    
      (ff): FeedForward(                                                                                     
        (net): ModuleList(                            
          (0): GELU(                                                                                         
            (proj): Linear(in_features=1280, out_features=1280, bias=True)                                   
          )                                                                                                  
          (1): Dropout(p=0.0, inplace=False)          
          (2): Linear(in_features=1280, out_features=5120, bias=True)                                        
        )                                                                                                    
      )                                               
      (norm2): FP32LayerNorm((5120,), eps=1e-05, elementwise_affine=True)                                      
    )                                                                                                        
  )                                                                                                          
  (blocks): ModuleList(                                                                                      
    (0-39): 40 x WanTransformerBlock(                                                                        
      (norm1): FP32LayerNorm((5120,), eps=1e-06, elementwise_affine=False)                                     
      (attn1): WanAttention(                                                                                 
        (to_q): Lanear(in_features=5120, out_features=5120, bias=True)                                       
        (to_k): Linear(in_features=5120, out_features=5120, bias=True)                                       
        (to_v): Lanear(in_features=5120, out_features=5120, bias=True)                                       
        (to_out): ModuleList(                                                                                
          (0): Linear(in_features=5120, out_features=5120, bias=True)                                        
          (1): Dropout(p=0.0, inplace=False)                                                                   
        )                                                                                                    
        (norm_q): RMSNorm((5120,), eps=1e-06, elementwise_affine=True)                                         
        (norm_k): RMSNorm((5120,), eps=1e-06, elementwise_affine=True)                                       
      )                                                                                                        
      (attn2): WanAttention(                                                                                 
        (to_q): Linear(in_features=5120, out_features=5120, bias=True)                                       
        (to_k): Linear(in_features=5120, out_features=5120, bias=True)                                       
        (to_v): Linear(in_features=5120, out_features=5120, bias=True)                                       
        (to_out): ModuleList(                                                                                  
          (0): Linear(in_features=5120, out_features=5120, bias=True)                                        
          (1): Dropout(p=0.0, inplace=False)                                                                 
        )                                             
        (norm_q): RMSNorm((5120,), eps=1e-06, elementwise_affine=True)                                       
        (norm_k): RMSNorm((5120,), eps=1e-06, elementwise_affine=True)                                       
        (add_k_proj): Linear(in_features=5120, out_features=5120, bias=True)                                 
        (add_v_proj): Linear(in_features=5120, out_features=5120, bias=True)                                 
        (norm_added_k): RMSNorm((5120,), eps=1e-06, elementwise_affine=True)                                 
      )                                                                                                      
      (norm2): FP32LayerNorm((5120,), eps=1e-06, elementwise_affine=True)                                    
      (ffn): FeedForward(                                                                                    
        (net): ModuleList(                                                                                     
          (0): GELU(                                                                                         
            (proj): Linear(in_features=5120, out_features=13824, bias=True)                                  
          )                                                                                                  
          (1): Dropout(p=0.0, inplace=False)          
          (2): Linear(in_features=13824, out_features=5120, bias=True)                                         
        )                                                                                                    
      )                                                                                                      
      (norm3): FP32LayerNorm((5120,), eps=1e-06, elementwise_affine=False)                                   
    )                                                                                                        

  (norm_out): FP32LayerNorm((5120,), eps=1e-06, elementwise_affine=False)                                    
  (proj_out): Linear(in_features=5120, out_features=64, bias=True)                                           
)  
"""
        if n == 1:
            return [(
                *tuple(
                    module
                    for i in range(0, 40)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            )]
        elif n == 2:
            return [(
                *tuple(
                    module
                    for i in range(0, 20)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            ), (
                *tuple(
                    module
                    for i in range(20, 40)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            )]
        elif n == 3:
            return [(
                *tuple(
                    module
                    for i in range(0, 13)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            ), (
                *tuple(
                    module
                    for i in range(13, 26)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            ), (
                *tuple(
                    module
                    for i in range(26, 40)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            )]
        elif n == 4:
            return [(
                *tuple(
                    module
                    for i in range(0, 10)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            ), (
                *tuple(
                    module
                    for i in range(10, 20)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            ), (
                *tuple(
                    module
                    for i in range(20, 30)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            ), (
                *tuple(
                    module
                    for i in range(30, 40)
                    for module in (
                        transformer.blocks[i].attn1,
                        transformer.blocks[i].attn2,
                    )
                ),
            )]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

