def splite_model(pipe, pipe_id, n):
    if pipe_id in ["flux", "krea2", "sd3", "wani2v", "want2v", "zimage"]:
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
                  (linear_1): Linear()
                  (act): SiLU()
                  (linear_2): Linear()
                )
                (guidance_embedder): TimestepEmbedding(
                  (linear_1): Linear()
                  (act): SiLU()
                  (linear_2): Linear()
                )
                (text_embedder): PixArtAlphaTextProjection(
                  (linear_1): Linear()
                  (act_1): SiLU()
                  (linear_2): Linear()
                )
              )
              (context_embedder): Linear()
              (x_embedder): Linear()
              (transformer_blocks): ModuleList(
                (0-18): 19 x FluxTransformerBlock(
                  (norm1): AdaLayerNormZero(
                    (silu): SiLU()
                    (linear): Linear()
                    (norm): LayerNorm()
                  )
                  (norm1_context): AdaLayerNormZero(
                    (silu): SiLU()
                    (linear): Linear()
                    (norm): LayerNorm()
                  )
                  (attn): FluxAttention(
                    (norm_q): RMSNorm()
                    (norm_k): RMSNorm()
                    (to_q): Linear()
                    (to_k): Linear()
                    (to_v): Linear()
                    (to_out): ModuleList(
                      (0): Linear()
                      (1): Dropout()
                    )
                    (norm_added_q): RMSNorm()
                    (norm_added_k): RMSNorm()
                    (add_q_proj): Linear()
                    (add_k_proj): Linear()
                    (add_v_proj): Linear()
                    (to_add_out): Linear()
                  )
                  (norm2): LayerNorm()
                  (ff): FeedForward(
                    (net): ModuleList(
                      (0): GELU(
                        (proj): Linear()
                      )
                      (1): Dropout()
                      (2): Linear()
                    )
                  )
                  (norm2_context): LayerNorm()
                  (ff_context): FeedForward(
                    (net): ModuleList(
                      (0): GELU(
                        (proj): Linear()
                      )
                      (1): Dropout()
                      (2): Linear()
                    )
                  )
                )
              )
              (single_transformer_blocks): ModuleList(
                (0): FluxSingleTransformerBlock(
                  (norm): AdaLayerNormZeroSingle(
                    (silu): SiLU()
                    (linear): Linear()
                    (norm): LayerNorm()
                  )
                  (proj_mlp): Linear()
                  (act_mlp): GELU()
                  (proj_out): Linear()
                  (attn): FluxAttention(
                    (norm_q): RMSNorm()
                    (norm_k): RMSNorm()
                    (to_q): Linear()
                    (to_k): Linear()
                    (to_v): Linear()
                  )
                )
                (1-37): 37 x FluxSingleTransformerBlock(
                  (norm): AdaLayerNormZeroSingle(
                    (silu): SiLU()
                    (linear): Linear()
                    (norm): LayerNorm()
                  )
                  (proj_mlp): Linear()
                  (act_mlp): GELU()
                  (proj_out): Linear()
                  (attn): FluxAttention(
                    (norm_q): RMSNorm()
                    (norm_k): RMSNorm()
                    (to_q): Linear()
                    (to_k): Linear()
                    (to_v): Linear()
                  )
                )
              )
              (norm_out): AdaLayerNormContinuous(
                (silu): SiLU()
                (linear): Linear()
                (norm): LayerNorm()
              )
              (proj_out): Linear()
            )
        """
        if n == 1:
            return [(
                *tuple(module for i in range(0, 19) for module in (
                    transformer.transformer_blocks[i].attn,
                )),
                *tuple(module for i in range(0, 38) for module in (
                    transformer.single_transformer_blocks[i].attn,
                )),
            )]
        elif n == 2:
            return [(
                *tuple(module or i in range(0, 19) for module in (
                    transformer.transformer_blocks[i].attn,
                )),
                *tuple(module for i in range(0, 11) for module in (
                    transformer.single_transformer_blocks[i].attn,
                )),
            ), (
                *tuple(module for i in range(11, 38) for module in (
                    transformer.single_transformer_blocks[i].attn,
                )),
            )]
        elif n == 3:
            return [(
                *tuple(module for i in range(0, 19) for module in (
                    transformer.transformer_blocks[i].attn,
                )),
            ), (
                *tuple(module for i in range(0, 19) for module in (
                    transformer.single_transformer_blocks[i].attn,
                )),
            ), (
                *tuple(module for i in range(19, 38) for module in (
                    transformer.single_transformer_blocks[i].attn,
                )),
            )]
        elif n == 4:
            return [(
                *tuple(module for i in range(0, 19) for module in (
                    transformer.transformer_blocks[i].attn,
                )),
            ), (
                *tuple(module for i in range(0, 16) for module in (
                    transformer.single_transformer_blocks[i].attn,
                )),
            ), (
                *tuple(module for i in range(16, 32) for module in (
                    transformer.single_transformer_blocks[i].attn,
                )),
            ), (
                *tuple(module for i in range(32, 38) for module in (
                    transformer.single_transformer_blocks[i].attn,
                )),
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "krea2":
        """
            Krea2Transformer2DModel(
              (img_in): Linear()
              (time_embed): Krea2TimestepEmbedding(
                (linear_1): Linear()
                (linear_2): Linear()
              )
              (time_mod_proj): Linear()
              (text_fusion): Krea2TextFusion(
                (layerwise_blocks): ModuleList(
                  (0-1): 2 x Krea2TextFusionBlock(
                    (norm1): Krea2RMSNorm()
                    (norm2): Krea2RMSNorm()
                    (attn): Krea2Attention(
                      (to_q): Linear()
                    (to_k): Linear()
                      (to_v): Linear()
                      (to_gate): Linear()
                      (norm_q): Krea2RMSNorm()
                      (norm_k): Krea2RMSNorm()
                      (to_out): ModuleList(
                        (0): Linear()
                        (1): Dropout()
                      )
                    )
                    (ff): Krea2SwiGLU(
                      (gate): Linear()
                      (up): Linear()
                      (down): Linear()
                    )
                  )
                )
                (projector): Linear()
                (refiner_blocks): ModuleList(
                  (0-1): 2 x Krea2TextFusionBlock(
                    (norm1): Krea2RMSNorm()
                    (norm2): Krea2RMSNorm()
                    (attn): Krea2Attention(
                      (to_q): Linear()
                      (to_k): Linear()
                      (to_v): Linear()
                    (to_gate): Linear()
                      (norm_q): Krea2RMSNorm()
                      (norm_k): Krea2RMSNorm()
                      (to_out): ModuleList(
                        (0): Linear()
                        (1): Dropout()
                      )
                    )
                    (ff): Krea2SwiGLU(
                      (gate): Linear()
                      (up): Linear()
                      (down): Linear()
                    )
                  )
                )
              )
              (txt_in): Krea2TextProjection(
                (norm): Krea2RMSNorm()
                (linear_1): Linear()
                (linear_2): Linear()
              )
              (rotary_emb): Krea2RotaryPosEmbed()
              (transformer_blocks): ModuleList(
                (0-27): 28 x Krea2TransformerBlock(
                  (norm1): Krea2RMSNorm()
                  (norm2): Krea2RMSNorm()
                  (attn): Krea2Attention(
                    (to_q): Linear()
                    (to_k): Linear()
                    (to_v): Linear()
                    (to_gate): Linear()
                    (norm_q): Krea2RMSNorm()
                    (norm_k): Krea2RMSNorm()
                    (to_out): ModuleList(
                      (0): Linear()
                      (1): Dropout()
                    )
                  )
                  (ff): Krea2SwiGLU(
                    (gate): Linear()
                    (up): Linear()
                    (down): Linear()
                  )
                )
              )
              (final_layer): Krea2FinalLayer(
                (norm): Krea2RMSNorm()
                (linear): Linear()
              )
            )
        """
        if n == 1:
            return [(
                *tuple(module for i in range(0, 2) for module in (
                    transformer.text_fusion.layerwise_blocks[i],
                    transformer.text_fusion.refiner_blocks[i],
                )),
                transformer.text_fusion.projector,
                transformer.txt_in,
                *tuple(module for i in range(0, 28) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
                transformer.final_layer,
            )]
        elif n == 2:
            return [(
                *tuple(module for i in range(0, 2) for module in (
                    transformer.text_fusion.layerwise_blocks[i],
                    transformer.text_fusion.refiner_blocks[i],
                )),
                transformer.text_fusion.projector,
                transformer.txt_in,
                *tuple(module for i in range(0, 12) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
            ), (
                *tuple(module for i in range(12, 28) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
                transformer.final_layer,
            )]
        elif n == 3:
            return [(
                *tuple(module for i in range(0, 2) for module in (
                    transformer.text_fusion.layerwise_blocks[i],
                    transformer.text_fusion.refiner_blocks[i],
                )),
                transformer.text_fusion.projector,
                transformer.txt_in,
                *tuple(module for i in range(0, 8) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
            ), (
                *tuple(module for i in range(8, 18) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
            ), (
                *tuple(module for i in range(18, 28) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
                transformer.final_layer,
            )]
        elif n == 4:
            return [(
                *tuple(module for i in range(0, 2) for module in (
                    transformer.text_fusion.layerwise_blocks[i],
                    transformer.text_fusion.refiner_blocks[i],
                )),
                transformer.text_fusion.projector,
                transformer.txt_in,
                *tuple(module for i in range(0, 6) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
            ), (
                *tuple(module for i in range(6, 14) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
            ), (
                *tuple(module for i in range(14, 22) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
            ), (
                *tuple(module for i in range(22, 28) for module in (
                    transformer.transformer_blocks[i].attn,
                    transformer.transformer_blocks[i].ff,
                )),
                transformer.final_layer,
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "zimage":
        """
            ZImageTransformer2DModel(
              (all_x_embedder): ModuleDict(
                (2-1): Linear()
              )
              (all_final_layer): ModuleDict(
                (2-1): FinalLayer(
                  (norm_final): LayerNorm()
                  (linear): Linear()
                  (adaLN_modulation): Sequential(
                    (0): SiLU()
                    (1): Linear()
                  )
                )
              )
              (noise_refiner): ModuleList(
                (0-1): 2 x ZImageTransformerBlock(
                  (attention): Attention(
                    (norm_q): RMSNorm()
                    (norm_k): RMSNorm()
                    (to_q): Linear()
                    (to_k): Linear()
                    (to_v): Linear()
                    (to_out): ModuleList(
                      (0): Linear()
                      (1): Dropout()
                    )
                  )
                  (feed_forward): FeedForward(
                    (w1): Linear()
                    (w2): Linear()
                    (w3): Linear()
                  )
                  (attention_norm1): RMSNorm()
                  (ffn_norm1): RMSNorm()
                  (attention_norm2): RMSNorm()
                  (ffn_norm2): RMSNorm()
                  (adaLN_modulation): Sequential(
                    (0): Linear()
                  )
                )
              )
              (context_refiner): ModuleList(
                (0-1): 2 x ZImageTransformerBlock(
                  (attention): Attention(
                    (norm_q): RMSNorm()
                    (norm_k): RMSNorm()
                    (to_q): Linear()
                    (to_k): Linear()
                    (to_v): Linear()
                    (to_out): ModuleList(
                      (0): Linear()
                      (1): Dropout()
                    )
                  )
                  (feed_forward): FeedForward(
                    (w1): Linear()
                    (w2): Linear()
                    (w3): Linear()
                  )
                  (attention_norm1): RMSNorm()
                  (ffn_norm1): RMSNorm()
                  (attention_norm2): RMSNorm()
                  (ffn_norm2): RMSNorm()
                )
              )
              (t_embedder): TimestepEmbedder(
                (mlp): Sequential(
                  (0): Linear()
                  (1): SiLU()
                  (2): Linear()
                )
              )
              (cap_embedder): Sequential(
                (0): RMSNorm()
                (1): Linear()
              )
              (layers): ModuleList(
                (0-29): 30 x ZImageTransformerBlock(
                  (attention): Attention(
                    (norm_q): RMSNorm()
                    (norm_k): RMSNorm()
                    (to_q): Linear()
                    (to_k): Linear()
                    (to_v): Linear()
                    (to_out): ModuleList(
                      (0): Linear()
                      (1): Dropout()
                    )
                  )
                  (feed_forward): FeedForward(
                    (w1): Linear()
                    (w2): Linear()
                    (w3): Linear()
                  )
                  (attention_norm1): RMSNorm()
                  (ffn_norm1): RMSNorm()
                  (attention_norm2): RMSNorm()
                  (ffn_norm2): RMSNorm()
                  (adaLN_modulation): Sequential(
                    (0): Linear()
                  )
                )
              )
            )
        """
        if n == 1:
            return [(
                *tuple(module for i in range(0, 2) for module in (
                    transformer.noise_refiner[i].attention,
                    transformer.noise_refiner[i].feed_forward,
                    transformer.context_refiner[i],
                )),
                *tuple(module for i in range(0, 30) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
                transformer.all_final_layer,
            )]
        elif n == 2:
            return [(
                *tuple(module for i in range(0, 2) for module in (
                    transformer.noise_refiner[i].attention,
                    transformer.noise_refiner[i].feed_forward,
                    transformer.context_refiner[i],
                )),
                *tuple(module for i in range(0, 14) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
            ), (
                *tuple(module for i in range(14, 30) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
                transformer.all_final_layer,
            )]
        elif n == 3:
            return [(
                *tuple(module for i in range(0, 2) for module in (
                    transformer.noise_refiner[i].attention,
                    transformer.noise_refiner[i].feed_forward,
                    transformer.context_refiner[i],
                )),
                *tuple(module for i in range(0, 9) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
            ), (
                *tuple(module for i in range(9, 20) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
            ), (
                *tuple(module for i in range(20, 30) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
                transformer.all_final_layer,
            )]
        elif n == 4:
            return [(
                *tuple(module for i in range(0, 2) for module in (
                    transformer.noise_refiner[i].attention,
                    transformer.noise_refiner[i].feed_forward,
                    transformer.context_refiner[i],
                )),
                *tuple(module for i in range(0, 6) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
            ), (
                *tuple(module for i in range(6, 14) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
            ), (
                *tuple(module for i in range(14, 22) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
            ), (
                *tuple(module for i in range(22, 30) for module in (
                    transformer.layers[i].attention,
                    transformer.layers[i].feed_forward,
                )),
                transformer.all_final_layer,
            )]
        else:
            raise NotImplementedError
    elif pipe_id in ["wani2v", "want2v"]: # TODO: might have to differentiate between 5b/14b, wan2.1/wan2.2
        """
            WanTransformer3DModel(
              (rope): WanRotaryPosEmbed()
              (patch_embedding): Conv3d(3)
              (condition_embedder): WanTimeTextImageEmbedding(
                (timesteps_proj): Timesteps()
                (time_embedder): TimestepEmbedding(
                  (linear_1): Linear()
                  (act): SiLU()
                  (linear_2): Linear()
                )
                (act_fn): SiLU()
                (time_proj): Linear()
                (text_embedder): PixArtAlphaTextProjection(
                  (linear_1): Linear()
                  (act_1): GELU()
                  (linear_2): Linear()
                )
              (image_embedder): WanImageEmbedding(
                  (norm1): FP32LayerNorm()
                  (ff): FeedForward(
                    (net): ModuleList(
                      (0): GELU(
                        (proj): Linear()
                      )
                      (1): Dropout()
                      (2): Linear()
                    )
                  )
                  (norm2): FP32LayerNorm()
                )
              )
              (blocks): ModuleList(
                (0-39): 40 x WanTransformerBlock(
                  (norm1): FP32LayerNorm()
                  (attn1): WanAttention(
                    (to_q): Lanear()
                    (to_k): Linear()
                    (to_v): Lanear()
                    (to_out): ModuleList(
                      (0): Linear()
                      (1): Dropout()
                    )
                    (norm_q): RMSNorm()
                    (norm_k): RMSNorm()
                  )
                  (attn2): WanAttention(
                    (to_q): Linear()
                    (to_k): Linear()
                    (to_v): Linear()
                    (to_out): ModuleList(
                      (0): Linear()
                      (1): Dropout()
                    )
                    (norm_q): RMSNorm()
                    (norm_k): RMSNorm()
                    (add_k_proj): Linear()
                    (add_v_proj): Linear()
                    (norm_added_k): RMSNorm()
                  )
                  (norm2): FP32LayerNorm()
                  (ffn): FeedForward(
                    (net): ModuleList(
                      (0): GELU(
                        (proj): Linear()
                      )
                      (1): Dropout()
                      (2): Linear()
                    )
                  )
                  (norm3): FP32LayerNorm()
                )

              (norm_out): FP32LayerNorm()
              (proj_out): Linear()
            )
        """
        if n == 1:
            return [(
                *tuple(module for i in range(0, 40) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            )]
        elif n == 2:
            return [(
                *tuple(module for i in range(0, 21) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            ), (
                *tuple(module for i in range(21, 40) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            )]
        elif n == 3:
            return [(
                *tuple(module for i in range(0, 14) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            ), (
                *tuple(module for i in range(14, 28) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            ), (
                *tuple(module for i in range(28, 40) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            )]
        elif n == 4:
            return [(
                *tuple(module for i in range(0, 10) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            ), (
                *tuple(module for i in range(10, 20) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            ), (
                *tuple(module for i in range(20, 30) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            ), (
                *tuple(module for i in range(30, 40) for module in (
                    transformer.blocks[i].attn1,
                    transformer.blocks[i].attn2,
                )),
            )]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

