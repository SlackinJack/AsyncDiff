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
    elif pipe_id == "flux":
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
        if n == 1:
            return [(
                *transformer.layers[0:30],
                # *transformer.noise_refiner[0:2],
                # *transformer.context_refiner[0:2],
                transformer.all_final_layer,
            )]
        elif n == 2:
            return [(
                *transformer.layers[0:15],
            ), (
                *transformer.layers[15:30],
                # *transformer.noise_refiner[0:2],
                # *transformer.context_refiner[0:2],
                transformer.all_final_layer,
            )]
        elif n == 3:
            return [(
                *transformer.layers[0:10],
            ), (
                *transformer.layers[10:20],
            ), (
                *transformer.layers[20:30],
                # *transformer.noise_refiner[0:2],
                # *transformer.context_refiner[0:2],
                transformer.all_final_layer,
            )]
        elif n == 4:
            return [(
                *transformer.layers[0:8],
            ), (
                *transformer.layers[8:16],
            ), (
                *transformer.layers[16:24],
            ), (
                *transformer.layers[24:30],
                # *transformer.noise_refiner[0:2],
                # *transformer.context_refiner[0:2],
                transformer.all_final_layer,
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "wani2v" or pipe_id == "want2v":
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

