import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple
from .block_denoise import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder


class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips, temb):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        ## save the features with the character of combination.
        # feature_1 = skips[-1].detach().cpu().numpy()
        # feature_2 = skips[-2].detach().cpu().numpy()
        # feature_3 = skips[-3].detach().cpu().numpy()
        # feature_4 = skips[-4].detach().cpu().numpy()
        

        # print(feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape, feature_5.shape)

        # np.save("./f1_only_d", feature_1)
        # np.save("./f2_only_d", feature_2)
        # np.save("./f3_only_d", feature_3)
        # np.save("./f4_only_d", feature_4)

        # feature_5 = skips[-5].detach().cpu().numpy()
        # rand_id = np.random.randint(0, 1000)
        # save_dir = f"./tsne_plot/denoise/feature_5/"
        # import os 
        # os.makedirs(save_dir, exist_ok=True)
        # np.save(os.path.join(save_dir, f"{rand_id}"), feature_5)
        # exit(0)

        lres_input_plt = lres_input

        # import matplotlib.pyplot as plt 
        # plt_x = skips[-1].detach().cpu().numpy()
        # print(plt_x.shape)
        # plt.subplot(1, 5, 1)
        # plt.imshow(plt_x[0, :, 2].sum(axis=0), cmap="gray")

        # plt_x2 = skips[-2].detach().cpu().numpy()
        # print(plt_x2.shape)
        # plt.subplot(1, 5, 2)
        # plt.imshow(plt_x2[0, :, 4].sum(axis=0), cmap="gray")

        # plt_x3 = skips[-3].detach().cpu().numpy()
        # print(plt_x3.shape)
        # plt.subplot(1, 5, 3)
        # plt.imshow(plt_x3[0, :, 8].sum(axis=0), cmap="gray")

        # plt_x4 = skips[-4].detach().cpu().numpy()
        # print(plt_x4.shape)
        # plt.subplot(1, 5, 4)
        # plt.imshow(plt_x4[0, :, 16].sum(axis=0), cmap="gray")

        # plt.savefig("./test15.png")

        # plt.colorbar()

        # plt_output = self.seg_layers[-1](x).argmax(dim=1).detach().cpu().numpy()
        # plt.subplot(1, 2, 2)
        # plt.imshow(plt_output[0, 8], cmap="gray")

        # exit(0)
        
        outputs = []
        for s in range(len(self.stages)):
            
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s]([x, temb])

            outputs.append(x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                
                # feature_5 = outputs[4].detach().cpu().numpy()
                # rand_id = np.random.randint(0, 1000)
                # save_dir = f"./tsne_plot/denoise/feature_5_decoder/"
                # import os 
                # os.makedirs(save_dir, exist_ok=True)
                # np.save(os.path.join(save_dir, f"{rand_id}"), feature_5)

                # feature_1 = outputs[0].detach().cpu().numpy()
                # feature_2 = outputs[1].detach().cpu().numpy()
                # feature_3 = outputs[2].detach().cpu().numpy()
                # feature_4 = outputs[3].detach().cpu().numpy()
                # feature_5 = outputs[4].detach().cpu().numpy()

                # print(feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape, feature_5.shape)

                # np.save("./f1_only_d_decoder", feature_1)
                # np.save("./f2_only_d_decoder", feature_2)
                # np.save("./f3_only_d_decoder", feature_3)
                # np.save("./f4_only_d_decoder", feature_4)
                # np.save("./f5_only_d_decoder", feature_5)
                # exit(0)
        
                # ## add visualization
                # import matplotlib.pyplot as plt 
                # plt_x = outputs[0].detach().cpu().numpy()
                # print(plt_x.shape)
                # plt.subplot(1, 6, 1)
                # plt.xticks([])
                # plt.yticks([])
                # plt.imshow(plt_x[0, :, 4].sum(axis=0), cmap="jet")

                # plt_x2 = outputs[1].detach().cpu().numpy()
                # print(plt_x2.shape)
                # plt.subplot(1, 6, 2)
                # plt.xticks([])
                # plt.yticks([])
                # plt.imshow(plt_x2[0, :, 8].sum(axis=0), cmap="jet")

                # plt_x3 = outputs[2].detach().cpu().numpy()
                # print(plt_x3.shape)
                # plt.subplot(1, 6, 3)
                # plt.xticks([])
                # plt.yticks([])
                # # plt.colorbar()
                # plt.imshow(plt_x3[0, :, 16].sum(axis=0), cmap="jet")

                # plt_x4 = outputs[3].detach().cpu().numpy()
                # print(plt_x4.shape)
                # plt.subplot(1, 6, 4)
                # plt.xticks([])
                # plt.yticks([])

                # plt.imshow(plt_x4[0, :, 32].sum(axis=0), cmap="jet")

                # plt_x5 = outputs[4].detach().cpu().numpy()
                # print(plt_x5.shape)
                # plt.subplot(1, 6, 5)
                # plt.imshow(plt_x5[0, :, 64].sum(axis=0), cmap="jet")

                # plt.xticks([])
                # plt.yticks([])

                # plt_x6 = outputs[4].detach().cpu().numpy()
                # print(plt_x6.shape)
                # plt.subplot(1, 6, 6)
                # plt.imshow(plt_x6[0, :, 64].sum(axis=0), cmap="gray")

                # plt.xticks([])
                # plt.yticks([])

                # plt.savefig("./test15.png")
                # exit(0)

                # plt_x3 = outputs[2].detach().cpu().numpy()
                # print(plt_x3.shape)
                # # plt.subplot(1, 6, 3)
                # plt.xticks([])
                # plt.yticks([])
                # plt.colorbar()
                # plt.imshow(plt_x3[0, :, 16].sum(axis=0), cmap="jet")
                # plt.savefig("./test14.png")
                # exit(0)
                # import matplotlib.pyplot as plt 
                # plt_x = x.detach().cpu().numpy()
                # print(plt_x.shape)
                # # exit(0)
                # plt.subplot(1, 2, 1)
                # plt.imshow(plt_x[0, 16, 60], cmap="jet")
                # plt.colorbar()

                # plt_output = self.seg_layers[-1](x).argmax(dim=1).detach().cpu().numpy()
                # plt.subplot(1, 2, 2)
                # plt.imshow(plt_output[0, 60], cmap="gray")

                # plt.savefig("./test11.png")
                # exit(0)

                seg_outputs.append(self.seg_layers[-1](x))
                # seg_outputs.append(x)
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output