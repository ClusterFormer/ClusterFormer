import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers.helpers import to_2tuple
import torch.nn.functional as F


# try:
#     from mmseg.models.builder import BACKBONES as seg_BACKBONES
#     from mmseg.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmseg = True
# except ImportError:
#     print("If for semantic segmentation, please install mmsegmentation first")
#     has_mmseg = False

# try:
#     from mmdet.models.builder import BACKBONES as det_BACKBONES
#     from mmdet.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmdet = True
# except ImportError:
#     print("If for detection, please install mmdetection first")
#     has_mmdet = False

try:
    from mmcls.models.builder import BACKBONES as cls_BACKBONES
    from mmcls.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmcls = True
except ImportError:
    print("If for cls, please install mmcls first")
    has_mmcls = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
}


class PointReducer(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
    x1 = F.normalize(x1,dim=-1)
    x2 = F.normalize(x2,dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Clustering(nn.Module):
    def __init__(self, dim, out_dim, center_w=2, center_h=2, window_w=2, window_h=2, heads=4, head_dim=24, return_center=False, num_clustering=1):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.conv1 = nn.Conv2d(dim, heads*head_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(heads*head_dim, out_dim, kernel_size=1)
        self.conv_c = nn.Conv2d(head_dim, head_dim, kernel_size=1)
        self.conv_v = nn.Conv2d(dim, heads*head_dim, kernel_size=1)
        self.conv_f = nn.Conv2d(dim, heads*head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((center_w,center_h))
        self.window_w = int(window_w)
        self.window_h = int(window_h)
        self.return_center = return_center
        self.softmax = nn.Softmax(dim=-2)
        self.num_clustering = num_clustering

    def forward(self, x): #[b,c,w,h]
        value = self.conv_v(x) 
        feature = self.conv_f(x)
        x = self.conv1(x)
 

        # multi-head
        b, c, w, h = x.shape
        x = x.reshape(b*self.heads, int(c/self.heads), w, h)
        value = value.reshape(b*self.heads, int(c/self.heads), w, h)
        feature = feature.reshape(b*self.heads, int(c/self.heads), w, h)

        # window token
        if self.window_w>1 and self.window_h>1:
            b, c, w, h = x.shape
            x = x.reshape(b*self.window_w*self.window_h, c, int(w/self.window_w), int(h/self.window_h))
            value = value.reshape(b*self.window_w*self.window_h, c, int(w/self.window_w), int(h/self.window_h))
            feature = feature.reshape(b*self.window_w*self.window_h, c, int(w/self.window_w), int(h/self.window_h))

        b, c, w, h = x.shape
        value = value.reshape(b, w*h, c)

        # centers
        centers = self.centers_proposal(x)
        b, c, c_w, c_h = centers.shape
        centers_feature = self.centers_proposal(feature).reshape(b, c_w*c_h, c)
        
        feature = feature.reshape(b, w*h, c)

        for _ in range(self.num_clustering):    # iterative clustering and updating centers
            centers = self.conv_c(centers).reshape(b, c_w*c_h, c)
            similarity = self.softmax((centers @ value.transpose(-2, -1)))
            centers = (similarity @ feature).reshape(b, c, c_w, c_h)

        # similarity
        similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(centers.reshape(b,c,-1).permute(0,2,1), x.reshape(b,c,-1).permute(0,2,1)))
        
        # assign each point to one center
        _, max_idx = similarity.max(dim=1, keepdim=True)
        mask = torch.zeros_like(similarity)
        mask.scatter_(1, max_idx, 1.)
        similarity= similarity*mask

        out = ( ( feature.unsqueeze(dim=1)*similarity.unsqueeze(dim=-1) ).sum(dim=2) + centers_feature)/ (mask.sum(dim=-1,keepdim=True)+ 1.0) 

        if self.return_center:
            out = out.reshape(b, c, c_w, c_h)
            return out
        else:
            out = (out.unsqueeze(dim=2)*similarity.unsqueeze(dim=-1)).sum(dim=1)
            out = out.reshape(b, c, w, h)

        # recover feature maps
        if self.window_w>1 and self.window_h>1:
            out = out.reshape(int(out.shape[0]/self.window_w/self.window_h), out.shape[1], out.shape[2]*self.window_w, out.shape[3]*self.window_h)
        
        out = out.reshape(int(out.shape[0]/self.heads), out.shape[1]*self.heads, out.shape[2], out.shape[3])
        out = self.conv2(out)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ClusterBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 center_w=2, center_h=2, window_w=2, window_h=2, heads=4, head_dim=24, return_center=False,num_clustering=1):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Clustering(dim=dim, out_dim=dim, center_w=center_w, center_h=center_h, window_w=window_w, window_h=window_h, heads=heads, head_dim=head_dim, return_center=return_center,num_clustering=num_clustering)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        self.return_center = return_center
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):

        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio = 4.,
                 act_layer = nn.GELU, norm_layer = GroupNorm,
                 drop_rate = .0, drop_path_rate = 0.,
                 use_layer_scale=True, layer_scale_init_value = 1e-5,
                 center_w = 5, center_h = 5, window_w = 5, window_h = 5, heads = 4, head_dim = 24, return_center = False, num_clustering=1):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ClusterBlock(
            dim, mlp_ratio = mlp_ratio,
            act_layer = act_layer, norm_layer = norm_layer,
            drop = drop_rate, drop_path = block_dpr,
            use_layer_scale = use_layer_scale,
            layer_scale_init_value = layer_scale_init_value,
            center_w = center_w, center_h = center_h, window_w = window_w, window_h = window_h,
            heads=heads, head_dim=head_dim, return_center=return_center, num_clustering= num_clustering
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class Cluster(nn.Module):
    def __init__(self, layers, embed_dims = None,
                 mlp_ratios = None, downsamples = None,
                 norm_layer = nn.BatchNorm2d, act_layer = nn.GELU,
                 num_classes = 1000,
                 in_patch_size = 4, in_stride = 4, in_pad = 0,
                 down_patch_size = 2, down_stride = 2, down_pad = 0,
                 drop_rate = 0., drop_path_rate = 0.,
                 use_layer_scale = True, layer_scale_init_value = 1e-5,
                 fork_feat = False,
                 init_cfg = None,
                 pretrained = None,
                 img_w = 224, img_h = 224,
                 center_w = [2,2,2,2], center_h = [2,2,2,2], window_w = [32, 16, 8, 4], window_h = [32, 16, 8, 4],
                 heads = [2,4,6,8], head_dim = [16,16,32,32], return_center = False, num_clustering= 1,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat



        self.patch_embed = PointReducer(
            patch_size = in_patch_size, stride = in_stride, padding = in_pad,
            in_chans = 5, embed_dim = embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio = mlp_ratios[i],
                                 act_layer = act_layer, norm_layer = norm_layer,
                                 drop_rate = drop_rate,
                                 drop_path_rate = drop_path_rate,
                                 use_layer_scale = use_layer_scale,
                                 layer_scale_init_value = layer_scale_init_value,
                                 center_w = center_w[i], center_h = center_h[i],
                                 window_w = window_w[i], window_h = window_h[i], heads = heads[i], head_dim = head_dim[i],
                                 return_center = return_center, num_clustering= num_clustering
                                 )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PointReducer(
                        patch_size = down_patch_size, stride = down_stride,
                        padding = down_pad,
                        in_chans = embed_dims[i], embed_dim = embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained = None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger = logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        _,c,img_w,img_h = x.shape
        # print(f"det img size is {img_w} * {img_h}")
        # register positional information buffer.
        range_w = torch.arange(0, img_w, step=1)/(img_w-1.0)
        range_h = torch.arange(0, img_h, step=1)/(img_h-1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing = 'ij'), dim = -1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos-0.5
        pos = fea_pos.permute(2,0,1).unsqueeze(dim=0).expand(x.shape[0],-1,-1,-1)
        x = self.patch_embed(torch.cat([x,pos], dim=1))
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out

if has_mmcls:
    # @seg_BACKBONES.register_module()
    # @det_BACKBONES.register_module()
    @cls_BACKBONES.register_module()
    class cluster_tiny(Cluster):
        def __init__(self, **kwargs):
            layers = [2, 2, 6, 2]
            norm_layer=GroupNorm
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [8, 8, 4, 4]
            downsamples = [True, True, True, True]
            center_w=[10, 10, 10, 10]
            center_h=[10, 10, 10, 10]
            window_w=[32, 16, 8, 4]
            window_h=[32, 16, 8, 4]
            heads=[3,6,12,24]
            head_dim=[32,32,32,32]
            down_patch_size=3
            down_pad = 1
            super().__init__(
                layers, embed_dims=embed_dims, norm_layer=norm_layer,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                down_patch_size = down_patch_size, down_pad=down_pad,
                center_w=center_w, center_h=center_h, window_w=window_w, window_h=window_h,
                heads=heads, head_dim=head_dim,
                fork_feat=True, return_center = False, num_clustering = 3,
                **kwargs)

    # @seg_BACKBONES.register_module()
    # @det_BACKBONES.register_module()
    @cls_BACKBONES.register_module()
    class cluster_small(Cluster):
        def __init__(self, **kwargs):
            layers = [2, 2, 18, 2]
            norm_layer=GroupNorm
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [8, 8, 4, 4]
            downsamples = [True, True, True, True]
            center_w=[10, 10, 10, 10]
            center_h=[10, 10, 10, 10]
            window_w=[32, 16, 8, 4]
            window_h=[32, 16, 8, 4]
            heads=[3,6,12,24]
            head_dim=[32,32,32,32]
            down_patch_size=3
            down_pad = 1
            super().__init__(
                layers, embed_dims=embed_dims, norm_layer=norm_layer,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                down_patch_size = down_patch_size, down_pad=down_pad,
                center_w=center_w, center_h=center_h, window_w=window_w, window_h=window_h,
                heads=heads, head_dim=head_dim,
                fork_feat=True,
                **kwargs)