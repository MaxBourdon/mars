import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair
import t3nsor as t3


###################################################################################################


def masked_dense_tt_matmul(matrix_a, tt_matrix_b, masks=None):
    "Perform masked matmul of a dense matrix and a TT-matrix."
    # Partially borrowed from https://github.com/KhrulkovV/tt-pytorch
    ndims = tt_matrix_b.ndims
    a_columns = matrix_a.shape[1]
    b_rows = tt_matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError(f'Arguments shapes should align got {matrix_a.shape} and {tt_matrix_b.shape} instead.')

    a_shape = matrix_a.shape
    b_shape = tt_matrix_b.shape
    b_raw_shape = tt_matrix_b.raw_shape
    data = matrix_a

    new_shape = [-1, ] + b_raw_shape[0] + [1, ]
    data = data.view(*new_shape)

    for core_idx in range(ndims):
        curr_core = tt_matrix_b.tt_cores[core_idx]
        data = torch.tensordot(data, curr_core, dims=[[1, -1], [1, 0]])
        if masks is not None and core_idx < ndims - 1:
            data *= masks[core_idx]

    return data.view(a_shape[0], b_shape[1])

def masked_full(tt_weight, masks=None):
    "Perform masked constructing of the full tensor from TT-cores."
    # Partially borrowed from https://github.com/KhrulkovV/tt-pytorch
    # Note: in current framework, constructing the full tensor and performing the "usual"
    # tensorized operation may work faster than performing the same operation via TT-cores.
    num_dims = tt_weight.ndims
    ranks = tt_weight.ranks
    shape = tt_weight.shape
    raw_shape = tt_weight.raw_shape
    res = tt_weight.tt_cores[0]

    for i in range(1, num_dims):
        res = res.view(-1, ranks[i])
        if masks is not None:
            res = res * masks[i - 1]
        curr_core = tt_weight.tt_cores[i].view(ranks[i], -1)
        res = torch.matmul(res, curr_core)

    if tt_weight.is_tt_matrix:
        intermediate_shape = []
        for i in range(num_dims):
            intermediate_shape.append(raw_shape[0][i])
            intermediate_shape.append(raw_shape[1][i])

        res = res.view(*intermediate_shape)
        transpose = []
        for i in range(0, 2 * num_dims, 2):
            transpose.append(i)
        for i in range(1, 2 * num_dims, 2):
            transpose.append(i)
        res = res.permute(*transpose)

    if tt_weight.is_tt_matrix:
        res = res.contiguous().view(*shape)
    else:
        res = res.view(*shape)
        
    return res


###################################################################################################


class TensorizedModel(nn.Module):
    def __init__(self):
        "A tensorized model base class."
        super().__init__()
        self._ranks = None
        self._cores = None
        self._total = None
        
    def __str__(self): 
        return "Abstract tensorized model"

    @property
    def cores(self):
        return self._cores

    @property
    def ranks(self):
        return self._ranks

    @property
    def total(self):
        return self._total

    def calc_dof(self, ranks=None):
        "Calculate degrees of freedom given ranks, i.e., the number of actually occupied parameters."
        raise NotImplementedError()


class FactorizedLinear(TensorizedModel):
    def __init__(self, in_features, out_features, rank=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._ranks = _single(rank)
        
        self.input_weight = nn.Parameter(torch.Tensor(self.ranks[0], self.in_features))
        self.output_weight = nn.Parameter(torch.Tensor(self.out_features, self.ranks[0]))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.reset_parameters()

        self._cores = [self.input_weight, self.output_weight]
        self._total = self.in_features * self.out_features
        
    def __str__(self): 
        str_from = str([self.in_features, self.out_features]) 
        str_to1 = str([self.in_features, self.ranks[0]])
        str_to2 = str([self.ranks[0], self.out_features]) 
        return "Factorized Linear: " + str_from + ' -> ' + str_to1 + '-' + str_to2

    def calc_dof(self, ranks=None):
        ranks = ranks or self.ranks
        return (self.in_features + self.out_features) * ranks[0]

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.input_weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.output_weight, a=np.sqrt(5))
        bound = 1 / np.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, masks=None):
        input_weight, output_weight = self.input_weight, self.output_weight

        if masks is not None and not self.training:
            input_weight = input_weight[masks[0]]
            output_weight = output_weight[:, masks[0]]

        x = F.linear(x, input_weight)
        if masks is not None and self.training:
            x *= masks[0]
        return F.linear(x, output_weight, self.bias)
        

class TuckerConv2d(TensorizedModel):
    def __init__(self, in_channels, out_channels, kernel_size, rank=32, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self._ranks = _pair(rank)
        self.stride = stride
        self.padding = padding
        
        self.first_weight = nn.Parameter(torch.Tensor(self.ranks[0], self.in_channels, 1, 1))
        self.core_weight = nn.Parameter(torch.Tensor(self.ranks[1], self.ranks[0], *self.kernel_size))
        self.last_weight = nn.Parameter(torch.Tensor(self.out_channels, self.ranks[1], 1, 1))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        self.reset_parameters()

        self._cores = [self.first_weight, self.core_weight, self.last_weight]
        self._total = self.in_channels * self.out_channels * np.prod(self.kernel_size)
        
    def __str__(self): 
        str_from = str([self.in_channels, self.out_channels, *self.kernel_size]) 
        str_to1 = str([self.in_channels, self.ranks[0]]) 
        str_to2 = str([self.ranks[0], self.ranks[1], *self.kernel_size]) 
        str_to3 = str([self.ranks[1], self.out_channels]) 
        return "Tucker Conv2d: " + str_from + ' -> ' + str_to1 + '-' + str_to2 + '-' + str_to3

    def calc_dof(self, ranks=None):
        ranks = ranks or self.ranks
        return self.in_channels * ranks[0] + np.prod(ranks) * np.prod(self.kernel_size) + self.out_channels * ranks[1]

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.first_weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.core_weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.last_weight, a=np.sqrt(5))
        bound = 1 / np.sqrt(self.in_channels * np.prod(self.kernel_size))
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, masks=None):
        first_weight, core_weight, last_weight = self.first_weight, self.core_weight, self.last_weight

        if masks is not None and not self.training:
            first_weight = first_weight[masks[0]]
            core_weight = core_weight[masks[1]][:, masks[0]]
            last_weight = last_weight[:, masks[1]]

        # A pointwise convolution that reduces the channels from C_{in} to r_1
        x = F.conv2d(x, first_weight)
        if masks is not None and self.training:
            x *= masks[0].view(-1, 1, 1)
        # A regular 2D convolution layer with r_1 input channels and r_2 output channels
        x = F.conv2d(x, core_weight, stride=self.stride, padding=self.padding)
        if masks is not None and self.training:
            x *= masks[1].view(-1, 1, 1)
        # A pointwise convolution that increases the channels from r_2 to C_{out}
        return F.conv2d(x, last_weight, self.bias)


class TTLinear(TensorizedModel):
    # Partially borrowed from https://github.com/KhrulkovV/tt-pytorch
    def __init__(self, in_features=None, out_features=None, bias=True, 
                 init=None, shape=None, auto_shapes=True, d=3, tt_rank=8, 
                 auto_shape_mode='ascending', auto_shape_criterion='entropy'):
        super(TTLinear, self).__init__()

        if auto_shapes:
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                    "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.glorot_initializer(shape, tt_rank=tt_rank)

        self.shape = shape
        self.tt_weight = init.to_parameter()
        self.parameters = self.tt_weight.parameter
        self.mm_op = masked_dense_tt_matmul

        if bias:
            self.bias = torch.nn.Parameter(1e-3 * torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

        self._cores = self.tt_weight.tt_cores
        self._ranks = self.tt_weight.ranks[1:-1]
        self._total = self.tt_weight.total

    def __str__(self): 
        s = str(self.tt_weight)
        s = s[:1].lower() + s[1:]
        return "TT Linear with " + s

    def calc_dof(self, ranks=None):
        if ranks is None:
            return self.tt_weight.dof
        ranks = [1] + ranks + [1]
        return sum([ranks[r] * self.shape[0][r] * self.shape[1][r] * ranks[r + 1] for r in range(len(self.cores))])

    def forward(self, x, masks=None, tt_weight=None):
        if not self.training and masks is not None:
            masks = [[True]] + masks + [[True]]
            cores = [core[masks[r]][..., masks[r + 1]] for r, core in enumerate(self.cores)]
            tt_weight = t3.TensorTrain(cores, convert_to_tensors=False)
            masks = None
        else:
            tt_weight = tt_weight or self.tt_weight

        res = self.mm_op(x, tt_weight, masks)
        if self.bias is not None:
            res += self.bias
        
        return res


class TTEmbedding(TensorizedModel):
    # Partially borrowed from https://github.com/KhrulkovV/tt-pytorch
    def __init__(self, voc_size, emb_size,
                 init=None, shape=None, 
                 auto_shapes=None,
                 auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 d=3, tt_rank=8,
                 batch_dim_last=None,
                 padding_idx=None):

        super(TTEmbedding, self).__init__()

        self.voc_size = int(voc_size)
        self.emb_size = int(emb_size)

        if auto_shapes:
            voc_quantization = t3.utils.suggest_shape(
                self.voc_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            emb_quantization = t3.utils.auto_shape(
                self.emb_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [voc_quantization, emb_quantization]
            self.shape = shape

        else:
            self.shape = shape

        if init is None:
            if shape is None:
                raise ValueError('if init is not provided, please specify shape')
        else:
            self.shape = init.raw_shape

        if init is None:
            init = t3.glorot_initializer(self.shape, tt_rank=tt_rank)

        self.tt_weight = init.to_parameter()
        self.parameters = self.tt_weight.parameter

        self.batch_dim_last = batch_dim_last
        self.padding_idx = padding_idx

        self.voc_quant = self.shape[0]
        self.emb_quant = self.shape[1]

        self._cores = self.tt_weight.tt_cores
        self._ranks = self.tt_weight.ranks[1:-1]
        self._total = self.voc_size * self.emb_size  # we need to compare against initial size

    def __str__(self): 
        s = str(self.tt_weight)
        s = s[:1].lower() + s[1:]
        return "TT Embedding with " + s

    def calc_dof(self, ranks=None):
        if ranks is None:
            return self.tt_weight.dof
        ranks = [1] + ranks + [1]
        return sum([ranks[r] * self.shape[0][r] * self.shape[1][r] * ranks[r + 1] for r in range(len(self.cores))])

    def forward(self, x, masks=None, tt_weight=None):
        if not self.training and masks is not None:
            masks = [[True]] + masks + [[True]]
            cores = [core[masks[r]][..., masks[r + 1]] for r, core in enumerate(self.cores)]
            tt_weight = t3.TensorTrain(cores, convert_to_tensors=False)
            masks = None
        else:
            tt_weight = tt_weight or self.tt_weight

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        full = masked_full(tt_weight, masks)
        rows = full[x]

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        return rows


class TTConv2d(TensorizedModel):
    # Partially borrowed from https://github.com/KhrulkovV/tt-pytorch
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=1, padding=0, dilation=1, bias=True, init=None, 
                 shape=None, auto_shapes=True, d=3, tt_rank=8, 
                 auto_shape_mode='ascending', auto_shape_criterion='entropy'):
        super(TTConv2d, self).__init__()

        if auto_shapes:
            if in_channels is None or out_channels is None or kernel_size is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                kernel_size ** 2 * in_channels, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_channels, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [out_quantization, in_quantization]
        else:
            if init is None:
                if shape is None:
                    raise ValueError(
                        "If init is not provided, please specify shape, or set auto_shapes=True")
            else:
                shape = init.raw_shape

            if in_channels is None and kernel_size is None:
                raise ValueError("At least one of 'in_channels' and 'kernel_size' parameters must be specified")
            if in_channels is None:
                in_channels = np.prod(shape[1]) // kernel_size ** 2
            if kernel_size is None:
                kernel_size = int(np.sqrt(np.prod(shape[1]) / in_channels))
            out_channels = np.prod(shape[0])

        if init is None:
            init = t3.glorot_initializer(shape, tt_rank=tt_rank)

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.shape = shape
        self.tt_weight = init.to_parameter()
        self.parameters = self.tt_weight.parameter

        if bias:
            self.bias = torch.nn.Parameter(1e-2 * torch.ones(out_channels))
        else:
            self.register_parameter('bias', None)

        self._cores = self.tt_weight.tt_cores
        self._ranks = self.tt_weight.ranks[1:-1]
        self._total = self.tt_weight.total

    def __str__(self): 
        s = str(self.tt_weight)
        s = s[:1].lower() + s[1:]
        return "TT Conv2d with " + s

    def calc_dof(self, ranks=None):
        if ranks is None:
            return self.tt_weight.dof
        ranks = [1] + ranks + [1]
        return sum([ranks[r] * self.shape[0][r] * self.shape[1][r] * ranks[r + 1] for r in range(len(self.cores))])

    def forward(self, x, masks=None, tt_weight=None):
        if not self.training and masks is not None:
            masks = [[True]] + masks + [[True]]
            cores = [core[masks[r]][..., masks[r + 1]] for r, core in enumerate(self.cores)]
            tt_weight = t3.TensorTrain(cores, convert_to_tensors=False)
            masks = None
        else:
            tt_weight = tt_weight or self.tt_weight

        inp_ch = x.shape[1]
        full = masked_full(tt_weight, masks).view(-1, inp_ch, self.kernel_size, self.kernel_size)
        return  F.conv2d(x, full, self.bias, self.stride, self.padding, self.dilation)
