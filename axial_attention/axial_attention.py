import torch
from torch import nn
from operator import itemgetter
from axial_attention.reversible import ReversibleSequence

# helper functions

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    """
    Purpose: create list of all possible tensor permutations that go [0, axial dimensions (in different orders), embedding dimension]
    num_dimensions = number of axial dimensions (images is 2, video is 3, or more)
    emb_dim = where is the embedding dimension (pass dim_index to it)
    """
    total_dimensions = num_dimensions + 2 # 1 for channel, 1 for padding in range fxns below
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions) # in case emb_dim is -1 or -2 etc.
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim] # list of dimensions that aren't the embedding

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims) # calcs all dims that arent in last_two_dims
        permutation = [*dims_rest, *last_two_dims] #unpacks and orders last_two_dims and dims_rest
        permutations.append(permutation) #appends permutation
      
    return permutations #list of all possible tensor permutations that go [0, axial dimensions (in different orders), embedding dimension]

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x

class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial

# axial pos emb

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x

# attention

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
            #b = batch?, t = dim 1?, d = dim 2?, h = # of heads, e = size of heads BUT WOULDN'T THIS BREAK FOR MORE DIMS?????/

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

# axial attention class

class AxialAttention(nn.Module):
    # THIS IS THE ONE I USE
    """
    dim = embedding dimension
    num_dimensions = number of axial dimensions (images is 2, video is 3, or more)
    heads = number of heads for multi-head attention (dim % heads must not be 0)
    dim_heads = dimension of each head. defaults to dim // heads if not supplied
    dim_index = where is the embedding dimension
    sum_axial_out = whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
    """
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads' #WHY? CHECK 
        super().__init__()
        self.dim = dim 
        self.total_dimensions = num_dimensions + 2 #WHY ADD 2? 1 for channel, 1 for batch?
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index): #line 23, list of all tensor permutations that go [0, axial dimensions, embedding dimension]
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))
            #PermuteToFrom = 
            #SelfAttention = 

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        #if summing attention across axes
        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        #if hierarchical attention
        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out) #apply the next (attention) fxn in axial_attentions
        return out 

# axial image transformer

class AxialImageTransformer(nn.Module):
    def __init__(self, dim, depth, heads = 8, dim_heads = None, dim_index = 1, reversible = True, axial_pos_emb_shape = None):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)

        get_ff = lambda: nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding = 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, 3, padding = 1)
        )

        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(axial_pos_emb_shape) else nn.Identity()

        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList([PermuteToFrom(permutation, PreNorm(dim, SelfAttention(dim, heads, dim_heads))) for permutation in permutations])
            conv_functions = nn.ModuleList([get_ff(), get_ff()])
            layers.append(attn_functions)
            layers.append(conv_functions)            

        execute_type = ReversibleSequence if reversible else Sequential
        self.layers = execute_type(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)
