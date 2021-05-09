import numpy as np
import torch

from mykaggle.model.transformer import TransformerDecoder


class TestTransformer:

    def test_transformer_decoder(self):
        batch_size = 2
        length = 9

        hidden_dim = 4
        num_heads = 2
        ffn_dim = hidden_dim * num_heads
        num_layers = 3

        decoder = TransformerDecoder(hidden_dim, ffn_dim, num_heads, num_layers)
        decoder.eval()

        mem = torch.rand(batch_size, 5, hidden_dim)
        first_x = torch.rand(batch_size, 1, hidden_dim)

        x1 = first_x
        for t in range(length - 1):
            # create mask for autoregressive decoding
            mask = 1 - np.triu(np.ones((batch_size, (t + 1), (t + 1))), k=1).astype(np.uint8)
            mask = torch.autograd.Variable(torch.from_numpy(mask))
            y = decoder(x1, mem, x_mask=mask)
            x1 = torch.cat([x1, y[:, -1:]], dim=1)

        assert x1.shape == (batch_size, length, hidden_dim)

        x2 = first_x
        x_cache = [torch.empty(batch_size, 0, hidden_dim) for i in range(num_layers)]
        for t in range(length - 1):
            y, x_cache = decoder.forward_last(x2[:, -1:], x_cache, mem)
            x2 = torch.cat([x2, y], dim=1)

        assert x2.shape == (batch_size, length, hidden_dim)
        diff = torch.abs(x1 - x2)
        assert torch.sum(diff) < 1e-5
