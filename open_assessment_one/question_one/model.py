import torch
import torch.nn as nn

class RNNCell(nn.Module):
    """ Vanilla RNN cell: one time step of a recurrent neural network. """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True)-> None:
        """ Create the two linear maps for x_t and h_prev.
        
        Args:
            input_size: Size of one input vector (embedding size).
            hidden_size: Size of the hidden state.
            bias: If True, add a bias after W_x, x_t.
        
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #W_x x_t +b
        self.input_proj = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias,)
        #W_h h_prev (no second bias, one bias is enough)
        self.hidden_proj = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        
    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """ Run one RNN Step
        
        Args:
            x_t: Input at time t, shape (batch_size, input_size).
            h_prev: Hidden state at t-1, shape (batch_size, hidden_size)
        
        Returns:
            h_t: Hidden state at t, shape (batch_size, hidden_size).
        """
        #pre_act = W_x x_t + w_h h_prev + b
        pre_act = self.input_proj(x_t) + self.hidden_proj(h_prev)
        h_t = torch.tanh(pre_act)
        return h_t
    
class RNN(nn.Module):
    """ Vanilla RNN: apply RNNCell oncer per time step."""
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        """ Build one shared cell, Sam weights are reused at every t.
        
        Args:
            input_size: size of each input vector (embedding size).
            hidden_size: size of the hidden state.
            bias: Passed through to RNNCell.
        """ 
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size=input_size, hidden_size=hidden_size, bias=bias) 
        
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """ Start hidden state: zeros.
        
        Args:
            batch_size: Number of sequences in the batch.
            device: Same device as the input tensor.
            
        Returns:
            h_0 of shape (batch_size, hidden_size).
        """
        return torch.zeros(batch_size, self.hidden_size, device=device)
    
    def forward(self, x: torch.tensor, hidden: torch.Tensor | None = None,) -> tuple[torch.Tensor, torch.Tensor]:
        """ Run the cell over the full sequence.
        
        Args:
            x: Input sequence, shape (batch_size, time_steps, input_size).
            hidden: Optional h_0, shape (batch_size, hidden_size).
                If None, use zeros.
        
        Returns:
            output: Hidden states for every t, shape (batch_size, time_steps, hidden_size).
            h_t: Hidden state after the last token, shape (batch_size, hidden_size).
        """
        batch_size, time_steps, input_size = x.shape
        if hidden is None:
            h_t = self.init_hidden(batch_size=batch_size, device=x.device)
        else:
            h_t = hidden
        
        outputs = []
        for t in range(time_steps):
            # x_t is the t-th word vector for every item in the batch.
            x_t = x[:, t, :]
            h_t = self.cell.forward(x_t=x_t, h_prev=h_t)
            outputs.append(h_t)
        
        # list of (batch, hidden) -> (batch, time, hidden)
        output = torch.stack(outputs, dim=1)
        return output, h_t

class LSTMCell(nn.Module):
    """LSTM cell: one time step with a cell state and four gates."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        """Create maps from x_t and h_prev to the four gates.

        Args:
            input_size: Size of one input vector (embedding size).
            hidden_size: Size of h and c.
            bias: If True, add bias on the x_t map.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # one vector of size 4H, later split into i, f, g, o
        self.input_proj = nn.Linear(
            in_features=input_size,
            out_features=4 * hidden_size,
            bias=bias,
        )
        self.hidden_proj = nn.Linear(
            in_features=hidden_size,
            out_features=4 * hidden_size,
            bias=False,
        )

        # forget gate starts open (bias = 1) so early steps keep memory
        if bias:
            forget_start = hidden_size
            forget_end = 2 * hidden_size
            self.input_proj.bias.data[forget_start:forget_end].fill_(1.0)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one LSTM step.

        Args:
            x_t: Input at time t, shape (batch_size, input_size).
            h_prev: Hidden state at t-1, shape (batch_size, hidden_size).
            c_prev: Cell state at t-1, shape (batch_size, hidden_size).

        Returns:
            h_t: Hidden state at t, shape (batch_size, hidden_size).
            c_t: Cell state at t, shape (batch_size, hidden_size).
        """
        gates = self.input_proj(x_t) + self.hidden_proj(h_prev)
        input_gate, forget_gate, candidate, output_gate = torch.chunk(
            gates, chunks=4, dim=-1
        )

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        c_tilde = torch.tanh(candidate)
        o_t = torch.sigmoid(output_gate)

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class LSTM(nn.Module):
    """LSTM: apply LSTMCell once per time step."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        """Build one shared LSTM cell.

        Args:
            input_size: Size of each input vector (embedding size).
            hidden_size: Size of h and c.
            bias: Passed through to LSTMCell.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
        )

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Start states: zeros for both h and c.

        Args:
            batch_size: Number of sequences in the batch.
            device: Same device as the input tensor.

        Returns:
            h_0, c_0 each of shape (batch_size, hidden_size).
        """
        h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h_0, c_0

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the cell over the full sequence.

        Args:
            x: Input sequence, shape (batch_size, time_steps, input_size).
            hidden: Optional (h_0, c_0). If None, use zeros.

        Returns:
            output: Hidden states for every t,
                    shape (batch_size, time_steps, hidden_size).
            hidden: (h_t, c_t) after the last token,
                    each shape (batch_size, hidden_size).
        """
        batch_size, time_steps, input_size = x.shape
        if hidden is None:
            h_t, c_t = self.init_hidden(batch_size=batch_size, device=x.device)
        else:
            h_t, c_t = hidden

        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :]
            h_t, c_t = self.cell.forward(x_t=x_t, h_prev=h_t, c_prev=c_t)
            outputs.append(h_t)

        output = torch.stack(outputs, dim=1)
        return output, (h_t, c_t)

class NextWordModel(nn.Module):
    """Embedding + recurrent backbone + vocab head."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        backbone: str = "lstm",
        pad_id: int = 0,
        dropout: float = 0.2,
    ) -> None:
        """Build the language model.

        Args:
            vocab_size: Number of tokens in the vocab.
            embed_size: Size of each word vector.
            hidden_size: Size of the RNN/LSTM hidden state.
            backbone: "lstm" or "rnn".
            pad_id: Id of <pad>; embedding row stays zero.
            dropout: Dropout on hidden states before the head.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.backbone_name = backbone
        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_id,
        )

        if backbone == "lstm":
            self.backbone = LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                bias=True,
            )
        elif backbone == "rnn":
            self.backbone = RNN(
                input_size=embed_size,
                hidden_size=hidden_size,
                bias=True,
            )
        else:
            raise ValueError("backbone must be 'lstm' or 'rnn'")

        self.dropout = nn.Dropout(p=dropout)
        self.lm_head = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=True,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predict a next-word distribution at every position.

        Args:
            input_ids: Token ids, shape (batch_size, time_steps).

        Returns:
            logits: Shape (batch_size, time_steps, vocab_size).
        """
        # (batch, time) -> (batch, time, embed)
        x = self.embedding(input_ids)
        # output: (batch, time, hidden)
        output, hidden = self.backbone.forward(x=x, hidden=None)
        output = self.dropout(output)
        # (batch, time, hidden) -> (batch, time, vocab)
        logits = self.lm_head(output)
        return logits