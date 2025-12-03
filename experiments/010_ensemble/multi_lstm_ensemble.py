import torch
import torch.nn as nn


class MultiLSTMEnsemble(nn.Module):
    """
    Multi-LSTM architecture where multiple LSTMs process the input,
    and their outputs are concatenated and fed to a final LSTM layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_lstms: int = 3,
        num_layers: int = 1,
        final_hidden_size: int = None,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        """
        Args:
            input_size: Size of input features
            hidden_size: Hidden size for each parallel LSTM
            num_lstms: Number of parallel LSTMs
            num_layers: Number of layers in each LSTM
            final_hidden_size: Hidden size for final LSTM (if None, uses hidden_size)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTMs
        """
        super(MultiLSTMEnsemble, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_lstms = num_lstms
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create multiple parallel LSTMs
        self.parallel_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
            for _ in range(num_lstms)
        ])

        # Final LSTM that takes concatenated outputs as input
        final_input_size = hidden_size * self.num_directions * num_lstms
        self.final_hidden_size = final_hidden_size or hidden_size

        self.final_lstm = nn.LSTM(
            input_size=final_input_size,
            hidden_size=self.final_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

    def forward(self, x, return_all_outputs=False):
        """
        Forward pass through the ensemble.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            return_all_outputs: If True, returns outputs from all LSTMs

        Returns:
            final_output: Output from final LSTM (batch_size, seq_len, final_hidden_size * num_directions)
            final_hidden: Tuple of (h_n, c_n) from final LSTM
            parallel_outputs: (Optional) List of outputs from parallel LSTMs
        """
        batch_size, seq_len, _ = x.size()

        # Process input through all parallel LSTMs
        parallel_outputs = []
        parallel_hiddens = []

        for lstm in self.parallel_lstms:
            output, (h_n, c_n) = lstm(x)
            parallel_outputs.append(output)
            parallel_hiddens.append((h_n, c_n))

        # Concatenate outputs from all parallel LSTMs along feature dimension
        # Shape: (batch_size, seq_len, hidden_size * num_directions * num_lstms)
        concatenated_outputs = torch.cat(parallel_outputs, dim=-1)

        # Pass concatenated outputs through final LSTM
        final_output, (final_h_n, final_c_n) = self.final_lstm(concatenated_outputs)

        if return_all_outputs:
            return final_output, (final_h_n, final_c_n), parallel_outputs
        else:
            return final_output, (final_h_n, final_c_n)


class MultiLSTMEnsembleWithProjection(nn.Module):
    """
    Multi-LSTM architecture with optional projection layers before concatenation.
    This allows for better control over the embedding space before the final LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_lstms: int = 3,
        num_layers: int = 1,
        projection_size: int = None,
        final_hidden_size: int = None,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        """
        Args:
            input_size: Size of input features
            hidden_size: Hidden size for each parallel LSTM
            num_lstms: Number of parallel LSTMs
            num_layers: Number of layers in each LSTM
            projection_size: Size to project each LSTM output to (if None, no projection)
            final_hidden_size: Hidden size for final LSTM (if None, uses hidden_size)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTMs
        """
        super(MultiLSTMEnsembleWithProjection, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_lstms = num_lstms
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.projection_size = projection_size

        # Create multiple parallel LSTMs
        self.parallel_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
            for _ in range(num_lstms)
        ])

        # Optional projection layers
        if projection_size is not None:
            self.projection_layers = nn.ModuleList([
                nn.Linear(hidden_size * self.num_directions, projection_size)
                for _ in range(num_lstms)
            ])
            final_input_size = projection_size * num_lstms
        else:
            self.projection_layers = None
            final_input_size = hidden_size * self.num_directions * num_lstms

        # Final LSTM
        self.final_hidden_size = final_hidden_size or hidden_size
        self.final_lstm = nn.LSTM(
            input_size=final_input_size,
            hidden_size=self.final_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, return_all_outputs=False):
        """
        Forward pass through the ensemble with projection.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            return_all_outputs: If True, returns outputs from all LSTMs

        Returns:
            final_output: Output from final LSTM
            final_hidden: Tuple of (h_n, c_n) from final LSTM
            parallel_outputs: (Optional) List of outputs from parallel LSTMs
        """
        batch_size, seq_len, _ = x.size()

        # Process input through all parallel LSTMs
        parallel_outputs = []

        for i, lstm in enumerate(self.parallel_lstms):
            output, _ = lstm(x)

            # Apply projection if available
            if self.projection_layers is not None:
                output = self.projection_layers[i](output)
                if self.dropout is not None:
                    output = self.dropout(output)

            parallel_outputs.append(output)

        # Concatenate outputs
        concatenated_outputs = torch.cat(parallel_outputs, dim=-1)

        # Pass through final LSTM
        final_output, (final_h_n, final_c_n) = self.final_lstm(concatenated_outputs)

        if return_all_outputs:
            return final_output, (final_h_n, final_c_n), parallel_outputs
        else:
            return final_output, (final_h_n, final_c_n)


# Example usage
if __name__ == "__main__":
    # Example 1: Basic Multi-LSTM Ensemble
    print("=" * 50)
    print("Example 1: Basic Multi-LSTM Ensemble")
    print("=" * 50)

    batch_size = 4
    seq_len = 10
    input_size = 64
    hidden_size = 128
    num_lstms = 3

    model = MultiLSTMEnsemble(
        input_size=input_size,
        hidden_size=hidden_size,
        num_lstms=num_lstms,
        num_layers=2,
        dropout=0.2
    )

    x = torch.randn(batch_size, seq_len, input_size)
    output, (h_n, c_n) = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {h_n.shape}")
    print(f"Cell state shape: {c_n.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Example 2: Multi-LSTM with Projection
    print("\n" + "=" * 50)
    print("Example 2: Multi-LSTM Ensemble with Projection")
    print("=" * 50)

    model_proj = MultiLSTMEnsembleWithProjection(
        input_size=input_size,
        hidden_size=hidden_size,
        num_lstms=num_lstms,
        num_layers=2,
        projection_size=64,  # Project each LSTM output to smaller dimension
        final_hidden_size=256,
        dropout=0.2
    )

    output_proj, (h_n_proj, c_n_proj), parallel_outs = model_proj(x, return_all_outputs=True)

    print(f"Input shape: {x.shape}")
    print(f"Number of parallel LSTMs: {len(parallel_outs)}")
    print(f"Each parallel output shape: {parallel_outs[0].shape}")
    print(f"Final output shape: {output_proj.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model_proj.parameters()):,}")

    # Example 3: Bidirectional version
    print("\n" + "=" * 50)
    print("Example 3: Bidirectional Multi-LSTM Ensemble")
    print("=" * 50)

    model_bidir = MultiLSTMEnsemble(
        input_size=input_size,
        hidden_size=hidden_size,
        num_lstms=num_lstms,
        num_layers=1,
        bidirectional=True
    )

    output_bidir, _ = model_bidir(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (bidirectional): {output_bidir.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model_bidir.parameters()):,}")
