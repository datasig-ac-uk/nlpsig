import signatory
import torch
import torch.nn as nn


class DeepSigNet(nn.Module):
    """ """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        sig_depth: int,
        post_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
        add_time: bool = False,
        augmentation_tp: str = "Conv1d",
        augmentation_layers: tuple = (),
    ):
        """


        Parameters
        ----------
        input_channels : int
            _description_
        output_channels : int
            _description_
        sig_depth : int
            # is this the depth of the signature? like level of sig_depth
            _description_
        post_dim : int
            # is this the dimension of the embeddings? should be renamed to something else
            _description_
        hidden_dim : int
            _description_
        output_dim : int
            _description_
        dropout_rate : float
            _description_
        add_time : bool, optional
            # why bother with this when we can just pass in whatever we want
            # PrepareData should do all the data stuff
            _description_, by default False
        augmentation_tp : str, optional
            # what does tp mean again?
            _description_, by default "Conv1d"
        augmentation_layers : tuple, optional
            # what is this and why is it just a default tuple?
            _description_, by default ()
        """
        super(DeepSigNet, self).__init__()
        self.input_channels = input_channels
        self.add_time = add_time
        self.augmentation_tp = augmentation_tp
        # Convolution
        self.conv = nn.Conv1d(
            input_channels, output_channels, 3, stride=1, padding=1
        ).double()
        self.augment = signatory.Augment(
            in_channels=input_channels,
            layer_sizes=augmentation_layers,
            kernel_size=3,
            padding=1,
            stride=1,
            include_original=False,
            include_time=False,
        ).double()
        # Non-linearity
        self.tanh1 = nn.Tanh()
        # Signature
        self.signature = signatory.LogSignature(depth=sig_depth)
        if self.add_time:
            input_dim = (
                signatory.logsignature_channels(output_channels + 1, sig_depth)
                + post_dim
            )
        else:
            input_dim = (
                signatory.logsignature_channels(output_channels, sig_depth) + post_dim
            )
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.relu1 = nn.ReLU()
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Linear function 2:
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        # Linear function 3 (readout):
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Convolution
        if self.augmentation_tp == "Conv1d":
            out = self.conv(
                x[:, : self.input_channels, :]
            )  # get only the path information
            out = self.tanh1(out)
            out = torch.transpose(out, 1, 2)  # swap dimensions
        else:
            out = self.augment(torch.transpose(x[:, : self.input_channels, :], 1, 2))

        # Add time for signature
        if self.add_time:
            out = torch.cat(
                (
                    out,
                    torch.transpose(
                        x[:, self.input_channels : (self.input_channels + 1), :], 1, 2
                    ),
                ),
                dim=2,
            )

        # Signature
        out = self.signature(out)

        # Combine Last Post Embedding
        out = torch.cat(
            (
                out,
                x[:, self.input_channels : (self.input_channels + 1), :].max(2)[0],
                x[:, (self.input_channels + 1) :, 0],
            ),
            dim=1,
        )

        # FFN: Linear function 1
        out = self.fc1(out.float())
        # Non-linearity 1
        out = self.relu1(out)
        # Dropout
        out = self.dropout(out)

        # FFN: Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        # Dropout
        out = self.dropout(out)

        # FFN: Linear function 3 (readout)
        out = self.fc3(out)
        return out


class StackedDeepSigNet(nn.Module):
    """
    __summary__
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        sig_depth: int,
        hidden_dim_lstm: int,
        post_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
        add_time: bool = False,
        augmentation_tp: str = "Conv1d",
        augmentation_layers: tuple = (),
        BiLSTM: bool = False,
        comb_method: str = "gated_addition",
        blocks: int = 2,
    ):
        """
        _summary_

        Parameters
        ----------
        input_channels : int
            _description_
        output_channels : int
            _description_
        sig_depth : int
            _description_
        hidden_dim_lstm : int
            _description_
        post_dim : int
            _description_
        hidden_dim : int
            _description_
        output_dim : int
            _description_
        dropout_rate : float
            _description_
        add_time : bool, optional
            _description_, by default False
        augmentation_tp : str, optional
            _description_, by default "Conv1d"
        augmentation_layers : tuple, optional
            _description_, by default ()
        BiLSTM : bool, optional
            _description_, by default False
        comb_method : str, optional
            _description_, by default "gated_addition"
        blocks : int, optional
            _description_, by default 2
        """
        super(StackedDeepSigNet, self).__init__()
        self.input_channels = input_channels
        self.add_time = add_time
        self.augmentation_tp = augmentation_tp
        self.comb_method = comb_method
        self.blocks = blocks
        input_bert_dim = 384

        # Convolution
        self.conv = nn.Conv1d(
            input_channels, output_channels, 3, stride=1, padding=1
        ).double()
        self.augment = signatory.Augment(
            in_channels=input_channels,
            layer_sizes=augmentation_layers,
            kernel_size=3,
            padding=1,
            stride=1,
            include_original=False,
            include_time=False,
        ).double()
        # Non-linearity
        self.tanh1 = nn.Tanh()
        # Signature with lift
        self.signature1 = signatory.LogSignature(depth=sig_depth, stream=True)
        if self.add_time:
            input_dim_lstm = signatory.logsignature_channels(
                output_channels + 1, sig_depth
            )
        else:
            input_dim_lstm = signatory.logsignature_channels(output_channels, sig_depth)

        # additional blocks in the network
        if blocks > 2:
            self.lstm0 = nn.LSTM(
                input_size=input_dim_lstm,
                hidden_size=hidden_dim_lstm[-2],
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            ).double()
            self.signature1b = signatory.LogSignature(depth=sig_depth, stream=True)
            input_dim_lstm = signatory.logsignature_channels(
                hidden_dim_lstm[-2], sig_depth
            )

        mult = 2 if BiLSTM else 1
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim_lstm,
            hidden_size=hidden_dim_lstm[-1],
            num_layers=1,
            batch_first=True,
            bidirectional=BiLSTM,
        ).double()
        if comb_method == "concatenation":
            input_dim = (
                signatory.logsignature_channels(mult * hidden_dim_lstm[-1], sig_depth)
                + post_dim
            )
        elif comb_method == "gated_addition":
            input_dim = input_bert_dim
            input_gated_linear = (
                signatory.logsignature_channels(mult * hidden_dim_lstm[-1], sig_depth)
                + 1
            )
            self.fc_scale = nn.Linear(input_gated_linear, input_bert_dim)
            # define the scaler parameter
            self.scaler = torch.nn.Parameter(torch.zeros(1, input_bert_dim))

        # Signature without lift
        self.signature2 = signatory.LogSignature(depth=sig_depth, stream=False)
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.relu1 = nn.ReLU()
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Linear function 2:
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        # Linear function 3 (readout):
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Convolution
        if self.augmentation_tp == "Conv1d":
            out = self.conv(
                x[:, : self.input_channels, :]
            )  # get only the path information
            out = self.tanh1(out)
            out = torch.transpose(out, 1, 2)  # swap dimensions
        else:
            out = self.augment(torch.transpose(x[:, : self.input_channels, :], 1, 2))

        # Add time for signature
        if self.add_time:
            out = torch.cat(
                (
                    out,
                    torch.transpose(
                        x[:, self.input_channels : (self.input_channels + 1), :], 1, 2
                    ),
                ),
                dim=2,
            )
        # Signature
        out = self.signature1(out)
        # if more blocks
        if self.blocks > 2:
            out, (_, _) = self.lstm0(out)
            out = self.signature1b(out)
        # LSTM
        out, (_, _) = self.lstm(out)
        # Signature
        out = self.signature2(out)
        # Combine Last Post Embedding
        if self.comb_method == "concatenation":
            out = torch.cat(
                (
                    out,
                    x[:, self.input_channels : (self.input_channels + 1), :].max(2)[0],
                    x[:, (self.input_channels + 1) :, 0],
                ),
                dim=1,
            )
        elif self.comb_method == "gated_addition":
            out_gated = torch.cat(
                (
                    out,
                    x[:, self.input_channels : (self.input_channels + 1), :].max(2)[0],
                ),
                dim=1,
            )
            out_gated = self.fc_scale(out_gated.float())
            out_gated = self.tanh1(out_gated)
            out_gated = torch.mul(self.scaler, out_gated)
            # concatenation with bert output
            out = out_gated + x[:, (self.input_channels + 1) :, 0]

        # FFN: Linear function 1
        out = self.fc1(out.float())
        # Non-linearity 1
        out = self.relu1(out)
        # Dropout
        out = self.dropout(out)

        # FFN: Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        # Dropout
        out = self.dropout(out)

        # FFN: Linear function 3 (readout)
        out = self.fc3(out)
        return out
