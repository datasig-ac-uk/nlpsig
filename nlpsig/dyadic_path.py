from typing import Optional, Tuple

import numpy as np
import signatory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DyadicSignatures:
    """
    Class for obtaining path signatures
    """

    def __init__(
        self,
        original_size: int,
        dim: int,
        dyadic_path_min: float,
        dyadic_path_max: float,
        sig_dim: int = 3,
        intervals: int = 1 / 12,
        k_history: Optional[int] = None,
        embedding_tp: str = "sentence",
        method: str = "attention",
        history_tp: str = "sig",
        add_time: bool = False,
    ):
        """
        Class for obtaining path signatures

        Parameters
        ----------
        original_size : int
            Total number of sentences
        dim : int
            Dimension of features
        dyadic_path_min : float
            Minimum year to start from
        dyadic_path_max : float
            Maximum end to end on
        sig_dim : int, optional
            Dimension of signature (does this), by default 3
        intervals : int, optional
            _description_, by default 1/12
        k_history : Optional[int], optional
            _description_, by default None
        embedding_tp : str, optional
            _description_, by default "sentence"
        method : str, optional
            _description_, by default "attention"
        history_tp : str, optional
            _description_, by default "sig"
        add_time : bool, optional
            Whether or not to include time?, by default False
        """
        self.original_size = original_size
        self.dim = dim
        self.sig_dim = sig_dim
        self.dyadic_path_min = dyadic_path_min
        self.dyadic_path_max = dyadic_path_max
        self.intervals = intervals
        self.embedding_tp = embedding_tp
        self.method = method
        self.channels = signatory.signature_channels(self.dim, self.sig_dim)
        self.history_tp = history_tp
        self.add_time = add_time
        self.k_history = k_history
        self.dyadic_path_dt = np.arange(
            self.dyadic_path_min, self.dyadic_path_max, self.intervals
        ).tolist()

    def _compute_signature(self, timeline: torch.Tensor) -> torch.Tensor:
        """
        [Private] Compute per sample dyadic signature function

        Parameters
        ----------
        timeline : torch.Tensor
            Path for a particular timeline

        Returns
        -------
        torch.Tensor
            Path signature for a particular timeline
        """
        count_points = 0
        dt = 1
        ind = 0
        # overall empty array to save results of dimensions: #samples x #intervals x #final dimensions
        # int( (self.dim ** (self.sig_d+1) - self.dim) * (self.dim - 1) ** (-1) )#overall array of signatures
        dyadic_signatures = torch.empty(
            (len(self.dyadic_path_dt) - 3, int(self.channels))
        )  # overall array of signatures
        dyadic_signatures[:] = 0
        # temp array to save the signature of each dyadic path: #sample x #final dimension
        sig_dt = torch.empty((1, int(self.channels)))  # temp array
        # array that saves the last index for each interval that falls in the dyadic path
        last_index_dt = torch.zeros((1, len(self.dyadic_path_dt) - 3))
        # save the index of the last point in the previous interval
        ind_start = 0
        ind_end = 0
        # range of the drt we are in
        low_range = self.dyadic_path_dt[0]
        # efficient precomputations
        path_class = signatory.Path(
            timeline.reshape(1, timeline.shape[0], timeline.shape[1]), self.sig_dim
        )
        # count the data points of timeline
        num_points = torch.count_nonzero(timeline[:, 0]).item()
        while (ind <= timeline.shape[0]) & (ind <= num_points):
            upper_range = self.dyadic_path_dt[dt]
            if (
                ind >= timeline.shape[0]
            ):  # checks that we didn't have an index larger than the timeline which would through an error
                date_t = upper_range
            else:
                date_t = timeline[ind][0].item()
            if (date_t < upper_range) & (date_t != 0):
                count_points += 1
                ind += 1
                ind_end += 1
            else:
                if count_points > 1:
                    if ind_start != 0:
                        ind_start -= 1  # this is because we need the previous point as a basepoint x0 for the dyadic path
                    if (
                        ind_end + 1 == num_points
                    ):  # get data from the next path if there is single data point
                        count_points += 1
                        ind += 1
                        ind_end += 1
                    # calculate regular path
                    sig_dt[:] = path_class.signature(ind_start, ind_end)
                else:
                    # case of empty path
                    sig_dt[:] = 0
                # add signgature in overall array
                dyadic_signatures[(dt - 1), :] = sig_dt[:]
                # zero out temp signature array
                sig_dt = torch.empty((1, int(self.channels)))
                # assign last index
                last_index_dt[:, (dt - 1)] = ind_end - 1 if (ind_end > 0) else 0
                low_range = self.dyadic_path_dt[dt]  # noqa: F841
                dt += 1
                upper_range = self.dyadic_path_dt[dt]
                if ind == num_points:
                    ind += 1
                if count_points != 1:
                    # push data to the next path if there is single data point
                    count_points = 0
                    ind_start = ind_end
                else:
                    print(
                        "single points that got pushed/pulled to near path ",
                        ind_start,
                        ind_end,
                        num_points,
                    )
        return (
            dyadic_signatures.reshape(
                1, dyadic_signatures.shape[0], dyadic_signatures.shape[1]
            ),
            last_index_dt,
        )

    def compute_signatures(self, path):
        """
        Compute dyadic signatures for the whole array

        Parameters
        ----------
        path : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        sig = torch.empty((0, len(self.dyadic_path_dt) - 3, int(self.channels)))
        last_index_dt_all = torch.empty((0, len(self.dyadic_path_dt) - 3))
        for sample in range(path.shape[0]):
            # calculate signature per timeline (sample)
            timeline = path[sample, :, :]
            # call dyadic path function
            s1, i1 = self._compute_signature(timeline)
            # concat in torch array
            sig = torch.cat((sig, s1), 0)
            last_index_dt_all = torch.cat((last_index_dt_all, i1), 0)
        return sig, last_index_dt_all

    def _combine_signatures(self, sig):
        """
        [Private] Combine dyadic signature paths into larger ones

        Parameters
        ----------
        sig : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        sig_cum = torch.empty((sig.shape[0], sig.shape[1], sig.shape[2]))
        sig_cum[:, 0, :] = sig[:, 0, :]
        for i in range(1, sig_cum.shape[1]):
            if sig[:, i, :].sum().item() != 0:
                sig_cum[:, i, :] = signatory.signature_combine(
                    sig_cum[:, i - 1, :], sig[:, i, :], self.dim, self.sig_dim
                )
            else:
                sig_cum[:, i, :] = sig_cum[:, i - 1, :]
        return sig_cum

    def combine_signatures(self, sig):
        """
        Combine dyadic signature paths into larger ones

        Parameters
        ----------
        sig : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        sig_combined = self._combine_signatures(sig)
        return sig_combined

    def _concat_features(
        self,
        timeline,
        path_class,
        last_index_dt_all,
        sig_combined,
        sample,
        ind,
        i,
        bert_embeddings=None,
        time_feature=None,
    ):
        """
        [Private] Construct features of history + current post for each sample

        Parameters
        ----------
        timeline : _type_
            _description_
        path_class : _type_
            _description_
        last_index_dt_all : _type_
            _description_
        sig_combined : _type_
            _description_
        sample : _type_
            _description_
        ind : _type_
            _description_
        i : _type_
            _description_
        bert_embeddings : _type_, optional
            _description_, by default None
        time_feature : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # FOR ONE SAMPLE FIRST (1. case of first sample, 2. case of sample in first bucket. 3. case of sample not in first bucket) 4. sample first in next bucket
        ind_x = last_index_dt_all[
            sample,
            (last_index_dt_all[sample, :] < ind) & (last_index_dt_all[sample, :] != 0),
        ]
        if self.k_history is None:
            if len(ind_x != 0):
                dyadic_ind = int(ind_x.max().item())
                try:
                    dyadic_int_ind = (
                        (last_index_dt_all[sample, :] == dyadic_ind)
                        .nonzero(as_tuple=True)[0]
                        .item()
                    )
                except Exception as err:
                    print(f"[WARNING] {err}")
                    dyadic_int_ind = (
                        (last_index_dt_all[sample, :] == dyadic_ind)
                        .nonzero(as_tuple=True)[0][0]
                        .item()
                    )
                # first part of path
                sig_intervals = sig_combined[sample, dyadic_int_ind, :].reshape(
                    1, sig_combined.shape[2]
                )
                if ind == dyadic_ind + 1:
                    history = sig_intervals
                else:
                    # residual part of path
                    sig_res = path_class.signature(dyadic_ind, ind).to(torch.float)
                    # combine
                    history = signatory.signature_combine(
                        sig_intervals, sig_res, self.dim, self.sig_dim
                    )
            elif ind <= 1:
                history = torch.zeros((1, self.channels))
            else:
                history = path_class.signature(0, ind).to(torch.float)
        else:
            # if we include only the last k-posts in the history signature
            if ind > 1:
                start_ind_history = ind - self.k_history
                if start_ind_history < 0:
                    start_ind_history = 0
                history = path_class.signature(start_ind_history, ind).to(torch.float)
            else:
                history = torch.zeros((1, self.channels))
        # logsignature case calculation
        if self.history_tp == "log":
            history = signatory.signature_to_logsignature(
                history, self.dim, self.sig_dim
            )
        # TYPE OF POST EMBEDDING
        if self.embedding_tp == "sentence":
            post_i = bert_embeddings[i, :].reshape(1, bert_embeddings.shape[1])
        elif self.embedding_tp == "reduced":
            post_i = timeline[ind, :].reshape(1, timeline.shape[1])  # fixed
        else:
            print(
                "ERROR: You need to specify a valid embedding type between: sentence , reduced"
            )
        # CONCATENATION METHOD
        if self.method == "attention":
            if history.shape[1] <= post_i.shape[1]:
                history = torch.cat(
                    (history, torch.zeros((1, post_i.shape[1] - history.shape[1]))), 1
                )
            else:
                post_i = torch.cat(
                    (post_i, torch.zeros((1, history.shape[1] - post_i.shape[1]))), 1
                )
            attention = DotProductAttention(post_i.shape[1])
            query = post_i.reshape(1, 1, post_i.shape[1])
            context = history.reshape(1, 1, history.shape[1]).to(torch.float)
            output, _ = attention(query, context)
            output = output.reshape(1, output.shape[2])
        elif self.method == "concatenation":
            output = torch.cat((history, post_i), 1)
        else:
            print(
                "ERROR: You need to specify a valid feature creation method between: attention , concatenation"
            )
            pass
        # TIME AS A FEATURE
        if self.add_time:
            output = torch.cat(
                (output, time_feature[i, :].reshape(1, time_feature.shape[1])), 1
            )
        return output

    def create_features(
        self,
        path,
        sig_combined,
        last_index_dt_all,
        bert_embeddings=None,
        time_feature=None,
    ):
        """
        Construct features of history + current post for each sample

        Parameters
        ----------
        path : _type_
            _description_
        sig_combined : _type_
            _description_
        last_index_dt_all : _type_
            _description_
        bert_embeddings : _type_, optional
            _description_, by default None
        time_feature : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if self.embedding_tp == "sentence":
            emb_dim = bert_embeddings.shape[1]
        else:
            emb_dim = path.shape[2]
        if self.history_tp == "log":
            channel_dim = signatory.logsignature_channels(self.dim, self.sig_dim)
        else:
            channel_dim = self.channels
        if self.method == "attention":
            overall_dim = max(channel_dim, emb_dim)
        else:
            overall_dim = channel_dim + emb_dim
        if self.add_time:
            overall_dim += 1
        x_train = torch.empty((self.original_size, overall_dim))
        i = 0
        for sample in range(path.shape[0]):
            # calculate signature per timeline (sample)
            timeline = path[sample, :, :]
            # for specific sample - precomputation
            path_class = signatory.Path(
                timeline.reshape(1, timeline.shape[0], timeline.shape[1]), self.sig_dim
            )
            for ind in range(timeline.shape[0]):
                if timeline[ind, 0].item() != 0:
                    x_train[i, :] = self._concat_features(
                        timeline,
                        path_class,
                        last_index_dt_all,
                        sig_combined,
                        sample,
                        ind,
                        i,
                        bert_embeddings,
                        time_feature,
                    )
                    i += 1
        return x_train


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and
    apply a softmax function to obtain the weights on the values

    source: https://github.com/sooftware/attentions/blob/master/attentions.py
    """

    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = (  # noqa: F841
            query.size(0),
            query.size(2),
            value.size(1),
        )

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(
            batch_size, -1, input_size
        )
        context = torch.bmm(attn, value)

        return context, attn
