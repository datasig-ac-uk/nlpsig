Data splitting functionality
============================

``nlpsig`` has functionality to split data (given as a pair of some inputs, ``x_data``, and some corresponding labels, ``y_data``)
into general train/validation/test splits using ``nlpsig.DataSplits``, or into :math:`K` number of folds using ``nlpsig.Folds``.
These allow the user to return the data as ``torch.Tensor`` objects or ``torch.utils.data.dataloader.DataLoader`` objects ready to be used within a PyTorch model.

.. automodule:: nlpsig.classification_utils
   :members:
   :undoc-members:
   :show-inheritance:
