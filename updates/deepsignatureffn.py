import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import signatory
import numpy as np

class DeepSigNet(nn.Module):
    def __init__(self, input_channels, output_channels, sig_d, post_dim, hidden_dim, output_dim, dropout_rate, add_time=False, augmentation_tp = 'Conv1d', augmentation_layers = ()):
        super(DeepSigNet, self).__init__()
        self.input_channels = input_channels
        self.add_time = add_time
        self.augmentation_tp = augmentation_tp
        #Convolution 
        self.conv = nn.Conv1d(input_channels, output_channels, 3, stride=1, padding=1).double()
        self.augment = signatory.Augment(in_channels=input_channels,
            layer_sizes = augmentation_layers,
            kernel_size=3,
            padding = 1,
            stride = 1,
            include_original=False,
            include_time=False).double()
        #Non-linearity
        self.tanh1 = nn.Tanh()
        #Signature
        self.signature = signatory.LogSignature(depth=sig_d)
        if self.add_time :
            input_dim = signatory.logsignature_channels(output_channels+1, sig_d) + post_dim
        else: 
            input_dim = signatory.logsignature_channels(output_channels, sig_d) + post_dim
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu1 = nn.ReLU()
        #Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Linear function 2: 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        # Linear function 3 (readout): 
        self.fc3 = nn.Linear(hidden_dim, output_dim)  


    def forward(self, x):
        #Convolution
        if (self.augmentation_tp == 'Conv1d'):
            out = self.conv(x[:,:self.input_channels, :]) #get only the path information
            out = self.tanh1(out)
            out = torch.transpose(out, 1,2) #swap dimensions
        else:
            out = self.augment(torch.transpose(x[:,:self.input_channels, :],1,2))


        #Add time for signature
        if self.add_time:
            out = torch.cat((out, torch.transpose(x[:,self.input_channels:(self.input_channels+1), :], 1,2)), dim=2)

        #Signature
        out = self.signature(out)

        #Combine Last Post Embedding
        out = torch.cat((out, x[:, self.input_channels:(self.input_channels+1),:].max(2)[0], x[:,(self.input_channels+1):, 0]), dim=1)

        #FFN: Linear function 1
        out = self.fc1(out.float())
        # Non-linearity 1
        out = self.relu1(out)
        #Dropout
        out = self.dropout(out)

        #FFN: Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        #Dropout
        out = self.dropout(out)

        #FFN: Linear function 3 (readout)
        out = self.fc3(out)
        return out

class StackedDeepSigNet(nn.Module):
    def __init__(self, input_channels, output_channels, sig_d, hidden_dim_lstm, post_dim, hidden_dim, output_dim, dropout_rate, add_time=False, augmentation_tp = 'Conv1d', augmentation_layers = (), BiLSTM=False, comb_method='gated_addition', blocks = 2):
        super(StackedDeepSigNet, self).__init__()
        self.input_channels = input_channels
        self.add_time = add_time 
        self.augmentation_tp = augmentation_tp 
        self.comb_method = comb_method
        self.blocks = blocks
        input_bert_dim = 384

        #Convolution
        self.conv = nn.Conv1d(input_channels, output_channels, 3, stride=1, padding=1).double()
        self.augment = signatory.Augment(in_channels=input_channels,
                    layer_sizes = augmentation_layers,
                    kernel_size=3,
                    padding = 1,
                    stride = 1,
                    include_original=False,
                    include_time=False).double()
        #Non-linearity
        self.tanh1 = nn.Tanh()
        #Signature with lift
        self.signature1 = signatory.LogSignature(depth=sig_d, stream=True)
        if self.add_time :
            input_dim_lstm = signatory.logsignature_channels(output_channels+1, sig_d)
        else: 
            input_dim_lstm = signatory.logsignature_channels(output_channels, sig_d)
        
        #additional blocks in the network
        if (blocks > 2):
            self.lstm0 = nn.LSTM(input_size=input_dim_lstm, hidden_size=hidden_dim_lstm[-2], num_layers=1, batch_first=True, bidirectional=False).double()
            self.signature1b = signatory.LogSignature(depth=sig_d, stream=True)
            input_dim_lstm = signatory.logsignature_channels(hidden_dim_lstm[-2], sig_d)

        if comb_method=='concatenation':
            #LSTM
            if BiLSTM:
                self.lstm = nn.LSTM(input_size=input_dim_lstm, hidden_size=hidden_dim_lstm[-1], num_layers=1, batch_first=True, bidirectional=True).double()
                input_dim = signatory.logsignature_channels(2*hidden_dim_lstm[-1], sig_d) + post_dim
            else:
                self.lstm = nn.LSTM(input_size=input_dim_lstm, hidden_size=hidden_dim_lstm[-1], num_layers=1, batch_first=True, bidirectional=False).double()
                input_dim = signatory.logsignature_channels(hidden_dim_lstm[-1], sig_d) + post_dim
        elif comb_method=='gated_addition':
            if BiLSTM:
                self.lstm = nn.LSTM(input_size=input_dim_lstm, hidden_size=hidden_dim_lstm[-1], num_layers=1, batch_first=True, bidirectional=True).double()
                input_dim = input_bert_dim
                input_gated_linear = signatory.logsignature_channels(2*hidden_dim_lstm[-1], sig_d) + 1
                self.fc_scale = nn.Linear(input_gated_linear, input_bert_dim)
                #define the scaler parameter
                self.scaler = torch.nn.Parameter(torch.zeros(1,input_bert_dim))
            else:
                self.lstm = nn.LSTM(input_size=input_dim_lstm, hidden_size=hidden_dim_lstm[-1], num_layers=1, batch_first=True, bidirectional=False).double()
                input_dim = input_bert_dim
                input_gated_linear = signatory.logsignature_channels(hidden_dim_lstm[-1], sig_d) + 1
                self.fc_scale = nn.Linear(input_gated_linear, input_bert_dim)
                #define the scaler parameter
                self.scaler = torch.nn.Parameter(torch.zeros(1,input_bert_dim))

        #Signature without lift
        self.signature2 = signatory.LogSignature(depth=sig_d, stream=False)
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu1 = nn.ReLU()
        #Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Linear function 2: 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        # Linear function 3 (readout): 
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        #Convolution
        if (self.augmentation_tp == 'Conv1d'):
            out = self.conv(x[:,:self.input_channels, :]) #get only the path information
            out = self.tanh1(out)
            out = torch.transpose(out, 1,2) #swap dimensions
        else:
            out = self.augment(torch.transpose(x[:,:self.input_channels, :],1,2))
        
        #Add time for signature
        if self.add_time:
            out = torch.cat((out, torch.transpose(x[:,self.input_channels:(self.input_channels+1), :], 1,2)), dim=2)
        #Signature
        out = self.signature1(out)
        #if more blocks
        if (self.blocks > 2):
            out, (_, _) = self.lstm0(out)
            out = self.signature1b(out)
        #LSTM
        out, (_, _) = self.lstm(out)
        #Signature
        out = self.signature2(out)
        #Combine Last Post Embedding
        if self.comb_method=='concatenation':
            out = torch.cat((out, x[:, self.input_channels:(self.input_channels+1),:].max(2)[0], x[:,(self.input_channels+1):, 0]), dim=1)
        elif self.comb_method=='gated_addition':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+1),:].max(2)[0]), dim=1)
            out_gated = self.fc_scale(out_gated.float())
            out_gated = self.tanh1(out_gated)
            out_gated = torch.mul(self.scaler, out_gated)
            #concatenation with bert output
            out = out_gated + x[:,(self.input_channels+1):, 0]


        #FFN: Linear function 1
        out = self.fc1(out.float())
        # Non-linearity 1
        out = self.relu1(out)
        #Dropout
        out = self.dropout(out)

        #FFN: Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        #Dropout
        out = self.dropout(out)

        #FFN: Linear function 3 (readout)
        out = self.fc3(out)
        return out


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu1 = nn.ReLU()

        #Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Linear function 2: 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3 (readout): 
        self.fc3 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
         # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        #Dropout
        out = self.dropout(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        #Dropout
        out = self.dropout(out)

        # Linear function 3 (readout)
        out = self.fc3(out)
        return out

#source: https://github.com/clcarwin/focal_loss_pytorch
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


#source: https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
class ClassBalanced_FocalLoss(nn.Module):
    def __init__(self, gamma, beta, no_of_classes, samples_per_cls):
        super(ClassBalanced_FocalLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.no_of_classes = no_of_classes
        self.samples_per_cls = samples_per_cls
        
    def forward(self, logits, labels):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        samples_per_cls: A python list of size [no_of_classes].
        no_of_classes: total number of classes. int
        loss_type: string. One of "sigmoid", "focal", "softmax".
        beta: float. Hyperparameter for Class balanced loss.
        gamma: float. Hyperparameter for Focal loss.
        Returns:
        cb_loss: A float tensor representing class balanced loss
        """
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,self.no_of_classes)

        cb_loss = self.focal_loss(labels_one_hot, logits, weights)
        return cb_loss

    def focal_loss(self, labels, logits, alpha):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """    
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss