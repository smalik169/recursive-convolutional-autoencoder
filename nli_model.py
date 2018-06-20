from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn

activation_protos = {"relu": nn.ReLU,
                    "tanh": nn.Tanh}

class NLINet(nn.Module):
    def __init__(self, encoder, classifier_hid_dim=512, embed=False,
                 classifier_activation="relu", dropout=0.0):
        super(NLINet, self).__init__()

        self.encoder = encoder
        classifier_input_dim = 4 * self.encoder.latent_dim

        activation_proto = activation_protos[classifier_activation]

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_input_dim, classifier_hid_dim),
            activation_proto(),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hid_dim, classifier_hid_dim),
            activation_proto(),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hid_dim, 3)
            )

        self.embed = embed
        self.criterion = nn.CrossEntropyLoss(size_average=False)

    def forward(self, (sents1, lens1), (sents2, lens2)):
        u = self.encoder(sents1, sent_len=lens1, embed=self.embed)
        v = self.encoder(sents2, sent_len=lens2, embed=self.embed)

        features = torch.cat([u, v, torch.abs(u-v), u*v], 1)
        return self.classifier(features)

    def train_on(self, batch_iterator, optimizer, logger, clip=None):
        self.train()
        losses = []
        errs = []
        for batch_no, (s1, s2, labels) in enumerate(batch_iterator):
            self.zero_grad()
            s1[0].requires_grad_()
            s2[0].requires_grad_()

            output = self.forward(s1, s2)
            loss = self.criterion(output, labels)
            loss.backward()

            if clip:
                total_norm = nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()

            _, predictions = output.data.max(dim=1)
            err_rate = 100. * (predictions != labels).sum().item()
            losses.append(loss.item())
            errs.append(err_rate)
            logger.train_log(batch_no, {'loss': losses[-1], 'err': errs[-1],},
                             named_params=self.named_parameters, num_samples=labels.size(0))

        return losses, errs

    def eval_on(self, batch_iterator):
        self.eval()

        errs = 0
        samples = 0
        total_loss = 0
        batch_cnt = 0

        for batch_no, (s1, s2, labels) in enumerate(batch_iterator):
            self.zero_grad()
            output = self.forward(s1, s2)
            total_loss += self.criterion(output, labels).item()

            _, predictions = output.data.max(dim=1)
            errs += (predictions != labels.data).sum().item()
            samples += labels.size(0)

        return {'loss': total_loss / samples,
                'acc': 100 - 100. * errs / samples}
