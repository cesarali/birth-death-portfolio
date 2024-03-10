import torch
from sklearn import metrics

def area_under_the_curve(self ,adjacency_logits):
    if self.real_adjacency is not None:
        y = self.real_adjacency.cpu().long().numpy()
        y = y[self.mask_triu_elems]

        # from model aoc
        pred = torch.zeros_like(self.beta_prior).to(self.device)
        pred[y == 1] = adjacency_logits[y == 1]
        pred[y == 0] = 1. - adjacency_logits[y == 0]

        pred = pred.detach().cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        aoc = metrics.auc(fpr, tpr)

        return aoc
    else:
        return torch.Tensor([0.])


if __name__=="__main__":
    prior_distribution = Bernoulli(beta_prior)
    real_adjacency = prior_distribution.sample()

    pred = torch.zeros_like(beta_prior)
    pred[real_adjacency == 1] = beta_prior[real_adjacency == 1]
    pred[real_adjacency == 0] = 1. - beta_prior[real_adjacency == 0]

    pred = pred.view(-1).detach().cpu().numpy()
    y = real_adjacency.view(-1).cpu().long().numpy()

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=0)
    metrics.auc(fpr, tpr)