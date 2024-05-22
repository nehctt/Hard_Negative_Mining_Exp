from sentence_transformers import losses
import torch
import torch.nn.functional as F


class InfoNCELoss(losses.MultipleNegativesRankingLoss):

    def __init__(self, model, similarity_fct):
        super().__init__(model=model, similarity_fct=similarity_fct)

    def forward(self, sentence_features, labels):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        
        return self.cross_entropy_loss(scores, labels)


class InfoNCEDynamicMarginLoss(losses.MultipleNegativesRankingLoss):

    def __init__(self, model, similarity_fct):
        super().__init__(model=model, similarity_fct=similarity_fct)

    def forward(self, sentence_features, labels):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        
        # pos_mean = torch.diag(scores).mean()
        # neg_mean = (scores.sum() - torch.diag(scores).sum()) / (scores.numel() - len(scores))
        # margin = (pos_mean + neg_mean) / 2
        # margin = scores.mean()
        # mask = scores < margin
        # mask.fill_diagonal_(False)
        # scores[mask] = 0
        for i in range(len(scores)):
            pos_mean = scores[i][i]
            neg_mean = (scores[i].sum() - pos_mean) / (scores.shape[1] - 1)
            margin = (pos_mean + neg_mean) / 2
            mask = scores[i] < margin
            mask[i] = False
            scores[i][mask] = 0
            # import IPython;IPython.embed(colors='linux');exit(1)
        return self.cross_entropy_loss(scores, labels)


class BCELoss(torch.nn.Module):

    def __init__(self, model):
        super(BCELoss, self).__init__()
        self.model = model
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.scale = 20.0

    def forward(self, sentence_features, labels):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_q = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        q_norm = torch.nn.functional.normalize(embeddings_q, p=2, dim=1)
        pos_norm = torch.nn.functional.normalize(embeddings_pos, p=2, dim=1)
        neg_norm = torch.nn.functional.normalize(embeddings_neg, p=2, dim=1)

        scores = torch.cat([(q_norm * pos_norm).sum(-1).unsqueeze(1),
                            (q_norm * neg_norm).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        scores = scores * self.scale 
        lebels = torch.tensor([0] * scores.shape[0], device=scores.device)

        return self.cross_entropy_loss(scores, labels)


class MixupMultipleNegativesRankingLoss(losses.MultipleNegativesRankingLoss):

    def __init__(self, model, similarity_fct):
        super().__init__(model, similarity_fct)

    def forward(self, sentence_features, labels):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        embeddings_a = reps[0]  # B * dim (queries)
        embeddings_b = torch.cat(reps[1:])  # B * dim (postives)
        # create mixup pseudo negatives
        embeddings_c = torch.cat([embeddings_b[1:], embeddings_b[0].unsqueeze(0)], dim=0)
        alphas = torch.rand(embeddings_b.shape[0]).unsqueeze(-1).to(embeddings_b.device)
        embeddings_c = alphas * embeddings_b + (1 - alphas) * embeddings_c
        embeddings_b = torch.cat([embeddings_b, embeddings_c])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(scores, labels)


class DCLLoss(torch.nn.Module):

    def __init__(self, model, similarity_fct, margin=0):
        super(DCLLoss, self).__init__()
        self.model = model
        self.similarity_fct=similarity_fct
        self.margin = margin
        self.scale = 20.0

    def dcl(self, scores):
        exp_scores = torch.exp(scores)
        positive = torch.diag(exp_scores)  # [B]
        
        diag_mask = torch.eye(scores.shape[0], scores.shape[1]).bool().to(scores.device)
        negative = torch.masked_select(exp_scores, ~diag_mask).view(scores.shape[0], scores.shape[1]-1)  # [B, 2B-1]
        return (-1) * torch.log(positive / negative.sum(dim=1)).mean()

    def forward(self, sentence_features, labels):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        return self.dcl(scores)
