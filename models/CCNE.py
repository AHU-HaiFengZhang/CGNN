import torch
import numpy as np
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
from models.metrics import top_k, compute_precision_k

class CCNE(torch.nn.Module):
    def __init__(self, s_input, t_input, output):#, not_share=False, is_bn=False, dropout=0):
        super().__init__()
        self.conv1 = GCNConv(s_input, 2 * output)
        self.conv2 = GCNConv(t_input, 2 * output)
        self.conv3 = GCNConv(2 * output, output)
        self.activation = nn.ReLU()
        # self.dropout = dropout
        # self.is_bn = is_bn
        # self.bn1 = BatchNorm(2 * output)
        # self.bn2 = BatchNorm(2 * output)
        # self.share = not not_share
        # self.conv4 = GCNConv(2 * output, output)

    def s_forward(self, x, edge_index):
        '''
        embeddings of source network g_s(V_s, E_s):
        x is node feature vectors of g_s, with shape [V_s, n_feats], V_s is the number of nodes,
            and n_feats is the dimension of features;
        edge_index is edges, with shape [2, 2 * E_s], E_s is the number of edges
        '''
        x = self.conv1(x, edge_index)
        # if self.is_bn:
        #     x = self.bn1(x)
        x = self.activation(x)
        # if self.dropout > 0:
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)

    def t_forward(self, x, edge_index):
        '''
        embeddings of target network g_t(V_t, E_t):
        x is node feature vectors of g_t, with shape [V_t, n_feats], V_t is the number of nodes,
            and n_feats is the dimension of features;
        edge_index is edges, with shape [2, 2 * E_t], E_t is the number of edges
        '''
        x = self.conv2(x, edge_index)
        # if self.is_bn:
        #     x = self.bn2(x)
        x = self.activation(x)
        # if self.dropout > 0:
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # if self.share:
        #     return self.conv3(x, edge_index)
        # return self.conv4(x, edge_index)
        return self.conv3(x, edge_index)

    def decoder(self, z, edge_index, sigmoid=True):
        '''
        reconstuct the original network by calculating the pairwise similarity of embedding vectors
        '''
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def single_recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        '''
        cross-entropy loss of positive and negative edges:
        z: the output of decoder
        pos_edge_index: index of positive edges
        neg_edge_index: index of negative edges
        '''
        EPS = 1e-15  # avoid zero when calculating logarithm

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean()  # loss of positive samples

        if neg_edge_index is None:
            neg_edge_index = torch_geometric.utils.negative_sampling(pos_edge_index, z.size(0))  # negative sampling
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean()  # loss of negative samples

        return pos_loss + neg_loss

    def intra_loss(self, zs, zt, s_pos_edge_index, t_pos_edge_index):
        '''
        intra-network loss to preserve intra-network structural features:
        zs: embeddings of source network g_s;
        zt: embeddings of target network g_t
        '''
        s_recon_loss = self.single_recon_loss(zs, s_pos_edge_index)
        t_recon_loss = self.single_recon_loss(zt, t_pos_edge_index)
        return s_recon_loss + t_recon_loss


def get_embedding(s_x, t_x, s_e, t_e, g_s, g_t, anchor, gt_mat, dim=128, lr=0.001, lamda=1, margin=0.8, neg=1, epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s_x = s_x.to(device)
    t_x = t_x.to(device)
    s_e = s_e.to(device)
    t_e = t_e.to(device)
    s_input = s_x.shape[1]
    t_input = t_x.shape[1]
    model = CCNE(s_input, t_input, dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cosine_loss=nn.CosineEmbeddingLoss(margin=margin)
    in_a, in_b, anchor_label = sample(anchor, g_s, g_t, neg=neg) # hard negative sampling

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # in_a, in_b, anchor_label = sample(anchor, g_s, g_t, neg=neg)
        zs = model.s_forward(s_x, s_e)
        zt = model.t_forward(t_x, t_e)

        intra_loss = model.intra_loss(zs, zt, s_e, t_e)
        anchor_label = anchor_label.view(-1).to(device)
        inter_loss = cosine_loss(zs[in_a], zt[in_b], anchor_label)
        loss = intra_loss + lamda * inter_loss
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            p10 = evaluate(zs, zt, gt_mat)
            print('Epoch: {:03d}, intra_loss: {:.8f}, inter_loss: {:.8f}, loss_train: {:.8f}, precision_10: {:.8f}'.format(epoch,\
                intra_loss, inter_loss, loss, p10))
    
    model.eval()
    s_embedding = model.s_forward(s_x, s_e)
    t_embedding = model.t_forward(t_x, t_e)
    s_embedding = s_embedding.detach().cpu()
    t_embedding = t_embedding.detach().cpu()
    return s_embedding, t_embedding

@torch.no_grad()
def evaluate(zs, zt, gt):
    '''
    calculate Precision@10 for evaluation
    '''
    z1 = zs.detach().cpu()
    z2 = zt.detach().cpu()
    S = cosine_similarity(z1, z2)
    pred_top_10 = top_k(S, 10)
    precision_10 = compute_precision_k(pred_top_10, gt)
    return precision_10

def sample(anchor_train, gs, gt, neg=1):
    '''
    sample non-anchors for each anchor
    '''
    triplet_neg = neg  # number of non-anchors for each anchor, when neg=1, there are two negtives for each anchor
    anchor_flag = 1
    anchor_train_len = anchor_train.shape[0]
    anchor_train_a_list = np.array(anchor_train.T[0])
    anchor_train_b_list = np.array(anchor_train.T[1])
    input_a = []
    input_b = []
    classifier_target = torch.empty(0)
    np.random.seed(5)
    index = 0
    while index < anchor_train_len:
        a = anchor_train_a_list[index]
        b = anchor_train_b_list[index]
        input_a.append(a)
        input_b.append(b)
        an_target = torch.ones(anchor_flag)
        classifier_target = torch.cat((classifier_target, an_target), dim=0)
        # an_negs_index = list(set(node_t) - {b}) # all nodes except anchor node
        an_negs_index = list(gt.neighbors(b)) # neighbors of each anchor node
        an_negs_index_sampled = list(np.random.choice(an_negs_index, triplet_neg, replace=True)) # randomly sample negatives
        an_as = triplet_neg * [a]
        input_a += an_as
        input_b += an_negs_index_sampled

        # an_negs_index1 = list(set(node_f) - {a})
        an_negs_index1 = list(gs.neighbors(a))
        an_negs_index_sampled1 = list(np.random.choice(an_negs_index1, triplet_neg, replace=True))
        an_as1 = triplet_neg * [b]
        input_b += an_as1
        input_a += an_negs_index_sampled1

        un_an_target = torch.zeros(triplet_neg * 2)
        classifier_target = torch.cat((classifier_target, un_an_target), dim=0)
        index += 1

    cosine_target = torch.unsqueeze(2 * classifier_target - 1, dim=1)  # labels are [1,-1,-1]
    # classifier_target = torch.unsqueeze(classifier_target, dim=1)  # labels are [1,0,0]

    # [ina, inb] is all anchors and sampled non-anchors, cosine_target is their labels
    ina = torch.LongTensor(input_a)
    inb = torch.LongTensor(input_b)

    return ina, inb, cosine_target