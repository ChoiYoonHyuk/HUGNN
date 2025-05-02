import numpy as np, argparse, random, math
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import to_dense_adj, dropout_adj
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
lr = 5e-3

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args       = parser.parse_args()
data_id    = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora',      name='Cora')
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer',  name='Citeseer')
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed',    name='Pubmed')
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel',  name='squirrel')
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas',  name='Texas')
else:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')

data        = dataset[0].to(device)
edge_index  = data.edge_index
num_class   = dataset.num_classes
d_in        = dataset.num_node_features
d_hidden    = 64
heads       = 4

class UncertaintyFn(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.linear1 = nn.Linear(1, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, var_tensor):
        if var_tensor.dim() == 1:
            var_tensor = var_tensor.unsqueeze(-1)
        h = F.relu(self.linear1(var_tensor))
        out = torch.sigmoid(self.linear2(h))
        return out.squeeze(-1)

# Local message‑passing
class UGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6):
        super().__init__(aggr='add', node_dim=0)
        self.heads  = heads
        self.out_c  = out_channels // heads
        self.lin    = nn.Linear(in_channels, heads * self.out_c, bias=False)
        self.drop_path = 0.1
        if in_channels != heads * self.out_c:
            self.res_lin = nn.Linear(in_channels, heads * self.out_c, bias=False)
        else:
            self.res_lin = nn.Identity()
        self.att    = nn.Parameter(torch.Tensor(1, heads, 2*self.out_c))
        self.unc_fn = UncertaintyFn(hidden_dim=out_channels//2)
        self.dropout= dropout
        self.layer_norm = nn.LayerNorm(heads * self.out_c)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if isinstance(self.res_lin, nn.Linear):
            nn.init.xavier_uniform_(self.res_lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, u):
        h_in   = x
        x_proj = self.lin(x).view(-1, self.heads, self.out_c)
        out, u_new = self.propagate(edge_index, x=x_proj, u=u)
        # concat heads
        h_out = out.view(-1, self.heads * self.out_c)
        h_res = self.res_lin(h_in) + h_out
        if self.training and torch.rand(1, device=x.device) < self.drop_path:
            h_res = self.layer_norm(self.res_lin(h_in))     # skip message
        else:
            h_res = self.layer_norm(h_res)
        return h_res, u_new

    # Message-passing
    def message(self, x_i, x_j, index, u_j):
        a_input   = torch.cat([x_i, x_j], dim=-1)
        alpha     = (a_input * self.att).sum(-1)
        alpha     = F.leaky_relu(alpha, 0.2)
        alpha     = softmax(alpha, index)
        # uncertainty‑based weight
        rho       = torch.exp(-u_j).unsqueeze(-1)
        rho       = softmax(rho, index).repeat(1, self.heads)
        m_coeff   = 0.5 * (alpha + rho)
        m_coeff   = F.dropout(m_coeff, p=0.6, training=self.training)
        return m_coeff.unsqueeze(-1) * x_j

    # Update node embedding and uncertainty
    def update(self, aggr_out, x, edge_index):
        out = aggr_out.view(-1, self.heads * self.out_c)
        row, col = edge_index
        diff = (x[row] - x[col]).pow(2).sum(-1)
        var  = scatter_mean(diff, row, dim=0, dim_size=x.size(0))
        var  = var.mean(dim=1)
        new_u = self.unc_fn(var)
        return out, new_u

# HU‑GNN model
class HUGNN(nn.Module):
    def __init__(self, d_in, d_hidden, num_class, heads=4, L=2):
        super().__init__()
        self.L  = L
        # Local UGATConv
        self.local_convs = nn.ModuleList()
        self.local_convs.append(UGATConv(d_in, d_hidden, heads))
        self.two_hop_lin = nn.Linear(d_in, d_hidden, bias=False)
        for _ in range(1, L):
            self.local_convs.append(UGATConv(d_hidden, d_hidden, heads))
        # Community assignment W_M
        self.assign_W  = nn.Linear(d_hidden, num_class, bias=False)
        # Community & global transform
        self.W_c       = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_g       = nn.Linear(d_hidden, d_hidden, bias=False)
        # Classifier
        self.W_f       = nn.Linear(d_hidden, num_class)
        # Uncertainty modules
        self.unc_fn_node  = UncertaintyFn(d_hidden)
        self.unc_fn_comm  = UncertaintyFn(d_hidden)
        self.unc_fn_glob  = UncertaintyFn(d_hidden)
        self.gate = nn.Sequential(
            nn.Linear(d_hidden*3 + 3, d_hidden//2),
            nn.ReLU(),
            nn.Linear(d_hidden//2, 3),
        )

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Local message‑passing
        u = x.new_full((x.size(0),), 0.5)
        for conv in self.local_convs:
            x, u = conv(x, edge_index, u)

        A2x = torch.matmul(to_dense_adj(edge_index)[0], torch.matmul(to_dense_adj(edge_index)[0], data.x))
        A2x = self.two_hop_lin(A2x)
        x   = x + 0.2 * A2x

        # Community pooling
        assign_logits = self.assign_W(x)
        S_soft        = F.softmax(assign_logits, dim=1)
        C_emb   = torch.matmul(S_soft.t(), self.W_c(x))
        C_size  = S_soft.sum(0, keepdim=True).t() + 1e-9
        C_emb   = C_emb / C_size
        # Uncertainty
        diff_C  = (self.W_c(x).unsqueeze(1) - C_emb.unsqueeze(0))**2
        var_C   = (S_soft.unsqueeze(-1) * diff_C).sum(0) / C_size
        u_C     = self.unc_fn_comm(var_C.sum(-1))

        # Global integration
        H_g     = self.W_g(C_emb).mean(0, keepdim=True)
        u_g     = self.unc_fn_glob(((C_emb - H_g)**2).mean(-1))
        u_g     = u_g.mean()

        # Find λ
        # local‑community‑global attention + uncertainty
        def compute_lambda(q, k, u_k):
            alpha = (q * k).sum(-1, keepdim=True)
            return torch.exp(alpha) , torch.exp(-u_k).unsqueeze(-1)

        # local λ_i
        lam_i_num  = 0.5*(softmax(compute_lambda(x, x, u)[0], torch.arange(x.size(0), device=x.device))
                          + softmax(compute_lambda(x, x, u)[1], torch.arange(x.size(0), device=x.device)))
        # community λ_c
        x2c        = torch.matmul(S_soft, C_emb)
        lam_c_num  = 0.5*(compute_lambda(x, x2c, u_C[S_soft.argmax(-1)])[0]
                          + compute_lambda(x, x2c, u_C[S_soft.argmax(-1)])[1])
        # global λ_g (broadcast)
        lam_g_num  = 0.5*(torch.exp((x*H_g).sum(-1, keepdim=True))
                          + torch.exp(-u_g).expand_as(lam_i_num))

        lam_sum    = lam_i_num + lam_c_num + lam_g_num
        lam_i, lam_c, lam_g = lam_i_num/lam_sum, lam_c_num/lam_sum, lam_g_num/lam_sum

        # Representation & logits
        x_c = x2c
        x_g = H_g.expand(x.size(0), -1)
        u_c_node = u_C[S_soft.argmax(-1)].unsqueeze(-1)
        u_g_node = u_g.expand(x.size(0),1)
        gate_in = torch.cat([x, x_c, x_g, u.unsqueeze(-1), u_c_node, u_g_node], dim=1)
        lam_raw = self.gate(gate_in)
        lam = F.softmax(lam_raw, dim=1)
        lam_i, lam_c, lam_g = lam[:,0:1], lam[:,1:2], lam[:,2:3]

        h_final = lam_i * x + lam_c * x_c + lam_g * x_g
        logits  = self.W_f(h_final)
        return F.log_softmax(logits, dim=1)

# Training loop
model  = HUGNN(d_in, d_hidden, num_class, heads).to(device)
optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=400, gamma=0.5)

best_val, best_test = 0, 0
for epoch in tqdm(range(300)):
    model.train()
    optim.zero_grad()
    out = model(data.x, edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optim.step()
    scheduler.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred  = model(data.x, edge_index).argmax(dim=1)
        val_acc  = (pred[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc
            print(f'epoch {epoch:3d} | best‑val {best_val*100:.2f}% | best‑test {best_test*100:.2f}%')

print('Validation / Test acc.  :', best_val*100, best_test*100)