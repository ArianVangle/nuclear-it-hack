"""
Графовая нейросеть для классификации заглушек (D-MPNN).

Зачем она, когда отпечатки уже перебраны. Отпечаток — это хеш подструктур
фиксированного радиуса: молекула превращается в мешок фрагментов, и порядок
их соединения теряется. Сеть с передачей сообщений работает с самим графом:
атомы обмениваются состояниями по связям, и представление молекулы строится
под задачу, а не задаётся заранее.

Это единственное принципиально другое представление, оставшееся
неиспользованным. Предобученные веса из рабочей среды недоступны, поэтому
сеть учится с нуля — на 320 молекулах она вряд ли обыграет отпечатки
в одиночку, но нас интересует другое: ошибается ли она в других местах.
Именно разнообразие формулировок давало прирост (смесь с моделью
на интервалах — минус 2,2% на лидерборде), а не сила отдельной модели.

Сообщения идут по направленным связям, как в chemprop: состояние ребра
обновляется из состояний входящих рёбер, исключая обратное к себе, —
это не даёт сообщению немедленно вернуться туда, откуда пришло.
"""

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 34]
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
       Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
       Chem.HybridizationType.SP3D2]
BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
         Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

ATOM_DIM = len(ELEMENTS) + 1 + 6 + 5 + len(HYB) + 1 + 5 + 1
BOND_DIM = len(BONDS) + 1 + 1 + 1


def _onehot(v, options):
    out = [0.0] * (len(options) + 1)
    out[options.index(v) if v in options else len(options)] = 1.0
    return out


def atom_features(a):
    return (_onehot(a.GetAtomicNum(), ELEMENTS)
            + _onehot(a.GetDegree(), [0, 1, 2, 3, 4])
            + _onehot(a.GetFormalCharge(), [-2, -1, 0, 1, 2])[:5]
            + _onehot(a.GetHybridization(), HYB)[:len(HYB) + 1]
            + _onehot(a.GetTotalNumHs(), [0, 1, 2, 3])[:5]
            + [1.0 if a.GetIsAromatic() else 0.0])


def bond_features(b):
    return (_onehot(b.GetBondType(), BONDS)
            + [1.0 if b.GetIsConjugated() else 0.0,
               1.0 if b.IsInRing() else 0.0])


def mol_to_graph(smi):
    """Молекула -> (атомы, рёбра, откуда, куда, обратное ребро)."""
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        return (np.zeros((1, ATOM_DIM), dtype=np.float32),
                np.zeros((0, BOND_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    af = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    src, dst, bf = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        bf += [f, f]
    bf = (np.array(bf, dtype=np.float32) if bf
          else np.zeros((0, BOND_DIM), dtype=np.float32))
    rev = np.array([k ^ 1 for k in range(len(src))], dtype=np.int64)
    return af, bf, np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), rev


class Batch:
    """Несколько молекул как один граф со смещением индексов."""

    def __init__(self, graphs):
        a, b, s, d, r, owner = [], [], [], [], [], []
        off_a = off_b = 0
        for gi, (af, bf, src, dst, rev) in enumerate(graphs):
            a.append(af)
            b.append(bf)
            s.append(src + off_a)
            d.append(dst + off_a)
            r.append(rev + off_b)
            owner.append(np.full(len(af), gi, dtype=np.int64))
            off_a += len(af)
            off_b += len(bf)
        self.atoms = torch.from_numpy(np.concatenate(a))
        self.bonds = torch.from_numpy(np.concatenate(b)) if off_b else torch.zeros((0, BOND_DIM))
        self.src = torch.from_numpy(np.concatenate(s)) if off_b else torch.zeros(0, dtype=torch.long)
        self.dst = torch.from_numpy(np.concatenate(d)) if off_b else torch.zeros(0, dtype=torch.long)
        self.rev = torch.from_numpy(np.concatenate(r)) if off_b else torch.zeros(0, dtype=torch.long)
        self.owner = torch.from_numpy(np.concatenate(owner))
        self.n_mols = len(graphs)
        self.n_atoms = off_a


class DMPNN(nn.Module):
    def __init__(self, hidden=128, steps=3, extra_dim=0, dropout=0.1):
        super().__init__()
        self.steps = steps
        self.w_in = nn.Linear(ATOM_DIM + BOND_DIM, hidden)
        self.w_msg = nn.Linear(hidden, hidden)
        self.w_out = nn.Linear(ATOM_DIM + hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + extra_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, batch, extra=None):
        if batch.bonds.shape[0] == 0:
            pooled = torch.zeros(batch.n_mols, self.w_out.out_features)
        else:
            h0 = torch.relu(self.w_in(
                torch.cat([batch.atoms[batch.src], batch.bonds], dim=1)))
            h = h0
            for _ in range(self.steps):
                # сумма входящих сообщений в начальный атом ребра
                agg = torch.zeros_like(h).index_add_(0, batch.dst, h)
                msg = agg[batch.src] - h[batch.rev]   # исключаем обратное ребро
                h = torch.relu(h0 + self.w_msg(self.drop(msg)))
            node = torch.zeros(batch.n_atoms, h.shape[1]).index_add_(0, batch.dst, h)
            node = torch.relu(self.w_out(torch.cat([batch.atoms, node], dim=1)))
            pooled = torch.zeros(batch.n_mols, node.shape[1]).index_add_(0, batch.owner, node)
            counts = torch.zeros(batch.n_mols).index_add_(
                0, batch.owner, torch.ones(batch.n_atoms)).clamp(min=1)
            pooled = pooled / counts.unsqueeze(1)
        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=1)
        return self.head(pooled).squeeze(1)


def train_gnn(smiles, y, extra=None, epochs=60, hidden=128, steps=3,
              lr=1e-3, batch_size=32, seed=0, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    graphs = [mol_to_graph(s) for s in smiles]
    y_t = torch.tensor(np.asarray(y, dtype=np.float32))
    ex = None if extra is None else torch.tensor(np.asarray(extra, dtype=np.float32))
    model = DMPNN(hidden, steps, extra_dim=0 if extra is None else extra.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = float(y_t.sum())
    w = torch.tensor((len(y_t) - pos) / max(pos, 1.0))
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)

    n = len(graphs)
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            b = Batch([graphs[j] for j in idx])
            e = None if ex is None else ex[idx]
            opt.zero_grad()
            loss = lossf(model(b, e), y_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if verbose and (ep + 1) % 20 == 0:
            print(f"    эпоха {ep + 1}: loss {tot / n:.4f}", flush=True)
    return model, graphs


@torch.no_grad()
def predict_gnn(model, smiles, extra=None, batch_size=64):
    model.eval()
    graphs = [mol_to_graph(s) for s in smiles]
    ex = None if extra is None else torch.tensor(np.asarray(extra, dtype=np.float32))
    out = []
    for i in range(0, len(graphs), batch_size):
        b = Batch(graphs[i:i + batch_size])
        e = None if ex is None else ex[i:i + batch_size]
        out.append(torch.sigmoid(model(b, e)).numpy())
    return np.concatenate(out) if out else np.zeros(0)
