"""Entry point limpio del proyecto "Attention".

Entrena/evalua el Relational Full-Attention Graph Transformer (src/model.py) en
KGC con ranking FULL-FILTERED (todos los nodos candidatos), mismo protocolo que se
uso para reproducir KnowFormer/NBFNet. Sin nada de KnowFormer (model.py es nuevo).

Uso tipico (FB15k-237 inductivo v1):
  source env.sh && PY=$(which python)
  $PY train.py --data_path ./data/inductive/fb15k-237_v1 \
      --num_layer 6 --hidden_dim 32 --num_heads 8 \
      --batch_size 16 --test_batch_size 16 --max_epochs 20 \
      --learning_rate 5e-3 --weight_decay 1e-4 --seed 42
"""

import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model import (GraphTransformer, NBFNet, SparseGraphTransformer,
                       SparseExpanderGraphTransformer, SparseNBFValueTransformer)
from src.data import TransductiveKnowledgeGraph, InductiveKnowledgeGraph
from src.metric import MRMetric, MRRMetric, HitsMetric


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class KGDataModule(pl.LightningDataModule):
    def __init__(self, data_path, inductive, num_workers, batch_size, test_batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        cls = InductiveKnowledgeGraph if inductive else TransductiveKnowledgeGraph
        self.data = cls(data_path)
        self.num_relation = self.data.num_relation

    def train_dataloader(self):
        # Solo relaciones forward (par); el collate genera las reversas (impar).
        triplets = self.data.train_triplets.clone()
        triplets = triplets[triplets[:, 1] % 2 == 0]
        return DataLoader(triplets, shuffle=True, collate_fn=self.data.train_collate_fn,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data.valid_triplets.clone(), shuffle=False,
                          collate_fn=self.data.valid_collate_fn,
                          batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data.test_triplets.clone(), shuffle=False,
                          collate_fn=self.data.test_collate_fn,
                          batch_size=self.test_batch_size, num_workers=self.num_workers)


# ----------------------------------------------------------------------------
# Lightning module
# ----------------------------------------------------------------------------
class GTLightningModule(pl.LightningModule):
    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 learning_rate, weight_decay, model='rfat', aggregate='pna',
                 use_rwse=False, rwse_dim=16, use_lappe=False, lappe_dim=16,
                 use_source_rw=False, source_rw_dim=8,
                 use_rpb=False, rpb_hops=4, rpb_dim=16, exp_degree=4,
                 attn='softmax', filtered_ce=False):
        super().__init__()
        self.save_hyperparameters()
        rwse = dict(use_rwse=use_rwse, rwse_dim=rwse_dim,
                    use_lappe=use_lappe, lappe_dim=lappe_dim,
                    use_source_rw=use_source_rw, source_rw_dim=source_rw_dim)
        if model == 'nbfnet':
            self.model = NBFNet(num_relation, num_layer, hidden_dim, aggregate=aggregate)
        elif model == 'sparse':
            self.model = SparseGraphTransformer(num_relation, num_layer, hidden_dim,
                                                num_heads, drop, attn=attn, **rwse)
        elif model == 'sparse_exp':
            self.model = SparseExpanderGraphTransformer(num_relation, num_layer, hidden_dim,
                                                        num_heads, drop,
                                                        exp_degree=exp_degree, attn=attn,
                                                        **rwse)
        elif model == 'sparse_nbfv':
            self.model = SparseNBFValueTransformer(num_relation, num_layer, hidden_dim,
                                                   num_heads, drop, aggregate=aggregate, **rwse)
        else:
            self.model = GraphTransformer(num_relation, num_layer, hidden_dim,
                                          num_heads, drop, **rwse,
                                          use_rpb=use_rpb, rpb_hops=rpb_hops, rpb_dim=rpb_dim)
        self.mr = MRMetric()
        self.mrr = MRRMetric()
        self.h1 = HitsMetric(topk=1)
        self.h3 = HitsMetric(topk=3)
        self.h10 = HitsMetric(topk=10)

    def remove_edge(self, batched_data):
        """Quita del grafo la arista (h,r,t) de cada query y su reversa (t,rev_r,h)
        para que el modelo no lea la respuesta. Devuelve graph_mask booleano (E,)."""
        h, r, t = batched_data['h_index'], batched_data['r_index'], batched_data['t_index']
        graph = batched_data['graph']
        ei = graph.edge_index.to(h.device)
        rev_r = torch.where(r % 2 == 0, r + 1, r - 1)
        h_rm = torch.cat([h, t], 0)
        r_rm = torch.cat([r, rev_r], 0)
        t_rm = torch.cat([t, h], 0)
        encode = lambda a, b, c: c + (a + b * graph.num_nodes) * graph.num_nodes
        src_hash = encode(ei[:, 0], ei[:, 1], ei[:, 2])
        tgt_hash = encode(h_rm, r_rm, t_rm)
        mask = ~torch.isin(src_hash, tgt_hash)
        batched_data['graph_mask'] = mask
        return batched_data

    def training_step(self, batched_data, batch_idx):
        batched_data = self.remove_edge(batched_data)
        score = self.model(batched_data)                          # (B, N)
        # CE de grafo completo: el gold es t_index; las demas entidades son negativos.
        t_index = batched_data['t_index']
        if self.hparams.filtered_ce:
            # CE filtrado SOLO-train: las otras respuestas conocidas EN TRAIN de la
            # misma query (h, r) salen del denominador (multi-answer label handling).
            # 'filter_mask' del train_collate_fn viene de data.train_filters (solo
            # triplets de train) => NO hay fuga de val/test, a diferencia del pipeline
            # de Exphormer_Max que filtraba con train+val+test.
            filt = batched_data['filter_mask'].bool().clone()
            filt[torch.arange(filt.size(0), device=filt.device), t_index] = False
            score = score.masked_fill(filt, float('-inf'))
        loss = F.cross_entropy(score, t_index)
        self.log('train_loss', loss, prog_bar=True)
        self.log('mem_gb', torch.cuda.max_memory_allocated() / 1024**3, prog_bar=True)
        return loss

    def _eval_step(self, batched_data):
        score = self.model(batched_data)
        answer = score.gather(1, batched_data['t_index'].unsqueeze(1))
        filt = batched_data['filter_mask'].bool()
        ranks = torch.sum((score >= answer) & (~filt), dim=1) + 1
        for m in (self.mr, self.mrr, self.h1, self.h3, self.h10):
            m.update(ranks)

    def validation_step(self, batched_data, batch_idx):
        self._eval_step(batched_data)

    def test_step(self, batched_data, batch_idx):
        self._eval_step(batched_data)

    def _log_epoch(self, split):
        vals = {f'{split}_mr': self.mr.compute(), f'{split}_mrr': self.mrr.compute(),
                f'{split}_hits1': self.h1.compute(), f'{split}_hits3': self.h3.compute(),
                f'{split}_hits10': self.h10.compute()}
        for m in (self.mr, self.mrr, self.h1, self.h3, self.h10):
            m.reset()
        for k, v in vals.items():
            self.log(k, v, prog_bar=k.endswith(('mrr', 'hits10')), sync_dist=True)

    def validation_epoch_end(self, outputs):
        self._log_epoch('valid')

    def test_epoch_end(self, outputs):
        self._log_epoch('test')

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'norm']
        groups = [
            {'params': [p for n, p in self.named_parameters()
                        if any(d in n for d in no_decay) and p.requires_grad],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.named_parameters()
                        if not any(d in n for d in no_decay) and p.requires_grad],
             'weight_decay': self.hparams.weight_decay},
        ]
        opt = torch.optim.Adam(groups, lr=self.hparams.learning_rate)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [10, 15], 0.1)
        return [opt], [{'scheduler': sched, 'interval': 'epoch'}]


INDUCTIVE_DATASETS = {'fb15k-237', 'wn18rr', 'nell-995'}


def is_inductive(data_path):
    name = os.path.basename(os.path.normpath(data_path))
    return '_v' in name and name.split('_v')[0] in INDUCTIVE_DATASETS


def main():
    p = ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--model', type=str, default='rfat',
                   choices=['rfat', 'nbfnet', 'sparse', 'sparse_nbfv', 'sparse_exp'])
    p.add_argument('--exp_degree', type=int, default=4,
                   help='Grado del grafo expander (aristas por nodo) para --model sparse_exp.')
    p.add_argument('--attn', type=str, default='softmax',
                   choices=['softmax', 'sigmoid', 'degree', 'anchor', 'rel', 'qc'],
                   help='Agregacion/parametrizacion de la atencion sparse (sparse/sparse_exp). '
                        'softmax: segment-softmax (promedio, pierde conteo de caminos); '
                        'sigmoid: gates sin normalizar => suma ponderada que conserva conteo de '
                        'caminos y grado (opcion A); degree: softmax x log(1+grado_in); '
                        'anchor: interpola softmax con la media uniforme via lambda aprendido '
                        'por cabeza => regulariza hacia la agregacion fija (opcion C); '
                        'rel: ABLATION sin termino q.k => logit puramente relacional '
                        '(b[head,rel] + compatibilidad r_q x rel), transferible por construccion; '
                        'qc: paquete QC-Exphormer (Q anclado a x0, logit trilineal con relacion '
                        'vectorial, conditioning c_q en q/k/e, exp-suma sin normalizar + residual BF).')
    p.add_argument('--filtered_ce', action='store_true',
                   help='CE filtrado solo-train: excluye del denominador las otras respuestas '
                        'conocidas EN TRAIN de la query (h, r). Sin fuga de val/test. Aplica a '
                        'todos los modelos; para comparar hay que re-correr el baseline con el flag.')
    p.add_argument('--aggregate', type=str, default='pna', choices=['pna', 'sum'],
                   help='NBFNet y sparse_nbfv: funcion de agregacion del message passing.')
    p.add_argument('--use_rwse', action='store_true',
                   help='Sumar RWSE (random-walk structural encoding) a x^0. Solo modelos '
                        'de atencion (rfat/sparse/sparse_nbfv); nbfnet lo ignora.')
    p.add_argument('--rwse_dim', type=int, default=16,
                   help='Largo del random walk para RWSE (k=1..rwse_dim).')
    p.add_argument('--use_lappe', action='store_true',
                   help='Anade Laplacian positional encoding (autovectores del Laplaciano) a x^0.')
    p.add_argument('--use_source_rw', action='store_true',
                   help='Labeling condicionado a la query: landing probs de random walk '
                        'desde el head por nodo (rompe la simetria inicial del transformer).')
    p.add_argument('--source_rw_dim', type=int, default=8,
                   help='Largo del random walk para source_rw (k=1..source_rw_dim).')
    p.add_argument('--lappe_dim', type=int, default=16,
                   help='Numero de autovectores no triviales para LapPE.')
    p.add_argument('--use_rpb', action='store_true',
                   help='Relational Path Bias (opcion B): bias par de camino relacional '
                        'query-conditioned en el logit de atencion (solo modelo rfat).')
    p.add_argument('--rpb_hops', type=int, default=4,
                   help='Profundidad K del camino relacional para RPB (fuera del horizonte).')
    p.add_argument('--rpb_dim', type=int, default=16,
                   help='Dim de los embeddings de relacion para el score de compatibilidad RPB.')
    p.add_argument('--num_layer', type=int, default=6)
    p.add_argument('--hidden_dim', type=int, default=32)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--drop', type=float, default=0.1)
    p.add_argument('--learning_rate', type=float, default=5e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--test_batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--max_epochs', type=int, default=20)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--precision', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint_save_path', type=str, default='./experiments/gt')
    p.add_argument('--limit_train_batches', type=float, default=1.0,
                   help='PL Trainer limit_train_batches (smoke test: e.g. 20)')
    p.add_argument('--limit_val_batches', type=float, default=1.0,
                   help='PL Trainer limit_val_batches (smoke test: e.g. 0 to skip)')
    p.add_argument('--resume_from', type=str, default='',
                   help='Path a un checkpoint (last.ckpt) para reanudar entrenamiento')
    p.add_argument('--eval_only', action='store_true',
                   help='Saltar fit; solo correr test sobre --eval_ckpt (eval-only).')
    p.add_argument('--eval_ckpt', type=str, default='',
                   help='Path al checkpoint a testear cuando --eval_only.')
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)

    inductive = is_inductive(args.data_path)
    dm = KGDataModule(args.data_path, inductive, args.num_workers,
                      args.batch_size, args.test_batch_size)
    model = GTLightningModule(dm.num_relation, args.num_layer, args.hidden_dim,
                              args.num_heads, args.drop, args.learning_rate, args.weight_decay,
                              model=args.model, aggregate=args.aggregate,
                              use_rwse=args.use_rwse, rwse_dim=args.rwse_dim,
                              use_lappe=args.use_lappe, lappe_dim=args.lappe_dim,
                              use_source_rw=args.use_source_rw, source_rw_dim=args.source_rw_dim,
                              use_rpb=args.use_rpb, rpb_hops=args.rpb_hops, rpb_dim=args.rpb_dim,
                              exp_degree=args.exp_degree, attn=args.attn,
                              filtered_ce=args.filtered_ce)

    ckpt = ModelCheckpoint(dirpath=args.checkpoint_save_path, monitor='valid_mrr',
                           mode='max', save_top_k=1, every_n_epochs=1, verbose=True,
                           save_last=True)
    trainer = pl.Trainer(accelerator='gpu', devices=args.devices, precision=args.precision,
                         max_epochs=args.max_epochs, callbacks=[ckpt],
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         num_sanity_val_steps=0, check_val_every_n_epoch=1, logger=False)
    if args.eval_only:
        print(f"[eval_only] testeando ckpt: {args.eval_ckpt}")
        trainer.test(model, datamodule=dm, ckpt_path=args.eval_ckpt)
        return
    trainer.fit(model, datamodule=dm, ckpt_path=(args.resume_from or None))
    print(f"[best ckpt] {ckpt.best_model_path}  valid_mrr={ckpt.best_model_score}")
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path)


if __name__ == '__main__':
    main()
