import os
from copy import deepcopy
from functools import partial

import einops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from src.model import Knowformer
from src.data import TransductiveKnowledgeGraph, InductiveKnowledgeGraph
from src.metric import MRMetric, MRRMetric, HitsMetric


class TransductiveDataModule(pl.LightningDataModule):
    def __init__(self, data_path, num_workers, batch_size, test_batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        
        self.data_object = TransductiveKnowledgeGraph(self.data_path)
        self.num_relation = self.data_object.num_relation

    def train_dataloader(self):
        return DataLoader(self.data_object.train_triplets.clone()[self.data_object.train_triplets[:, 1]%2==0],
                          shuffle=True,
                          collate_fn=self.data_object.train_collate_fn,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        # Fix leak (2026-06-12): antes servia test_triplets+test_collate_fn => el
        # ModelCheckpoint (monitor valid_mrr) seleccionaba el mejor ckpt SOBRE EL TEST SET.
        # Ahora val = valid_triplets (model selection limpia; test queda solo en trainer.test).
        return DataLoader(self.data_object.valid_triplets.clone(),
                          shuffle=False,
                          collate_fn=self.data_object.valid_collate_fn,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.data_object.test_triplets.clone(), 
                          shuffle=False, 
                          collate_fn=self.data_object.test_collate_fn, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.num_workers)
        

class InductiveDataModule(pl.LightningDataModule):
    def __init__(self, data_path, num_workers, batch_size, test_batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        
        self.data_object = InductiveKnowledgeGraph(self.data_path)
        self.num_relation = self.data_object.num_relation

    def train_dataloader(self):
        return DataLoader(self.data_object.train_triplets.clone()[self.data_object.train_triplets[:, 1]%2==0],
                          shuffle=True,
                          collate_fn=self.data_object.train_collate_fn,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        # Fix leak (2026-06-12): antes servia test_triplets (test graph disjunto) => model
        # selection sobre el test set. Ahora val = valid_triplets sobre el TRAIN graph
        # (convencion CLAUDE.md: en inductivo val usa el train graph; test usa el disjunto).
        return DataLoader(self.data_object.valid_triplets.clone(),
                          shuffle=False,
                          collate_fn=self.data_object.valid_collate_fn,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.data_object.test_triplets.clone(), 
                          shuffle=False, 
                          collate_fn=self.data_object.test_collate_fn, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.num_workers)


class KnowformerLightningModule(pl.LightningModule):
    def __init__(self, num_relation, num_layer, num_qk_layer, num_v_layer, hidden_dim, num_heads, drop,
                 remove_all, loss_fn, num_negative_sample, optimizer, learning_rate, weight_decay, adversarial_temperature,
                 rala_kv=False, rala_phi='none', pairnorm_qk=0.0, qk_anchor_head=False,
                 qk_pe='none', qk_pe_walk_len=8, qk_pe_kind='rwpe', qk_pe_groups=8, qk_pe_lap_k=8,
                 qk_pe_readout=False,
                 cpa_k=0, cpa_compose='distmult', cpa_mode='relcond', cpa_random_pivots=False, cpa_v_layer=0,
                 cpa_center=False,
                 rerank_k=0, rerank_layers=2, rerank_heads=4, rerank_weight=1.0,
                 rerank_pair_bias=False, rerank_pair_shuffle=False):
        super().__init__()
        self.save_hyperparameters()

        self.model = Knowformer(self.hparams.num_relation, self.hparams.num_layer, self.hparams.num_qk_layer, self.hparams.num_v_layer,
                                self.hparams.hidden_dim, self.hparams.num_heads, self.hparams.drop,
                                rala_kv=self.hparams.rala_kv, rala_phi=self.hparams.rala_phi,
                                pairnorm_qk=self.hparams.pairnorm_qk, qk_anchor_head=self.hparams.qk_anchor_head,
                                qk_pe=self.hparams.qk_pe, qk_pe_walk_len=self.hparams.qk_pe_walk_len,
                                qk_pe_kind=self.hparams.qk_pe_kind, qk_pe_groups=self.hparams.qk_pe_groups,
                                qk_pe_lap_k=self.hparams.qk_pe_lap_k, qk_pe_readout=self.hparams.qk_pe_readout,
                                cpa_k=self.hparams.cpa_k, cpa_compose=self.hparams.cpa_compose,
                                cpa_mode=self.hparams.cpa_mode, cpa_random_pivots=self.hparams.cpa_random_pivots,
                                cpa_v_layer=self.hparams.cpa_v_layer, cpa_center=self.hparams.cpa_center,
                                rerank_k=self.hparams.rerank_k, rerank_layers=self.hparams.rerank_layers,
                                rerank_heads=self.hparams.rerank_heads,
                                rerank_pair_bias=self.hparams.rerank_pair_bias,
                                rerank_pair_shuffle=self.hparams.rerank_pair_shuffle)
        
        self.mr_metric_fn = MRMetric()
        self.mrr_metric_fn = MRRMetric()
        self.hits1_metric_fn = HitsMetric(topk=1)
        self.hits3_metric_fn = HitsMetric(topk=3)
        self.hits10_metric_fn = HitsMetric(topk=10)

    def remove_edge(self, batched_data):
        h_index, r_index, t_index, graph = (batched_data['h_index'], 
                                            batched_data['r_index'], 
                                            batched_data['t_index'],
                                            batched_data['graph'])
        h_index_remove = torch.cat([h_index, t_index], 0)
        r_index_remove = torch.cat([r_index, torch.where(r_index%2==0, r_index + 1, r_index - 1)], 0)
        t_index_remove = torch.cat([t_index, h_index], 0)
        
        if self.hparams.remove_all:
            # remove all edges between head and tail entities
            encode_fn = lambda x, y: x + y * graph.num_nodes
            source_hash = encode_fn(graph.edge_index[:, 0], graph.edge_index[:, 2])
            target_hash = encode_fn(h_index_remove, t_index_remove)
            mask = ~torch.isin(source_hash, target_hash)
        else:
            encode_fn = lambda x, y, z: z + (x + y * graph.num_nodes) * graph.num_nodes
            source_hash = encode_fn(graph.edge_index[:, 0], graph.edge_index[:, 1], graph.edge_index[:, 2])
            target_hash = encode_fn(h_index_remove, r_index_remove, t_index_remove)
            mask = ~torch.isin(source_hash, target_hash)

        batched_data.update({'graph_mask': mask})
        return batched_data
    
    def compute_loss(self, score, batched_data):
        positive_index = batched_data['positive_index']
        negative_index = batched_data['negative_index']
        all_index = torch.cat([positive_index, negative_index], 1)
        filter_mask = batched_data['filter_mask'].bool()
        if self.hparams.loss_fn == 'bce':
            logits = torch.gather(score, 1, all_index)
            target = torch.zeros_like(logits)
            target[:, 0] = 1
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            weights = torch.ones_like(logits)
            with torch.no_grad():
                weights[:, 1:] = F.softmax(logits[:, 1:]/self.hparams.adversarial_temperature, dim=-1)
            loss = (loss * weights).sum()
        else:
            loss = F.cross_entropy(score, positive_index.view(-1))
            
        return loss
    
    def training_step(self, batched_data):
        batch_size = batched_data['h_index'].size(0)
        num_nodes = batched_data['graph'].num_nodes
        
        batched_data['positive_index'] = batched_data['t_index'].unsqueeze(1)
        batched_data['negative_index'] = negative_sample(batched_data['filter_mask'].bool(), 
                                                         min(num_nodes, 2**self.hparams.num_negative_sample))
        
        out = self.model(self.remove_edge(batched_data))
        if self.hparams.rerank_k > 0:
            # A1-v0: loss conjunta = CE full sobre el score base (entrena el modelo base) +
            # CE restringida a top-K∪{gold} sobre los scores reordenados (entrena el reranker,
            # el gold debe quedar primero). El reranker arranca como identidad (delta init-0).
            score, rr_logits, rr_target = out
            base_loss = self.compute_loss(score, batched_data)
            rerank_loss = F.cross_entropy(rr_logits, rr_target)
            loss = base_loss + self.hparams.rerank_weight * rerank_loss
            self.log('rerank_loss', rerank_loss, prog_bar=True)
        else:
            loss = self.compute_loss(out, batched_data)

        self.log('memory', torch.cuda.max_memory_allocated()/(1024**3), prog_bar=True)

        return loss


    def validation_step(self, batched_data, batch_idx):
        score = self.model(batched_data)
        
        answer_score = score.gather(1, batched_data['t_index'].unsqueeze(1))
        filter_mask = batched_data['filter_mask'].bool()
        ranks = torch.sum((score >= answer_score) & (~filter_mask), dim=1) + 1

        self.mr_metric_fn.update(ranks)
        self.mrr_metric_fn.update(ranks)
        self.hits1_metric_fn.update(ranks)
        self.hits3_metric_fn.update(ranks)
        self.hits10_metric_fn.update(ranks)

    def validation_epoch_end(self, outputs):
        mr = self.mr_metric_fn.compute()
        mrr = self.mrr_metric_fn.compute()
        hits1 = self.hits1_metric_fn.compute()
        hits3 = self.hits3_metric_fn.compute()
        hits10 = self.hits10_metric_fn.compute()

        self.mr_metric_fn.reset()
        self.mrr_metric_fn.reset()
        self.hits1_metric_fn.reset()
        self.hits3_metric_fn.reset()
        self.hits10_metric_fn.reset()

        self.log('valid_mr', mr, prog_bar=True, sync_dist=True)
        self.log('valid_mrr', mrr, prog_bar=True, sync_dist=True)
        self.log('valid_hits1', hits1, prog_bar=True, sync_dist=True)
        self.log('valid_hits3', hits3, prog_bar=False, sync_dist=True)
        self.log('valid_hits10', hits10, prog_bar=True, sync_dist=True)
        
    def test_step(self, batched_data, batch_idx):
        score = self.model(batched_data)
        answer_score = score.gather(1, batched_data['t_index'].unsqueeze(1))
        filter_mask = batched_data['filter_mask'].bool()
        ranks = torch.sum((score >= answer_score) & (~filter_mask), dim=1) + 1

        # Autopsia de errores (PLAN_MRR.md §5 / Stage 0): vuelca el rank por query junto
        # con (h, r, t) para estratificar el reciprocal rank por d(h,t)/grado/frecuencia.
        # Gated por env var KNOWFORMER_RANK_AUTOPSY (default off => baseline intacto). El
        # grafo de test se reconstruye en el script de autopsia desde el dataset.
        if os.environ.get('KNOWFORMER_RANK_AUTOPSY'):
            if not hasattr(self, '_autopsy_buf'):
                self._autopsy_buf = []
            self._autopsy_buf.append({
                'h': batched_data['h_index'].detach().cpu(),
                'r': batched_data['r_index'].detach().cpu(),
                't': batched_data['t_index'].detach().cpu(),
                'rank': ranks.detach().cpu(),
            })

        # Gate §3 (PLAN_FASE1_RCT): dump por query de los estados de nodo de los candidatos
        # (top-K negativos puros por score base + gold) = feature (i) del gate_relpath. El gold
        # esta en filter_mask => no contamina los negativos. Gated por KNOWFORMER_GATE_DUMP
        # (ruta de salida; default off => baseline intacto). Solo guarda queries con gold
        # rank<=Kd (confundibles potenciales); las demas no pueden estar en ningun top-K<=Kd.
        if os.environ.get('KNOWFORMER_GATE_DUMP') and hasattr(self.model, '_gate_x_state'):
            Kd = min(128, score.size(1) - 1)
            x_state = self.model._gate_x_state                        # (b, N, d)
            t_index = batched_data['t_index']                         # (b,)
            d = x_state.size(-1)
            masked = score.masked_fill(filter_mask, float('-inf'))    # gold ya en filter_mask
            neg_ids = masked.topk(Kd, dim=1).indices                  # (b, Kd) negativos puros
            neg_states = x_state.gather(1, neg_ids.unsqueeze(-1).expand(-1, -1, d))     # (b,Kd,d)
            gold_states = x_state.gather(1, t_index.view(-1, 1, 1).expand(-1, 1, d)).squeeze(1)  # (b,d)
            keep = ranks <= Kd
            if not hasattr(self, '_gate_buf'):
                self._gate_buf = []
            self._gate_buf.append({
                'h': batched_data['h_index'][keep].detach().cpu(),
                'r': batched_data['r_index'][keep].detach().cpu(),
                't': t_index[keep].detach().cpu(),
                'rank': ranks[keep].detach().cpu(),
                'neg_ids': neg_ids[keep].detach().cpu(),
                'neg_states': neg_states[keep].detach().cpu().float(),
                'gold_state': gold_states[keep].detach().cpu().float(),
            })

        self.mr_metric_fn.update(ranks)
        self.mrr_metric_fn.update(ranks)
        self.hits1_metric_fn.update(ranks)
        self.hits3_metric_fn.update(ranks)
        self.hits10_metric_fn.update(ranks)

    def test_epoch_end(self, outputs):
        mr = self.mr_metric_fn.compute()
        mrr = self.mrr_metric_fn.compute()
        hits1 = self.hits1_metric_fn.compute()
        hits3 = self.hits3_metric_fn.compute()
        hits10 = self.hits10_metric_fn.compute()

        self.mr_metric_fn.reset()
        self.mrr_metric_fn.reset()
        self.hits1_metric_fn.reset()
        self.hits3_metric_fn.reset()
        self.hits10_metric_fn.reset()

        # Autopsia: persistir el dump de ranks por query (ver test_step). El valor de la env
        # var es la ruta de salida (default 'autopsy_ranks.pt'). Concatena todos los batches.
        dump_path = os.environ.get('KNOWFORMER_RANK_AUTOPSY')
        if dump_path and getattr(self, '_autopsy_buf', None):
            buf = self._autopsy_buf
            out = {k: torch.cat([b[k] for b in buf]) for k in ('h', 'r', 't', 'rank')}
            if dump_path == '1':
                dump_path = 'autopsy_ranks.pt'
            torch.save(out, dump_path)
            print(f"[autopsy] volcados {out['rank'].numel()} ranks por query -> {dump_path}")
            self._autopsy_buf = []

        # Gate §3: persistir el dump de candidatos (ver test_step).
        gate_path = os.environ.get('KNOWFORMER_GATE_DUMP')
        if gate_path and getattr(self, '_gate_buf', None):
            buf = self._gate_buf
            keys = ('h', 'r', 't', 'rank', 'neg_ids', 'neg_states', 'gold_state')
            out = {k: torch.cat([b[k] for b in buf]) for k in keys}
            if gate_path == '1':
                gate_path = 'gate_dump.pt'
            torch.save(out, gate_path)
            print(f"[gate] volcadas {out['rank'].numel()} queries (gold rank<=Kd) -> {gate_path}")
            self._gate_buf = []

        self.log('test_mr', mr, prog_bar=False, sync_dist=True)
        self.log('test_mrr', mrr, prog_bar=True, sync_dist=True)
        self.log('test_hits1', hits1, prog_bar=False, sync_dist=True)
        self.log('test_hits3', hits3, prog_bar=False, sync_dist=True)
        self.log('test_hits10', hits10, prog_bar=False, sync_dist=True)
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_optimizer_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if any([d in n for d in no_decay]) and p.requires_grad],
                'weight_decay': 0.0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not any([d in n for d in no_decay]) and p.requires_grad],
                'weight_decay': self.hparams.weight_decay
            }
        ]
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            grouped_optimizer_parameters,
            lr=self.hparams.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], 0.1)
        scheduler = {
            'scheduler': scheduler, 
            'interval': 'epoch', 
            'frequency': 1
        }

        return [optimizer], [scheduler]
    
    
def add_data_specific_args(parent_args):
    parser = parent_args.add_argument_group('Data')
    parser.add_argument('--data_path', type=str, help="the path to dataset directory")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    return parent_args

def add_model_specific_args(parent_args):
    parser = parent_args.add_argument_group('Model')
    parser.add_argument('--num_layer', type=int, default=3, help="number of layers")
    parser.add_argument('--num_qk_layer', type=int, default=2, help="number of layers to get qk")
    parser.add_argument('--num_v_layer', type=int, default=2, help="number of layers to get v")
    parser.add_argument('--hidden_dim', type=int, default=64, help="the size of feature")
    parser.add_argument('--num_heads', type=int, default=4, help="number of heads")
    parser.add_argument('--drop', type=float, default=.1, help="dropout rate")
    parser.add_argument('--remove_all', action='store_true', help="whether or not remove all one hop edges")
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['bce', 'ce'], help="loss function")
    parser.add_argument('--num_negative_sample', type=int, default=7, help="number of negative examples")
    parser.add_argument('--optimizer', type=str, default='Adam', help="the optimizer")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="the initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="the weight decay of optimizer")
    parser.add_argument('--adversarial_temperature', type=float, default=1.0)
    # RALA (breaking_low_rank.tex). Optional augmentations of the linear attention, off by default.
    parser.add_argument('--rala_kv', action='store_true',
                        help="RALA mod 1: context-aware KV buffer reweighting (alpha_j). Parameter-free.")
    parser.add_argument('--rala_phi', type=str, default='none', choices=['none', 'x', 'v'],
                        help="RALA mod 2: output Hadamard modulation phi(X)*Y. x=accumulated layer input, v=V-RMPNN stream.")
    # Anti-over-smoothing: PairNorm tras cada capa del QK-RMPNN (des-smoothea el feature map K
    # que alimenta la atencion). 0.0 = off (baseline). Tipico s en [0.1, 2.0].
    parser.add_argument('--pairnorm_qk', type=float, default=0.0,
                        help="PairNorm scale on the QK-RMPNN stream (anti-over-smoothing). 0 = off.")
    # Fase A (PLAN_SOLUCION.md): labeling trick estilo NBFNet en el QK-stream.
    parser.add_argument('--qk_anchor_head', action='store_true',
                        help="Inyecta el one-hot del head al input del QK-stream (espejo del V-stream). off = baseline (ruido gaussiano).")
    # Fase C (PLAN_SOLUCION.md): RWPE desde el head como PE estructural global del QK-stream.
    parser.add_argument('--qk_pe', type=str, default='none', choices=['none', 'input', 'post'],
                        help="RWPE desde el head sumado al QK-stream. input=al input del RMPNN; post=tras el RMPNN (antes de fc_to_qk). none=off.")
    parser.add_argument('--qk_pe_walk_len', type=int, default=8,
                        help="Longitud L del random walk para el RWPE/noise (canales por componente).")
    # Structural Query-PE (PLAN_STRUCTURAL_PE.md, L1-L3). Componentes ENSAMBLABLES del PE.
    parser.add_argument('--qk_pe_kind', type=str, default='rwpe',
                        help="Componentes del PE (coma-separados): rwpe,rel,deg,lap,noise. "
                             "rwpe=Fase C (head, rel-agnostic); rel=L1 (head por grupo de relacion); "
                             "deg=L2 grado/relacion intrinseco; lap=L2 Laplacian eigvecs intrinseco; "
                             "noise=control de capacidad. Orden canonico interno fijo.")
    parser.add_argument('--qk_pe_groups', type=int, default=8,
                        help="Numero de grupos de relacion (rel % G) para rel/deg.")
    parser.add_argument('--qk_pe_lap_k', type=int, default=8,
                        help="Numero de eigenvectores del Laplaciano para el componente lap.")
    parser.add_argument('--qk_pe_readout', action='store_true',
                        help="L3: inyecta el PE del candidato tambien al readout (mlp_out), no solo a la atencion.")
    # CPA (PLAN_MRR.md P1): Compositional Pivot Attention. Atencion de composicion sobre k pivotes
    # query-conditioned (x_{h,u} ⊙ x_{u,t}, 2L hops). Aditiva, gate init-0 => baseline anidado.
    parser.add_argument('--cpa_k', type=int, default=0,
                        help="Numero de pivotes de CPA. 0 = off (baseline). Tipico {4,8,16}.")
    parser.add_argument('--cpa_compose', type=str, default='distmult', choices=['distmult', 'mlp'],
                        help="Composicion x_{h,u} ⊙ x_{u,t}: distmult (Hadamard) o mlp (MLP sobre concat).")
    parser.add_argument('--cpa_mode', type=str, default='relcond', choices=['relcond', 'agnostic'],
                        help="2a corrida: relcond (condicionada a la relacion de la query) o agnostic (token aprendido, rel-agnostico).")
    parser.add_argument('--cpa_random_pivots', action='store_true',
                        help="Control de capacidad: pivotes aleatorios en vez de top-k (separa 'mas computo' de 'composicion dirigida').")
    parser.add_argument('--cpa_v_layer', type=int, default=0,
                        help="Capas internas de la 2a corrida del V-RMPNN. 0 = num_v_layer (mitigacion (b): usar menos).")
    parser.add_argument('--cpa_center', action='store_true',
                        help="Fix Resultado 12: centra x_{u,v} sobre los pivotes (resta el backbone comun, aisla el residuo pivote-especifico). Ataca la cuasi-invariancia al ancla (cos 0.992).")
    # A1-v0 (PLAN A1): candidate-set attention. Reordena el top-K por score base con atencion
    # softmax sobre los candidatos (pointwise->listwise). 0 = off (baseline). delta init-0 => anidado.
    parser.add_argument('--rerank_k', type=int, default=0,
                        help="A1: nº de candidatos top-K a reordenar. 0 = off (baseline). Tipico {32,64,128}.")
    parser.add_argument('--rerank_layers', type=int, default=2,
                        help="A1: nº de capas de atencion del reranker sobre los candidatos.")
    parser.add_argument('--rerank_heads', type=int, default=4,
                        help="A1: nº de cabezas de atencion del reranker.")
    parser.add_argument('--rerank_weight', type=float, default=1.0,
                        help="A1: peso de la CE restringida (reranker) en la loss conjunta.")
    parser.add_argument('--rerank_pair_bias', action='store_true',
                        help="A1 paso 2 (el claim): bias por par estructural (adyacencia + vecinos comunes) "
                             "en la atencion entre candidatos. off = v0 minima (sin bias, R14 negativa).")
    parser.add_argument('--rerank_pair_shuffle', action='store_true',
                        help="A1 control de capacidad: baraja las features de par (estructura desalineada "
                             "de los tokens) => aisla si gana la SEÑAL de par o solo los params.")
    return parent_args

def positive_sample(mask):
    p = torch.ones_like(mask).float()
    p = p * mask
    pos = torch.multinomial(p, num_samples=1)
    return pos

def negative_sample(mask, num_negative_sample):
    p = torch.ones_like(mask).float()
    p = p * (~mask)
    neg = torch.multinomial(p, num_samples=num_negative_sample, replacement=True)
    return neg

# The below functions are from huggingface transformers

def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step > num_training_steps:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # as LambdaLR multiplies by lr_init

def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        _get_polynomial_decay_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        power=power,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_constant_schedule(optimizer, last_epoch=-1):
    return LambdaLR(optimizer, lambda x: 1.0, last_epoch=last_epoch)
