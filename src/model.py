"""Relational Full-Attention Graph Transformer (RFAT) para KGC inductivo.

Proyecto "Attention" (2026-06): partir desde cero para responder una pregunta de
sanidad antes de invertir en atencion sparse/lineal:

    Un Graph Transformer con atencion FULL (densa, all-pairs O(N^2)) -- que es el
    techo de expresividad de cualquier variante sparse -- ¿supera a NBFNet en
    FB15k-237 inductivo v1? Si la version densa no gana, la sparse no tiene caso.

Diseno (justificado en transformer_vs_nbfnet.tex):
  * NO hay embeddings de entidad ni positional encoding de nodo => inductivo puro
    (el grafo de test tiene entidades disjuntas; nada que transferir salvo relaciones).
  * Labeling trick (NBFNet): x^0_v = emb(r_q) si v == head, si no 0. Ancla la
    propagacion a la fuente y condiciona a la relacion de la query.
  * Cada capa: atencion multi-head DENSA all-pairs (reach global en una capa) con
       (i)  bias escalar relacional b[head, rel] en pares conectados por arista
            (estilo Graphormer: le dice a la atencion quien es vecino y por que relacion);
       (ii) correccion de valor relacional sum_edges alpha * (V_w (.) g[rel])
            (composicion estilo DistMult; es lo que da poder de razonamiento por caminos).
  * Readout puntual MLP(x^L_v) -> score(B, N). Loss = CE de grafo completo (en lightning).

Una sola torre de atencion. NADA de V-RMPNN / QK-RMPNN (eso era KnowFormer y fallo).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# RWSE -- Random-Walk Structural Encoding (Dwivedi et al., "Graph Neural Networks
# with Learnable Structural and Positional Representations", ICLR 2022).
# ============================================================================
# p_k(v) = (P^k)_{vv}, con P = D^-1 A la matriz de transicion del random walk sobre
# la adyacencia RELATION-AGNOSTIC del grafo (k=1..walk_length). Es la probabilidad de
# que un random walk que parte en v vuelva a v en exactamente k pasos: una firma
# estructural local de cada nodo. PURAMENTE ESTRUCTURAL => inductivo (NO usa identidad
# de nodo; depende solo de la estructura, transfiere al grafo de test disjunto). Por eso
# NO viola la lista negra #3 (que prohibe PE de nodo / embeddings de entidad).


def compute_rwse(edge_index, num_nodes, walk_length):
    """Devuelve (N, walk_length): diag(P^k) para k=1..walk_length.

    edge_index: (E, 3) = (h, r, t). Se ignora la relacion; se usa la adyacencia
    simetrizada (el grafo ya trae aristas reversas, se simetriza por robustez).
    """
    device = edge_index.device
    src, dst = edge_index[:, 0], edge_index[:, 2]
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    deg = A.sum(-1, keepdim=True).clamp(min=1.0)
    P = A / deg                                   # D^-1 A
    Pk = P.clone()
    diags = []
    for _ in range(walk_length):
        diags.append(torch.diagonal(Pk).clone())  # diag(P^k)
        Pk = Pk @ P
    return torch.stack(diags, dim=-1)             # (N, walk_length)


def rwse_features(graph, device, walk_length, proj):
    """RWSE proyectada a hidden_dim, (N, hidden). Cachea el RWSE crudo (N, walk_length)
    en el objeto graph: es fijo por split (train/val = train graph; test = grafo ind)."""
    cache = getattr(graph, '_rwse_cache', None)
    if cache is None or cache.size(-1) != walk_length or cache.device != device:
        ei = graph.edge_index.to(device)
        cache = compute_rwse(ei, graph.num_nodes, walk_length)
        graph._rwse_cache = cache
    return proj(cache)                            # (N, hidden)


def compute_lappe(edge_index, num_nodes, k):
    """Devuelve (N, k): los k autovectores no triviales del Laplaciano normalizado
    simetrico L = I - D^-1/2 A D^-1/2, correspondientes a los k autovalores mas chicos
    (se salta el trivial ~0, indice 0). Si N-1 < k se rellena con ceros.

    Como RWSE: depende SOLO de la estructura (no de identidad de nodo) => inductivo-safe,
    transfiere al grafo de test disjunto. NO viola lista negra #3. La relacion se ignora
    (adyacencia simetrizada). Los autovectores tienen ambiguedad de signo -> se hace
    sign-flip aleatorio en train (ver lappe_features)."""
    device = edge_index.device
    src, dst = edge_index[:, 0], edge_index[:, 2]
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    deg = A.sum(-1)
    dinv = deg.clamp(min=1.0).pow(-0.5)
    L = torch.eye(num_nodes, device=device) - dinv.unsqueeze(1) * A * dinv.unsqueeze(0)
    # eigh: autovalores ascendentes. Se computa en float64 para estabilidad numerica.
    evals, evecs = torch.linalg.eigh(L.double())
    evecs = evecs.to(A.dtype)                     # (N, N), columnas = autovectores
    # Saltar el trivial (indice 0), tomar los siguientes k.
    pe = evecs[:, 1:k + 1]                         # (N, <=k)
    if pe.size(1) < k:                            # grafo chico: rellenar con ceros
        pad = torch.zeros(num_nodes, k - pe.size(1), device=device)
        pe = torch.cat([pe, pad], dim=1)
    return pe                                     # (N, k)


def lappe_features(graph, device, k, proj, training):
    """LapPE proyectado a hidden_dim, (N, hidden). Cachea los autovectores crudos (N, k)
    en el objeto graph (fijos por split). En train se aplica sign-flip aleatorio por
    autovector para no memorizar el signo arbitrario de eigh (tratamiento canonico de
    LapPE); en eval se usan tal cual."""
    cache = getattr(graph, '_lappe_cache', None)
    if cache is None or cache.size(-1) != k or cache.device != device:
        ei = graph.edge_index.to(device)
        cache = compute_lappe(ei, graph.num_nodes, k)
        graph._lappe_cache = cache
    pe = cache
    if training:
        sign = torch.randint(0, 2, (k,), device=device, dtype=pe.dtype) * 2 - 1  # +-1
        pe = pe * sign.unsqueeze(0)
    return proj(pe)                               # (N, hidden)


# ----------------------------------------------------------------------------
# Source-conditioned random-walk labeling (labeling trick nativo de transformer).
# ----------------------------------------------------------------------------
# A DIFERENCIA de RWSE (diag(P^k), global por nodo) y LapPE (autovectores del
# Laplaciano, globales), esto es CONDICIONADO A LA QUERY: para cada query, el feature
# del nodo v es [P^1[head,v], ..., P^K[head,v]] = probabilidad de landing de un random
# walk de k pasos DESDE el head. Motivacion: con el labeling clasico de NBFNet
# (x^0_v = emb(r_q) si v==head, si no 0) todos los nodos no-source arrancan IDENTICOS
# (cero) => la capa 1 del full attention es degenerada (Q/K/V iguales) y el transformer
# gasta capas reconstruyendo asimetria via las aristas (rehaciendo message passing). Este
# labeling rompe esa simetria dandole a cada nodo una coordenada de proximidad al head,
# distinta por query. Depende SOLO de estructura + head (no de identidad de nodo) =>
# inductivo-safe, transfiere al grafo disjunto. NO viola lista negra #3 (no es PE de
# nodo estatica: cambia con la query, es el labeling trick condicionado a la fuente).


def compute_source_rw(edge_index, num_nodes, walk_length):
    """Devuelve (walk_length, N, N): P^k para k=1..walk_length con P = D^-1 A (adyacencia
    simetrizada, relacion ignorada como en RWSE/LapPE). La fila head de P^k son las
    probabilidades de landing de un random walk de k pasos que arranca en head."""
    device = edge_index.device
    src, dst = edge_index[:, 0], edge_index[:, 2]
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    deg = A.sum(-1, keepdim=True).clamp(min=1.0)
    P = A / deg                                   # D^-1 A
    Pk = P.clone()
    powers = []
    for _ in range(walk_length):
        powers.append(Pk.clone())                 # P^k
        Pk = Pk @ P
    return torch.stack(powers, dim=0)             # (walk_length, N, N)


def source_rw_features(graph, device, walk_length, proj, h_index):
    """Labeling condicionado a la query. Devuelve (B, N, hidden): para cada query b del
    batch, feature del nodo v = proj([P^1[head_b,v], ..., P^K[head_b,v]]). Cachea el stack
    (K, N, N) por split (estructura fija). El gather por head es constante (no grad); el
    grad fluye solo por proj (barato)."""
    cache = getattr(graph, '_source_rw_cache', None)
    if cache is None or cache.size(0) != walk_length or cache.device != device:
        ei = graph.edge_index.to(device)
        cache = compute_source_rw(ei, graph.num_nodes, walk_length)
        graph._source_rw_cache = cache
    feats = cache[:, h_index, :]                  # (K, B, N)
    feats = feats.permute(1, 2, 0)                # (B, N, K)
    return proj(feats)                            # (B, N, hidden)


# ----------------------------------------------------------------------------
# Relational Path Bias (RPB) -- opcion B: encoding de camino RELACIONAL como
# termino PAR (bias) en la atencion, no como feature de nodo en x^0.
# ----------------------------------------------------------------------------
# Diagnostico de source_rw / RWSE / LapPE: re-codificar estructura como feature
# de nodo en x^0 es redundante con el MP dentro del horizonte y diluye el label
# de la fuente. RPB es distinto en 3 ejes: (a) es RELACIONAL (compone tipos de
# relacion a lo largo del camino, no adyacencia relation-agnostic), (b) esta
# CONDICIONADO A LA QUERY (la compatibilidad de cada arista con r_q modula el
# camino), (c) entra como BIAS PAR en el logit de atencion, no corrompe la
# condicion de borde. La profundidad K del camino esta DESACOPLADA de num_layer,
# asi que inyecta evidencia relacional compuesta FUERA del horizonte de L saltos
# -- el unico margen que GOALS.md deja abierto para superar al message passing.
#
# Mecanica: potencial de evidencia sembrado en la fuente s, propagado K saltos
# por el grafo dirigido, con cada arista pesada por phi_q(rel)=tanh(<u_{r_q},w_rel>)
# (score DistMult del tipo de relacion contra la query, acotado). p^k[b,v] =
# (1/deg_in[v]) sum_{i->v} phi_q(rel) p^{k-1}[b,i]. Es un path-ranking score
# diferenciable, query-conditioned, fuente->candidato. Se proyecta (K -> H) con
# init CERO => el modelo arranca IDENTICO al full attention baseline y aprende a
# usar el prior de camino. Escalar por nodo (no dilucion 64-dim como source_rw).
# Inductivo-safe: solo aristas, tipos de relacion (compartidos) y la fuente.


class RelationalPathBias(nn.Module):
    """Prior de camino relacional query-conditioned como bias per-key de atencion."""

    def __init__(self, num_relation, num_heads, hops, dim):
        super().__init__()
        self.num_relation = num_relation
        self.num_heads = num_heads
        self.hops = hops
        self.dim = dim
        self.scale = 1.0 / math.sqrt(dim)
        # u_{r_q}: embedding de la relacion de la query; w_rel: embedding de la
        # relacion de cada arista. phi = tanh(<u,w>/sqrt(dim)) in [-1,1].
        self.query_emb = nn.Embedding(num_relation, dim)
        self.rel_emb = nn.Embedding(num_relation, dim)
        # K evidencias por nodo -> bias por cabeza. Init CERO (baseline al arranque).
        self.to_bias = nn.Linear(hops, num_heads)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)

    def forward(self, edges, h_index, r_index, num_nodes):
        """Devuelve bias (B, H, N): evidencia de camino relacional s->v por query,
        proyectada a un bias per-key para el logit de atencion."""
        src, rel, dst = edges
        B = h_index.size(0)
        N = num_nodes
        E = src.size(0)
        device = h_index.device

        # phi_q(rel) para cada arista y cada query del batch: (B, E), acotado.
        u = self.query_emb(r_index)                      # (B, dim)
        w = self.rel_emb.weight                          # (R, dim)
        phi_rel = torch.tanh((u @ w.t()) * self.scale)   # (B, R)
        phi_e = phi_rel[:, rel]                           # (B, E)

        # grado de entrada (normaliza la propagacion; usa el grafo ya enmascarado).
        deg_in = torch.zeros(N, device=device)
        deg_in.index_add_(0, dst, torch.ones(E, device=device))
        deg_in = deg_in.clamp(min=1.0)                    # (N,)

        # potencial sembrado en la fuente.
        p = torch.zeros(B, N, device=device)
        p[torch.arange(B, device=device), h_index] = 1.0
        src_idx = src.unsqueeze(0).expand(B, E)
        dst_idx = dst.unsqueeze(0).expand(B, E)
        evid = []
        for _ in range(self.hops):
            msg = phi_e * p.gather(1, src_idx)            # (B, E) evidencia src->dst
            p_new = torch.zeros(B, N, device=device)
            p_new.scatter_add_(1, dst_idx, msg)           # agrega en dst
            p_new = p_new / deg_in.unsqueeze(0)           # normaliza por grado
            evid.append(p_new)
            p = p_new
        evid = torch.stack(evid, dim=-1)                  # (B, N, K)
        bias = self.to_bias(evid)                         # (B, N, H)
        return bias.permute(0, 2, 1)                      # (B, H, N)


class RelationalAttentionLayer(nn.Module):
    """Una capa: atencion full all-pairs con bias y valor relacional + FFN (pre-LN)."""

    def __init__(self, hidden_dim, num_heads, num_relation, drop):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim debe ser divisible por num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_relation = num_relation
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        # (i) bias escalar relacional por (cabeza, relacion).
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, num_relation))
        # (ii) modulacion DistMult del valor por (cabeza, relacion). Init ~1 => arranca
        #      cerca de "pasar el valor del vecino sin transformar".
        self.rel_value = nn.Parameter(torch.ones(num_heads, num_relation, self.head_dim))

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, edges, path_bias=None):
        """x: (B, N, D). edges: (src, rel, dst) cada uno (E,), arista src --rel--> dst.
        path_bias: (B, H, N) opcional -- bias per-key de camino relacional (RPB).

        El nodo destino dst agrega informacion de src (igual que el message passing de
        NBFNet para predecir cola: la evidencia fluye de la fuente hacia el candidato).
        """
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        src, rel, dst = edges

        h = self.norm1(x)
        q = self.to_q(h).view(B, N, H, hd).transpose(1, 2)  # (B,H,N,hd)
        k = self.to_k(h).view(B, N, H, hd).transpose(1, 2)
        v = self.to_v(h).view(B, N, H, hd).transpose(1, 2)

        # Logits densos all-pairs.
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,N,N)

        # (i) bias relacional: a la celda (dst, src) se suma rel_bias[:, rel].
        #     Construimos un denso (H,N,N) (compartido en el batch: el grafo es fijo).
        bias = logits.new_zeros(H, N * N)
        flat_idx = dst * N + src                      # posicion (dst,src) aplanada
        bias.index_add_(1, flat_idx, self.rel_bias[:, rel])  # (H, E) -> columnas
        logits = logits + bias.view(1, H, N, N)

        # (i-b) bias de camino relacional (RPB): per-key (src=dim -1), broadcast en
        #       el nodo que atiende (dst). dst atiende mas a src ricos en evidencia
        #       de camino relacional compuesto desde la fuente de la query.
        if path_bias is not None:
            logits = logits + path_bias.unsqueeze(2)  # (B,H,1,N) -> broadcast en dst

        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)

        # Salida base full attention.
        out = torch.matmul(attn, v)                   # (B,H,N,hd)

        # (ii) correccion de valor relacional sobre aristas (composicion DistMult):
        #      out[dst] += alpha[dst, src] * (V[src] (.) g[rel]).
        a_e = attn[:, :, dst, src]                    # (B,H,E)
        v_e = v[:, :, src, :]                         # (B,H,E,hd)
        g_e = self.rel_value[:, rel, :].unsqueeze(0)  # (1,H,E,hd)
        msg = a_e.unsqueeze(-1) * v_e * g_e           # (B,H,E,hd)
        out.index_add_(2, dst, msg)                   # scatter-add en el destino

        out = out.transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.to_out(out))           # residual atencion
        x = x + self.ffn(self.norm2(x))               # residual FFN (pre-LN)
        return x


class GraphTransformer(nn.Module):
    """Stack de capas RelationalAttention + labeling trick + readout puntual."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 use_rwse=False, rwse_dim=16, use_lappe=False, lappe_dim=16,
                 use_source_rw=False, source_rw_dim=8,
                 use_rpb=False, rpb_hops=4, rpb_dim=16):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim
        self.use_lappe = use_lappe
        self.lappe_dim = lappe_dim
        self.use_source_rw = use_source_rw
        self.source_rw_dim = source_rw_dim
        self.use_rpb = use_rpb

        # Embedding de relacion para el labeling trick (query). Compartido train/test.
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        # RWSE: encoding estructural por nodo (inductivo) sumado a x^0.
        if use_rwse:
            self.rwse_proj = nn.Linear(rwse_dim, hidden_dim)
        # LapPE: autovectores del Laplaciano por nodo (inductivo) sumados a x^0.
        if use_lappe:
            self.lappe_proj = nn.Linear(lappe_dim, hidden_dim)
        # Source-conditioned RW: labeling condicionado a la query (K landing probs
        # desde el head por nodo). NO lleva unsqueeze en forward: ya es (B, N, .).
        if use_source_rw:
            self.source_rw_proj = nn.Linear(source_rw_dim, hidden_dim)
        # RPB: bias de camino relacional query-conditioned (opcion B). Compartido
        # entre capas (se computa una vez por batch y se pasa a cada capa).
        if use_rpb:
            self.rpb = RelationalPathBias(num_relation, num_heads, rpb_hops, rpb_dim)

        self.layers = nn.ModuleList([
            RelationalAttentionLayer(hidden_dim, num_heads, num_relation, drop)
            for _ in range(num_layer)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batched_data):
        h_index = batched_data['h_index']   # (B,)
        r_index = batched_data['r_index']   # (B,)
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)     # (E, 3) = (h, r, t)
        # En train se quita la arista de la query (y su reversa) via graph_mask.
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        edges = (src, rel, dst)

        # Labeling trick: x^0_v = emb(r_q) si v == head, si no 0.
        x = torch.zeros(B, N, self.hidden_dim, device=device)
        q = self.query_emb(r_index)                   # (B, D)
        x[torch.arange(B, device=device), h_index] = q
        # RWSE estructural sumado a todos los nodos (broadcast en el batch).
        if self.use_rwse:
            x = x + rwse_features(graph, device, self.rwse_dim, self.rwse_proj).unsqueeze(0)
        if self.use_lappe:
            x = x + lappe_features(graph, device, self.lappe_dim, self.lappe_proj,
                                   self.training).unsqueeze(0)
        if self.use_source_rw:
            x = x + source_rw_features(graph, device, self.source_rw_dim,
                                       self.source_rw_proj, h_index)

        # Bias de camino relacional (opcion B): se computa una vez y entra en el
        # logit de cada capa como termino par (per-key), sin tocar x^0.
        path_bias = self.rpb(edges, h_index, r_index, N) if self.use_rpb else None

        for layer in self.layers:
            x = layer(x, edges, path_bias)

        x = self.norm_out(x)
        score = self.readout(x).squeeze(-1)           # (B, N)
        return score


# ============================================================================
# NBFNet (baseline para comparacion apples-to-apples en este mismo harness).
# ============================================================================
# Reimplementacion fiel del Neural Bellman-Ford Network (Zhu et al. 2021) con la
# config inductiva de FB15k-237 (NBFNet/config/inductive/fb15k237.yaml):
#   message=distmult, aggregate=pna, short_cut=yes, layer_norm=yes, dependent=yes.
# Mismo interfaz que GraphTransformer: forward(batched_data) -> score (B, N), misma
# data y mismo eval full-filtered. SIN atencion: solo message passing condicionado a
# la fuente. Es el caso "w/o attention" y el techo que el RFAT debe superar.


def _scatter_reduce(msg, index, num_nodes, reduce):
    """Agrega msg (B, M, d) por index (M,) en (B, N, d) con la reduccion dada."""
    B, M, d = msg.shape
    out = msg.new_zeros(B, num_nodes, d)
    idx = index.view(1, M, 1).expand(B, M, d)
    out.scatter_reduce_(1, idx, msg, reduce=reduce, include_self=False)
    return out


class NBFNetConv(nn.Module):
    """Una capa de Bellman-Ford generalizado (mensajes DistMult dependientes de query)."""

    def __init__(self, input_dim, output_dim, num_relation, aggregate='pna', layer_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.aggregate = aggregate

        # dependent=yes: las relaciones por capa se derivan de la query.
        self.relation_linear = nn.Linear(input_dim, num_relation * input_dim)

        n_aggr = 12 if aggregate == 'pna' else 1   # pna: 4 aggregadores x 3 scalers
        self.linear = nn.Linear((n_aggr + 1) * input_dim, output_dim)  # +1: concat input
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None

    def forward(self, h, query, edges, num_nodes, boundary):
        """h: (B,N,d) estado previo. query: (B,d). edges: (src,rel,dst). boundary: (B,N,d)."""
        B, N, d = h.shape
        src, rel, dst = edges
        E = src.size(0)

        rel_emb = self.relation_linear(query).view(B, self.num_relation, d)  # (B, R, d)

        # Mensaje DistMult sobre aristas: m_e = h[src] (.) rel_emb[rel].
        msg_edge = h[:, src, :] * rel_emb[:, rel, :]                 # (B, E, d)
        # Boundary como self-loop (condicion inicial v=u): se agrega como mensaje propio.
        node_out = torch.cat([dst, torch.arange(N, device=h.device)])  # (E+N,)
        msg = torch.cat([msg_edge, boundary], dim=1)                 # (B, E+N, d)

        if self.aggregate == 'sum':
            update = _scatter_reduce(msg, node_out, N, 'sum')        # (B, N, d)
        else:  # pna
            mean = _scatter_reduce(msg, node_out, N, 'mean')
            sq = _scatter_reduce(msg * msg, node_out, N, 'mean')
            mx = _scatter_reduce(msg, node_out, N, 'amax')
            mn = _scatter_reduce(msg, node_out, N, 'amin')
            std = (sq - mean * mean).clamp(min=1e-6).sqrt()
            feat = torch.stack([mean, mx, mn, std], dim=-1).flatten(-2)  # (B,N,4d)
            deg = torch.zeros(N, device=h.device).index_add_(
                0, node_out, torch.ones(E + N, device=h.device)).unsqueeze(-1)  # (N,1)
            scale = deg.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale,
                                1.0 / scale.clamp(min=1e-6)], dim=-1)        # (N,3)
            update = (feat.unsqueeze(-1) * scales.view(1, N, 1, 3)).flatten(-2)  # (B,N,12d)

        out = self.linear(torch.cat([h, update], dim=-1))           # combine [input, update]
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        out = F.relu(out)
        return out


class NBFNet(nn.Module):
    """Neural Bellman-Ford Network condicionado a la fuente (tail prediction)."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads=None, drop=0.0,
                 aggregate='pna', short_cut=True):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.short_cut = short_cut

        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        self.layers = nn.ModuleList([
            NBFNetConv(hidden_dim, hidden_dim, num_relation, aggregate=aggregate)
            for _ in range(num_layer)
        ])
        # Readout puntual sobre h^L_{u->v} (concat con la query para condicionar el score).
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, batched_data):
        """Corre el message passing y devuelve los estados de nodo h^L_{u->v} (B, N, d),
        antes del readout. Reusado por SparseNBFValueTransformer como V-stream."""
        h_index = batched_data['h_index']
        r_index = batched_data['r_index']
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        edges = (edge_index[:, 0], edge_index[:, 1], edge_index[:, 2])

        query = self.query_emb(r_index)                              # (B, d)
        boundary = torch.zeros(B, N, self.hidden_dim, device=device)
        boundary[torch.arange(B, device=device), h_index] = query

        h = boundary
        for layer in self.layers:
            out = layer(h, query, edges, N, boundary)
            if self.short_cut and out.shape == h.shape:
                out = out + h
            h = out
        return h                                                    # (B, N, d)

    def forward(self, batched_data):
        h = self.encode(batched_data)
        score = self.mlp(h).squeeze(-1)                             # (B, N)
        return score


# ============================================================================
# Sparse Graph Transformer: atencion restringida a la ADYACENCIA del grafo.
# ============================================================================
# Variante sparse del RFAT: cada nodo destino atiende SOLO a sus vecinos entrantes
# (aristas), no a los N nodos. Softmax por nodo sobre sus aristas entrantes + un
# self-loop. Estructuralmente es "NBFNet con agregacion APRENDIDA por atencion en vez
# de sum/pna fija" (caso (b) del analisis: graph attention ~ un hop con pesos aprendidos).
# Mismo labeling trick / bias+valor relacional / readout / eval que el RFAT denso.


class SparseRelationalAttentionLayer(nn.Module):
    """Atencion sparse sobre aristas + FFN (pre-LN).

    attn controla la agregacion por nodo destino:
      - 'softmax': segment-softmax (promedio ponderado convexo; ciego al numero de
        aristas de soporte y al grado, porque normaliza a sum(alpha)=1).
      - 'sigmoid': gates sigmoides por arista SIN normalizar => suma ponderada
        aprendida. Conserva el conteo de caminos de evidencia y la sensibilidad al
        grado (como la agregacion sum de NBFNet) manteniendo la selectividad por query.
      - 'degree': segment-softmax reescalado por log(1+grado_in) del destino =>
        reinyecta el conteo como scaler (estilo PNA) sin abandonar el softmax.
    """

    def __init__(self, hidden_dim, num_heads, num_relation, drop, attn='softmax'):
        super().__init__()
        assert hidden_dim % num_heads == 0
        assert attn in ('softmax', 'sigmoid', 'degree')
        self.attn = attn
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # +1 relacion extra reservada al self-loop (indice = num_relation).
        self.self_rel = num_relation
        R = num_relation + 1

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        self.rel_bias = nn.Parameter(torch.zeros(num_heads, R))
        self.rel_value = nn.Parameter(torch.ones(num_heads, R, self.head_dim))

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, edges):
        """edges: (src, rel, dst) YA aumentadas con self-loops (rel==self_rel)."""
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        src, rel, dst = edges
        E = src.size(0)

        h = self.norm1(x)
        q = self.to_q(h).view(B, N, H, hd)
        k = self.to_k(h).view(B, N, H, hd)
        v = self.to_v(h).view(B, N, H, hd)

        # Logit por arista: (q_dst . k_src)/sqrt(d) + bias relacional. (B, E, H)
        logit = (q[:, dst] * k[:, src]).sum(-1) * self.scale
        logit = logit + self.rel_bias[:, rel].transpose(0, 1).unsqueeze(0)  # (1,E,H)

        if self.attn == 'sigmoid':
            # Gate por arista, sin normalizacion (no necesita estabilizacion por max).
            alpha = torch.sigmoid(logit)                            # (B, E, H)
        else:
            # Segment-softmax por nodo destino (sobre sus aristas entrantes + self-loop).
            idx = dst.view(1, E, 1).expand(B, E, H)
            node_max = logit.new_full((B, N, H), float('-inf'))
            node_max.scatter_reduce_(1, idx, logit, reduce='amax', include_self=False)
            logit = (logit - node_max.gather(1, idx)).exp()
            denom = logit.new_zeros(B, N, H)
            denom.index_add_(1, dst, logit)
            alpha = logit / (denom.gather(1, idx) + 1e-9)           # (B, E, H)
        alpha = self.drop(alpha)

        # Mensaje relacional (composicion DistMult) ponderado por la atencion.
        g = self.rel_value[:, rel, :].permute(1, 0, 2).unsqueeze(0)  # (1, E, H, hd)
        msg = alpha.unsqueeze(-1) * v[:, src] * g
        out = msg.new_zeros(B, N, H, hd)
        out.index_add_(1, dst, msg)                                # (B, N, H, hd)

        if self.attn == 'degree':
            deg = torch.zeros(N, device=x.device)
            deg.index_add_(0, dst, torch.ones(E, device=x.device))
            out = out * torch.log1p(deg).view(1, N, 1, 1)

        out = out.reshape(B, N, D)
        x = x + self.drop(self.to_out(out))
        x = x + self.ffn(self.norm2(x))
        return x


class SparseGraphTransformer(nn.Module):
    """Stack de capas de atencion sparse por adyacencia + labeling trick + readout."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 attn='softmax',
                 use_rwse=False, rwse_dim=16, use_lappe=False, lappe_dim=16,
                 use_source_rw=False, source_rw_dim=8):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.self_rel = num_relation
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim
        self.use_lappe = use_lappe
        self.lappe_dim = lappe_dim
        self.use_source_rw = use_source_rw
        self.source_rw_dim = source_rw_dim
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        if use_rwse:
            self.rwse_proj = nn.Linear(rwse_dim, hidden_dim)
        if use_lappe:
            self.lappe_proj = nn.Linear(lappe_dim, hidden_dim)
        # Source-conditioned RW: labeling condicionado a la query (K landing probs
        # desde el head por nodo). NO lleva unsqueeze en forward: ya es (B, N, .).
        if use_source_rw:
            self.source_rw_proj = nn.Linear(source_rw_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SparseRelationalAttentionLayer(hidden_dim, num_heads, num_relation, drop,
                                           attn=attn)
            for _ in range(num_layer)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, batched_data):
        h_index = batched_data['h_index']
        r_index = batched_data['r_index']
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        # Aumentar con self-loops (cada nodo se atiende a si mismo via relacion self).
        loop = torch.arange(N, device=device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])
        rel = torch.cat([rel, torch.full((N,), self.self_rel, device=device)])
        edges = (src, rel, dst)

        x = torch.zeros(B, N, self.hidden_dim, device=device)
        x[torch.arange(B, device=device), h_index] = self.query_emb(r_index)
        if self.use_rwse:
            x = x + rwse_features(graph, device, self.rwse_dim, self.rwse_proj).unsqueeze(0)
        if self.use_lappe:
            x = x + lappe_features(graph, device, self.lappe_dim, self.lappe_proj,
                                   self.training).unsqueeze(0)
        if self.use_source_rw:
            x = x + source_rw_features(graph, device, self.source_rw_dim,
                                       self.source_rw_proj, h_index)

        for layer in self.layers:
            x = layer(x, edges)

        x = self.norm_out(x)
        return self.readout(x).squeeze(-1)


# ============================================================================
# Sparse attention + expander graphs (estilo Exphormer).
# ============================================================================
# Las aristas expander (grafo aleatorio d-regular) dan atajos estructurales fuera del
# horizonte de propagacion. Fieles a Exphormer, NO llevan relacion KG real (son aleatorias
# => seria inyectar hechos falsos) sino un tipo de arista aprendido propio: aqui una
# relacion sintetica reservada R_exp = num_relation+1 (el self-loop usa num_relation), que
# entra en los dos canales relacionales del sparse (bias b[head,R_exp] y valor g[R_exp]).
# Inductive-safe: R_exp se comparte train/test y las aristas no dependen de identidad de
# entidad. En Exphormer el edge_attr expander es un nn.Embedding(1) compartido; aqui su
# analogo es la fila R_exp de rel_bias/rel_value. Ver SESSION_NOTES.md (2026-07-20).

def generate_expander_edges(num_nodes, degree, seed=0):
    """Grafo aleatorio d-regular simetrico (permutation algorithm de Exphormer,
    generate_random_regular_graph1). Devuelve (src, dst) long en CPU, sin self-loops;
    simetrico: si (x,y) esta, (y,x) tambien."""
    if num_nodes <= degree + 1:
        # grafo demasiado chico: conectar todos con todos (sin self-loops).
        idx = torch.arange(num_nodes)
        src = idx.repeat_interleave(num_nodes)
        dst = idx.repeat(num_nodes)
    else:
        g = torch.Generator().manual_seed(seed)
        base = torch.arange(num_nodes)
        senders = base.repeat(degree)                                  # [0..n-1]*degree
        receivers = torch.cat([base[torch.randperm(num_nodes, generator=g)]
                               for _ in range(degree)])
        src = torch.cat([senders, receivers])
        dst = torch.cat([receivers, senders])                          # simetrizar
    mask = src != dst                                                  # quitar self-loops
    return src[mask].contiguous(), dst[mask].contiguous()


class SparseExpanderGraphTransformer(nn.Module):
    """SparseGraphTransformer aumentado con aristas expander (estilo Exphormer).

    Identico al sparse por adyacencia pero concatenando, a las aristas reales + self-loops,
    un grafo aleatorio d-regular cuyas aristas llevan la relacion sintetica reservada
    R_exp (num_relation+1). Las tablas relacionales de la capa se dimensionan a
    num_relation+2 (self-loop en num_relation, expander en num_relation+1)."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 exp_degree=4, exp_seed=0, attn='softmax',
                 use_rwse=False, rwse_dim=16, use_lappe=False, lappe_dim=16,
                 use_source_rw=False, source_rw_dim=8):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.self_rel = num_relation          # relacion reservada del self-loop
        self.exp_rel = num_relation + 1        # relacion reservada de las aristas expander
        self.exp_degree = exp_degree
        self.exp_seed = exp_seed
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim
        self.use_lappe = use_lappe
        self.lappe_dim = lappe_dim
        self.use_source_rw = use_source_rw
        self.source_rw_dim = source_rw_dim
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        if use_rwse:
            self.rwse_proj = nn.Linear(rwse_dim, hidden_dim)
        if use_lappe:
            self.lappe_proj = nn.Linear(lappe_dim, hidden_dim)
        if use_source_rw:
            self.source_rw_proj = nn.Linear(source_rw_dim, hidden_dim)
        # Se pasa num_relation+1 a la capa => reserva una relacion extra (R_exp) ademas
        # del self-loop: R = (num_relation+1)+1 = num_relation+2 filas en rel_bias/rel_value.
        self.layers = nn.ModuleList([
            SparseRelationalAttentionLayer(hidden_dim, num_heads, num_relation + 1, drop,
                                           attn=attn)
            for _ in range(num_layer)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self._exp_cache = {}   # num_nodes -> (src, dst) en CPU (generado una vez por grafo)

    def _expander(self, num_nodes, device):
        if num_nodes not in self._exp_cache:
            self._exp_cache[num_nodes] = generate_expander_edges(
                num_nodes, self.exp_degree, self.exp_seed)
        src, dst = self._exp_cache[num_nodes]
        return src.to(device), dst.to(device)

    def forward(self, batched_data):
        h_index = batched_data['h_index']
        r_index = batched_data['r_index']
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        # Aumentar con self-loops (relacion self_rel).
        loop = torch.arange(N, device=device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])
        rel = torch.cat([rel, torch.full((N,), self.self_rel, device=device)])
        # Aristas expander (relacion sintetica reservada R_exp). No se enmascaran con
        # graph_mask: son estructurales e independientes de la query.
        exp_src, exp_dst = self._expander(N, device)
        src = torch.cat([src, exp_src])
        dst = torch.cat([dst, exp_dst])
        rel = torch.cat([rel, torch.full((exp_src.size(0),), self.exp_rel, device=device)])
        edges = (src, rel, dst)

        x = torch.zeros(B, N, self.hidden_dim, device=device)
        x[torch.arange(B, device=device), h_index] = self.query_emb(r_index)
        if self.use_rwse:
            x = x + rwse_features(graph, device, self.rwse_dim, self.rwse_proj).unsqueeze(0)
        if self.use_lappe:
            x = x + lappe_features(graph, device, self.lappe_dim, self.lappe_proj,
                                   self.training).unsqueeze(0)
        if self.use_source_rw:
            x = x + source_rw_features(graph, device, self.source_rw_dim,
                                       self.source_rw_proj, h_index)

        for layer in self.layers:
            x = layer(x, edges)

        x = self.norm_out(x)
        return self.readout(x).squeeze(-1)


# ============================================================================
# Sparse attention con V proveniente de un stream NBFNet (lista negra #6, a proposito).
# ============================================================================
# Variante experimental pedida explicitamente para completar la tabla. Recrea el patron
# refutado de KnowFormer (V-stream separado alimentando la atencion): Q,K salen del stream
# de atencion (labeling trick), pero V sale de las representaciones de nodo de un NBFNet
# corrido aparte (h^L_{u->v}, query-conditioned). La atencion sparse solo REPONDERA por
# adyacencia lo que NBFNet ya calculo => se espera redundante (mejor caso ~= NBFNet). Se
# implementa para tenerlo medido, no porque se espere que gane (ver SESSION_NOTES.md).


class SparseNBFValueLayer(nn.Module):
    """Igual a SparseRelationalAttentionLayer pero V proviene de un tensor externo
    (v_src, el stream NBFNet) en vez del estado x. Q,K siguen saliendo de x."""

    def __init__(self, hidden_dim, num_heads, num_relation, drop):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.self_rel = num_relation
        R = num_relation + 1

        self.norm1 = nn.LayerNorm(hidden_dim)   # Q,K (desde x)
        self.norm_v = nn.LayerNorm(hidden_dim)  # V (desde el stream NBFNet)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        self.rel_bias = nn.Parameter(torch.zeros(num_heads, R))
        self.rel_value = nn.Parameter(torch.ones(num_heads, R, self.head_dim))

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, v_src, edges):
        """x: (B,N,D) estado de atencion (Q,K). v_src: (B,N,D) stream NBFNet (V).
        edges: (src, rel, dst) YA aumentadas con self-loops (rel==self_rel)."""
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        src, rel, dst = edges
        E = src.size(0)

        h = self.norm1(x)
        q = self.to_q(h).view(B, N, H, hd)
        k = self.to_k(h).view(B, N, H, hd)
        v = self.to_v(self.norm_v(v_src)).view(B, N, H, hd)         # V del stream NBFNet

        # Logit por arista: (q_dst . k_src)/sqrt(d) + bias relacional. (B, E, H)
        logit = (q[:, dst] * k[:, src]).sum(-1) * self.scale
        logit = logit + self.rel_bias[:, rel].transpose(0, 1).unsqueeze(0)  # (1,E,H)

        # Segment-softmax por nodo destino (sobre sus aristas entrantes + self-loop).
        idx = dst.view(1, E, 1).expand(B, E, H)
        node_max = logit.new_full((B, N, H), float('-inf'))
        node_max.scatter_reduce_(1, idx, logit, reduce='amax', include_self=False)
        logit = (logit - node_max.gather(1, idx)).exp()
        denom = logit.new_zeros(B, N, H)
        denom.index_add_(1, dst, logit)
        alpha = logit / (denom.gather(1, idx) + 1e-9)              # (B, E, H)
        alpha = self.drop(alpha)

        # Mensaje relacional (composicion DistMult) ponderado por la atencion.
        g = self.rel_value[:, rel, :].permute(1, 0, 2).unsqueeze(0)  # (1, E, H, hd)
        msg = alpha.unsqueeze(-1) * v[:, src] * g
        out = msg.new_zeros(B, N, H, hd)
        out.index_add_(1, dst, msg)                                # (B, N, H, hd)

        out = out.reshape(B, N, D)
        x = x + self.drop(self.to_out(out))
        x = x + self.ffn(self.norm2(x))
        return x


class SparseNBFValueTransformer(nn.Module):
    """Atencion sparse por adyacencia cuyo V proviene de un stream NBFNet separado.

    El stream NBFNet se corre una vez para producir h^L_{u->v} (B,N,d); esas
    representaciones son el V de TODAS las capas de atencion (estilo KnowFormer V-stream).
    Q,K vienen del propio stream de atencion (labeling trick). Ambos streams se entrenan
    end-to-end (el gradiente fluye al NBFNet via V). Mismo readout/eval que el sparse."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 aggregate='pna', short_cut=True, use_rwse=False, rwse_dim=16,
                 use_lappe=False, lappe_dim=16, use_source_rw=False, source_rw_dim=8):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.self_rel = num_relation
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim
        self.use_lappe = use_lappe
        self.lappe_dim = lappe_dim
        self.use_source_rw = use_source_rw
        self.source_rw_dim = source_rw_dim

        # V-stream: NBFNet completo (se reusa encode() para tomar los estados de nodo).
        self.nbf = NBFNet(num_relation, num_layer, hidden_dim,
                          aggregate=aggregate, short_cut=short_cut)

        # Q,K-stream: labeling trick propio.
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        if use_rwse:
            self.rwse_proj = nn.Linear(rwse_dim, hidden_dim)
        if use_lappe:
            self.lappe_proj = nn.Linear(lappe_dim, hidden_dim)
        # Source-conditioned RW: labeling condicionado a la query (K landing probs
        # desde el head por nodo). NO lleva unsqueeze en forward: ya es (B, N, .).
        if use_source_rw:
            self.source_rw_proj = nn.Linear(source_rw_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SparseNBFValueLayer(hidden_dim, num_heads, num_relation, drop)
            for _ in range(num_layer)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, batched_data):
        h_index = batched_data['h_index']
        r_index = batched_data['r_index']
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        loop = torch.arange(N, device=device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])
        rel = torch.cat([rel, torch.full((N,), self.self_rel, device=device)])
        edges = (src, rel, dst)

        # V-stream: representaciones de nodo de NBFNet (query-conditioned). Mismo
        # graph_mask via batched_data => consistente con el stream de atencion.
        v_src = self.nbf.encode(batched_data)                      # (B, N, d)

        # Q,K-stream: labeling trick (+ RWSE / LapPE estructural).
        x = torch.zeros(B, N, self.hidden_dim, device=device)
        x[torch.arange(B, device=device), h_index] = self.query_emb(r_index)
        if self.use_rwse:
            x = x + rwse_features(graph, device, self.rwse_dim, self.rwse_proj).unsqueeze(0)
        if self.use_lappe:
            x = x + lappe_features(graph, device, self.lappe_dim, self.lappe_proj,
                                   self.training).unsqueeze(0)
        if self.use_source_rw:
            x = x + source_rw_features(graph, device, self.source_rw_dim,
                                       self.source_rw_proj, h_index)

        for layer in self.layers:
            x = layer(x, v_src, edges)

        x = self.norm_out(x)
        return self.readout(x).squeeze(-1)

