#include "psgn.h"

/* ── helpers ──────────────────────────────────────────────────────── */

static Edge *find_edge(const PredictiveGraph *g, int from, int to) {
    Edge *e = g->adjacency[from];
    while (e) { if (e->target == to) return e; e = e->next; }
    return NULL;
}

static int pred_cmp_desc(const void *a, const void *b) {
    float sa = ((const Prediction *)a)->score;
    float sb = ((const Prediction *)b)->score;
    return (sa < sb) ? 1 : (sa > sb) ? -1 : 0;
}

/* ── init / free ──────────────────────────────────────────────────── */

void graph_init(PredictiveGraph *g, int ctx_size, int scr_size) {
    memset(g->adjacency,   0, sizeof(g->adjacency));
    memset(g->node_exists, 0, sizeof(g->node_exists));
    g->ctx_sdr_size = ctx_size;
    g->scr_sdr_size = scr_size;
}

void graph_free_edge(Edge *e) {
    if (e->context_sdr) sdr_free(e->context_sdr);
    for (int i = 0; i < e->num_exemplars; i++) {
        if (e->scr_exemplars[i]) free(e->scr_exemplars[i]);
    }
}

void graph_free(PredictiveGraph *g) {
    for (int i = 0; i < MAX_NODES; i++) {
        Edge *e = g->adjacency[i];
        while (e) {
            Edge *nx = e->next;
            graph_free_edge(e);
            free(e);
            e = nx;
        }
    }
}

/* ── node / edge ops ─────────────────────────────────────────────── */

void graph_add_node(PredictiveGraph *g, int id) {
    if (id >= 0 && id < MAX_NODES) g->node_exists[id] = 1;
}

float graph_get_edge_weight(const PredictiveGraph *g, int from, int to) {
    if (from < 0 || from >= MAX_NODES) return 0.0f;
    Edge *e = find_edge(g, from, to);
    return e ? e->weight : 0.0f;
}

void graph_set_edge_weight(PredictiveGraph *g, int from, int to, float w,
                           const uint8_t *ctx, const uint8_t *scr) {
    graph_add_node(g, from);
    graph_add_node(g, to);

    Edge *e = find_edge(g, from, to);
    if (!e) {
        e = (Edge *)calloc(1, sizeof(Edge));
        e->target = to;
        e->num_exemplars = 0;
        e->next = g->adjacency[from];
        g->adjacency[from] = e;
    }
    e->weight = w;

    /* Context SDR: OR-merge (L2 context is coarse-grained, OR is fine) */
    if (ctx && g->ctx_sdr_size > 0) {
        if (!e->context_sdr) {
            e->context_sdr = sdr_create(g->ctx_sdr_size);
            sdr_copy(e->context_sdr, ctx, g->ctx_sdr_size);
        } else {
            sdr_or_inplace(e->context_sdr, ctx, g->ctx_sdr_size);
        }
    }

    /* Sparse Scratchpad Storage: Store active bit indices for pure set-membership lookup. */
    if (scr && g->scr_sdr_size > 0 && e->num_exemplars < MAX_EXEMPLARS) {
        int count = sdr_count(scr, g->scr_sdr_size);
        if (count > 0) {
            int *indices = (int *)malloc(count * sizeof(int));
            int idx = 0;
            for (int i = 0; i < g->scr_sdr_size; i++) {
                if (scr[i]) indices[idx++] = i;
            }
            e->scr_exemplars[e->num_exemplars] = indices;
            e->scr_exemplar_counts[e->num_exemplars] = count;
            e->num_exemplars++;
        }
    }
}

/* ── predictions with best-exemplar matching ─────────────────────── */

int graph_get_predictions(const PredictiveGraph *g, int cur,
                          const uint8_t *ctx_bias, const uint8_t *scr_bias,
                          Prediction *out, int max_out) {
    (void)ctx_bias;
    if (cur < 0 || cur >= MAX_NODES || !g->node_exists[cur]) return 0;
    
    int scr_bias_active = 0;
    if (scr_bias) scr_bias_active = sdr_count(scr_bias, g->scr_sdr_size);

    int n = 0;
    for (Edge *e = g->adjacency[cur]; e && n < max_out; e = e->next) {
        float score = e->weight;

        /* Scratchpad bias — Max-Sharpened Sparse Match */
        if (scr_bias && e->num_exemplars > 0) {
            float max_evidence = 0.0f;
            for (int ex = 0; ex < e->num_exemplars; ex++) {
                int overlap = 0;
                int *indices = e->scr_exemplars[ex];
                int count    = e->scr_exemplar_counts[ex];

                /* Fast intersection via bit-checks on dense scr_bias */
                for (int i = 0; i < count; i++) {
                    if (scr_bias[indices[i]]) overlap++;
                }

                float ovl_ratio = (float)overlap / count;
                /* Jaccard Similarity for Sub-Symbolic Rule Grounding */
                int current_active = sdr_count(scr_bias, g->scr_sdr_size);
                float jaccard = (float)overlap / (count + current_active - overlap);
                
                if (jaccard > 0.85f) { 
                    float evidence = powf(jaccard, 8.0f);
                    if (evidence > max_evidence) max_evidence = evidence;
                }
            }
            if (max_evidence > 0.00001f) {
                printf("      [VSA] Edge %d->%d Match Evidence: %.4f\n", cur, e->target, max_evidence);
            }
            score += max_evidence * 200000.0f;
        }

        out[n].node_id = e->target;
        out[n].score   = score;
        n++;
    }
    qsort(out, n, sizeof(Prediction), pred_cmp_desc);
    return n;
}

/* ── graph queries ───────────────────────────────────────────────── */

int graph_get_all_nodes(const PredictiveGraph *g, int *out, int max_out) {
    int n = 0;
    for (int i = 0; i < MAX_NODES && n < max_out; i++)
        if (g->node_exists[i]) out[n++] = i;
    return n;
}

int graph_get_incoming(const PredictiveGraph *g, int node,
                       int *src_out, float *w_out, int max_out) {
    int n = 0;
    for (int i = 0; i < MAX_NODES && n < max_out; i++) {
        Edge *e = find_edge(g, i, node);
        if (e) { src_out[n] = i; w_out[n] = e->weight; n++; }
    }
    return n;
}

float graph_contextual_similarity(const PredictiveGraph *g, int a, int b) {
    /* Heap-allocate to avoid stack overflow in deep call chains */
    int *set_a = (int *)calloc(MAX_NODES, sizeof(int));
    int *set_b = (int *)calloc(MAX_NODES, sizeof(int));
    for (Edge *e = g->adjacency[a]; e; e = e->next) set_a[e->target] = 1;
    for (Edge *e = g->adjacency[b]; e; e = e->next) set_b[e->target] = 1;
    int out_inter = 0, out_union = 0;
    for (int i = 0; i < MAX_NODES; i++) {
        if (set_a[i] && set_b[i]) out_inter++;
        if (set_a[i] || set_b[i]) out_union++;
    }
    float out_sim = out_union > 0 ? (float)out_inter / out_union : 0.0f;

    int in_inter = 0, in_union = 0;
    for (int i = 0; i < MAX_NODES; i++) {
        int ia = find_edge(g, i, a) != NULL;
        int ib = find_edge(g, i, b) != NULL;
        if (ia && ib) in_inter++;
        if (ia || ib) in_union++;
    }
    float in_sim = in_union > 0 ? (float)in_inter / in_union : 0.0f;
    free(set_a); free(set_b);
    return (out_sim + in_sim) / 2.0f;
}
