#include "psgn.h"

static int token_is_stable(const char *tok) {
    if (strcmp(tok, "+") == 0 || strcmp(tok, "=") == 0 || 
        strcmp(tok, ".") == 0 || strcmp(tok, "-") == 0 ||
        strcmp(tok, "*") == 0 || strcmp(tok, "/") == 0) return 1;
    const char *p = tok;
    if (*p == '-') p++;
    if (!*p) return 0;
    int dot = 0;
    while (*p) {
        if (*p == '.') { if (dot) return 0; dot = 1; }
        else if (!isdigit((unsigned char)*p)) return 0;
        p++;
    }
    return 1;
}

void updater_init(GraphUpdater *u, PredictiveGraph *g, SDREncoder *enc,
                  float lr, float dr) {
    u->graph    = g;
    u->encoder  = enc;
    u->learning_rate = lr;
    u->decay_rate    = dr;
    u->overlap_threshold = 0.8f;
    u->neuroplasticity_threshold = 0.3f;
    u->drift_rate = 2;
    u->neuroplastic_enabled = 1;
    u->previous_node_id = -1;
    u->update_count = 0;
}

float updater_calc_overlap(const uint8_t *a, const uint8_t *b, int size) {
    int overlap = 0, active = 0;
    for (int i = 0; i < size; i++) {
        overlap += __builtin_popcount(a[i] & b[i]);
        active  += __builtin_popcount(a[i]);
    }
    return active > 0 ? (float)overlap / active : 0.0f;
}

void updater_neuroplasticity_check(GraphUpdater *u, int node_id) {
    SDREncoder *enc = u->encoder;
    const char *t1  = enc->id_to_token[node_id];
    if (token_is_stable(t1)) return; /* don't mutate topological encodings */

    int *nodes = (int *)malloc(MAX_NODES * sizeof(int));
    int nn = graph_get_all_nodes(u->graph, nodes, MAX_NODES);

    for (int i = 0; i < nn; i++) {
        int other = nodes[i];
        if (other == node_id) continue;
        const char *t2 = enc->id_to_token[other];
        if (token_is_stable(t2)) continue;

        float sim = graph_contextual_similarity(u->graph, node_id, other);
        if (sim >= u->neuroplasticity_threshold) {
            sdr_encoder_mutate(enc, t1, t2, u->drift_rate);
            sdr_encoder_mutate(enc, t2, t1, u->drift_rate);
        }
    }
    free(nodes);
}

int updater_update(GraphUpdater *u, const uint8_t *sdr, int node_id,
                   const uint8_t *ctx, const uint8_t *scr) {
    int surprise = 0;
    int sz = u->encoder->sdr_size;

    if (u->previous_node_id < 0) {
        graph_add_node(u->graph, node_id);
        u->previous_node_id = node_id;
        const char *tok = u->encoder->id_to_token[node_id];
        if (tok[0]) sdr_encoder_record_bits(u->encoder, tok, sdr);
        return 0;
    }

    const char *tok = u->encoder->id_to_token[node_id];
    if (tok[0]) sdr_encoder_record_bits(u->encoder, tok, sdr);

    int matched = 0;
    /* Optimization: check ground-truth edge first */
    float bw = graph_get_edge_weight(u->graph, u->previous_node_id, node_id);
    if (bw > 0.0f) {
        uint8_t *pred_sdr = sdr_encoder_encode(u->encoder, tok);
        if (updater_calc_overlap(sdr, pred_sdr, sz) >= u->overlap_threshold) {
             graph_set_edge_weight(u->graph, u->previous_node_id, node_id,
                                  bw + u->learning_rate, ctx, scr);
             matched = 1;
        }
    }

    if (!matched) {
        Prediction preds[512];
        int np = graph_get_predictions(u->graph, u->previous_node_id, ctx, scr,
                                       preds, 512);
        for (int i = 0; i < np; i++) {
            int pid = preds[i].node_id;
            const char *pt = u->encoder->id_to_token[pid];
            if (!pt[0]) continue;
            uint8_t *pred_sdr = sdr_encoder_encode(u->encoder, pt);
            float ovl = updater_calc_overlap(sdr, pred_sdr, sz);
            if (ovl >= u->overlap_threshold) {
                float base = graph_get_edge_weight(u->graph, u->previous_node_id, pid);
                graph_set_edge_weight(u->graph, u->previous_node_id, pid,
                                     base + u->learning_rate, ctx, scr);
                matched = 1;
                if (pid == node_id) break;
            }
        }
    }

    if (!matched) {
        surprise = 1;
        graph_add_node(u->graph, node_id);
        float cur = graph_get_edge_weight(u->graph, u->previous_node_id, node_id);
        graph_set_edge_weight(u->graph, u->previous_node_id, node_id,
                              cur + u->learning_rate, ctx, scr);
    }

    /* Decay edges from previous node */
    if (u->previous_node_id >= 0 && u->previous_node_id < MAX_NODES) {
        Edge **pp = &u->graph->adjacency[u->previous_node_id];
        while (*pp) {
            Edge *e = *pp;
            float nw = e->weight - u->decay_rate;
            if (nw <= 0.0f) {
                *pp = e->next;
                free(e->context_sdr);
                int num_to_free = e->num_exemplars;
                if (num_to_free > MAX_EXEMPLARS) num_to_free = MAX_EXEMPLARS;
                for (int j = 0; j < num_to_free; j++)
                    free(e->scr_exemplars[j]);
                free(e);
            } else {
                e->weight = nw;
                pp = &e->next;
            }
        }
    }

    u->previous_node_id = node_id;
    u->update_count++;
    /* Neuroplasticity check disabled to improve induction performance */
    // if (u->neuroplastic_enabled && (u->update_count % 5 == 0))
    //     updater_neuroplasticity_check(u, node_id);

    return surprise;
}
