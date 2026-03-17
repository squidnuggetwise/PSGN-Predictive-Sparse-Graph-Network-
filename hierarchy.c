#include "psgn.h"

/* ═══════════════  WorkingMemory (XOR-binding)  ═══════════════ */

void wm_init(WorkingMemory *wm, int sdr_size) {
    wm->sdr_size   = sdr_size;
    wm->buffer_sdr = sdr_create(sdr_size);
    wm->seed_sdr   = sdr_create(sdr_size);
    /* High-entropy seed (50% density) for VSA orthogonality */
    for (int i = 0; i < sdr_size; i++) {
        wm->seed_sdr[i] = (rand() % 2);
    }
    sdr_copy(wm->buffer_sdr, wm->seed_sdr, sdr_size);
    wm->held_count = 0;
    for (int i = 0; i < 16; i++) wm->pos_sdrs[i] = NULL;
}

void wm_free(WorkingMemory *wm) {
    sdr_free(wm->buffer_sdr);
    sdr_free(wm->seed_sdr);
}

void wm_hold(WorkingMemory *wm, const uint8_t *sdr) {
    /* VSA Prime-Shifted Binding: 
     * Shifting by a large prime (10007) per token index mathematically isolates
     * digits at different positions, preventing 'Equation Aliasing'. */
    sdr_or_shifted_inplace(wm->buffer_sdr, sdr, wm->sdr_size, (wm->held_count + 1) * 10007);
    wm->held_count++;
}

uint8_t *wm_read(const WorkingMemory *wm) { 
    if (wm->held_count == 0) return NULL;
    return wm->buffer_sdr; 
}

void wm_clear(WorkingMemory *wm) {
    sdr_clear(wm->buffer_sdr, wm->sdr_size);
    /* Superposition starts with a clean slate */
    wm->held_count = 0;
}

/* ═══════════════  HierarchicalNetwork  ═══════════════ */

void hierarchy_init(HierarchicalNetwork *h,
                    int l1_size, double l1_sp,
                    int l2_size, double l2_sp, float decay) {
    sdr_encoder_init(&h->l1_encoder, l1_size, l1_sp);
    graph_init(&h->l1_graph, l2_size, l1_size);
    updater_init(&h->l1_updater, &h->l1_graph, &h->l1_encoder, 0.1f, decay);

    sdr_encoder_init(&h->l2_encoder, l2_size, l2_sp);
    graph_init(&h->l2_graph, l2_size, l2_size);
    updater_init(&h->l2_updater, &h->l2_graph, &h->l2_encoder, 0.1f, decay * 0.5f);

    wm_init(&h->scratchpad, l1_size);
    h->l1_seq_len     = 0;
    h->current_l2_sdr = sdr_create(l2_size);
}


void hierarchy_free(HierarchicalNetwork *h) {
    sdr_encoder_free(&h->l1_encoder);
    graph_free(&h->l1_graph);
    sdr_encoder_free(&h->l2_encoder);
    graph_free(&h->l2_graph);
    wm_free(&h->scratchpad);
    free(h->current_l2_sdr);
}

void hierarchy_get_dynamic_context(HierarchicalNetwork *h, uint8_t *out) {
    int l1sz = h->l1_encoder.sdr_size;
    int l2sz = h->l2_encoder.sdr_size;
    memset(out, 0, l2sz);
    uint8_t *mapped = (uint8_t *)malloc(l2sz);

    for (int i = 0; i < h->l1_seq_len; i++) {
        uint8_t *l1s = sdr_encoder_encode(&h->l1_encoder, h->l1_seq_buf[i]);
        memset(mapped, 0, l2sz);
        if (l1sz < l2sz) {
            int tiles = l2sz / l1sz;
            for (int t = 0; t < tiles; t++)
                memcpy(mapped + t * l1sz, l1s, l1sz);
        } else {
            memcpy(mapped, l1s, l2sz);
        }
        sdr_or_inplace(out, mapped, l2sz);
    }
    if (h->current_l2_sdr)
        sdr_or_inplace(out, h->current_l2_sdr, l2sz);
    free(mapped);
}

int hierarchy_process_token(HierarchicalNetwork *h, const char *word,
                            char *concept_out, int *trigger_out, int train) {
    *trigger_out = (strcmp(word, "=") == 0);
    int concept_formed = 0;
    if (concept_out) concept_out[0] = '\0';

    int l1sz = h->l1_encoder.sdr_size;
    int l2sz = h->l2_encoder.sdr_size;

    /* 1. Encode at L1 */
    uint8_t *l1_sdr_ref = sdr_encoder_encode(&h->l1_encoder, word);
    int l1_id = sdr_encoder_get_node_id(&h->l1_encoder, word);
    uint8_t *l1_sdr = sdr_create(l1sz);
    sdr_copy(l1_sdr, l1_sdr_ref, l1sz);

    /* Add to sequence buffer */
    if (h->l1_seq_len < MAX_SEQ_BUF) {
        strncpy(h->l1_seq_buf[h->l1_seq_len], word, MAX_TOKEN_LEN - 1);
        h->l1_seq_buf[h->l1_seq_len][MAX_TOKEN_LEN - 1] = '\0';
        h->l1_seq_len++;
    }

    /* 2. Read scratchpad BEFORE holding current token */
    uint8_t *scr = wm_read(&h->scratchpad);

    /* 3. Get dynamic L2 context */
    uint8_t *dyn = (uint8_t *)malloc(l2sz);
    hierarchy_get_dynamic_context(h, dyn);

    /* 4. Update L1 (ONLY IF TRAINING) */
    int surprise = 0;
    if (train) {
        surprise = updater_update(&h->l1_updater, l1_sdr, l1_id, dyn, scr);
    }

    /* 5. Hold token (Recursive History Recording) 
     * Record ALL tokens (Operands, Scan Markers, and Results) to ensure a unique cumulative history. */
    wm_hold(&h->scratchpad, l1_sdr);
    sdr_free(l1_sdr);

    /* 6. Boundary detection */
    if (surprise && h->l1_seq_len > 1) {
        concept_formed = 1;
        if (concept_out) {
            concept_out[0] = '\0';
            for (int i = 0; i < h->l1_seq_len; i++) {
                if (i > 0) strcat(concept_out, "_");
                strncat(concept_out, h->l1_seq_buf[i], 1023 - strlen(concept_out));
            }
        }

        uint8_t *l2_sdr = sdr_create(l2sz);
        uint8_t *mapped = (uint8_t *)malloc(l2sz);
        for (int i = 0; i < h->l1_seq_len; i++) {
            uint8_t *l1s = sdr_encoder_encode(&h->l1_encoder, h->l1_seq_buf[i]);
            memset(mapped, 0, l2sz);
            if (l1sz < l2sz) {
                int tiles = l2sz / l1sz;
                for (int t = 0; t < tiles; t++)
                    memcpy(mapped + t * l1sz, l1s, l1sz);
            } else {
                memcpy(mapped, l1s, l2sz);
            }
            sdr_or_inplace(l2_sdr, mapped, l2sz);
        }
        free(mapped);

        int l2_id = sdr_encoder_get_node_id(&h->l2_encoder, concept_out);
        free(h->l2_encoder.token_sdrs[l2_id]);
        h->l2_encoder.token_sdrs[l2_id] = l2_sdr;
        updater_update(&h->l2_updater, l2_sdr, l2_id, NULL, NULL);

        free(h->current_l2_sdr);
        h->current_l2_sdr = sdr_create(l2sz);
        sdr_copy(h->current_l2_sdr, l2_sdr, l2sz);

        strncpy(h->l1_seq_buf[0], word, MAX_TOKEN_LEN - 1);
        h->l1_seq_buf[0][MAX_TOKEN_LEN - 1] = '\0';
        h->l1_seq_len = 1;
    }

    free(dyn);
    return concept_formed;
}

void hierarchy_reset_sequence(HierarchicalNetwork *h) {
    h->l1_seq_len = 0;
    h->l1_updater.previous_node_id = -1;
    h->l2_updater.previous_node_id = -1;
    if (h->current_l2_sdr) {
        sdr_clear(h->current_l2_sdr, h->l2_encoder.sdr_size);
    }
}

const char *hierarchy_get_l2_prediction(HierarchicalNetwork *h) {
    if (h->l2_updater.previous_node_id < 0) return NULL;
    Prediction preds[16];
    int np = graph_get_predictions(&h->l2_graph, h->l2_updater.previous_node_id,
                                   NULL, NULL, preds, 16);
    if (np > 0) return h->l2_encoder.id_to_token[preds[0].node_id];
    return NULL;
}
