#include "psgn.h"

/* ═══════════════  HashMap  ═══════════════ */

static uint32_t djb2(const char *s) {
    uint32_t h = 5381;
    int c;
    while ((c = (unsigned char)*s++)) h = ((h << 5) + h) + c;
    return h;
}

void hashmap_init(HashMap *m) {
    memset(m->entries, 0, sizeof(m->entries));
    m->count = 0;
}

int hashmap_put(HashMap *m, const char *key, int value) {
    uint32_t idx = djb2(key) % HASHMAP_CAP;
    for (int i = 0; i < HASHMAP_CAP; i++) {
        uint32_t p = (idx + i) % HASHMAP_CAP;
        if (!m->entries[p].occupied) {
            strncpy(m->entries[p].key, key, MAX_TOKEN_LEN - 1);
            m->entries[p].key[MAX_TOKEN_LEN - 1] = '\0';
            m->entries[p].value = value;
            m->entries[p].occupied = 1;
            m->count++;
            return 0;
        }
        if (strcmp(m->entries[p].key, key) == 0) {
            m->entries[p].value = value;
            return 0;
        }
    }
    return -1;
}

int hashmap_get(const HashMap *m, const char *key, int *value) {
    uint32_t idx = djb2(key) % HASHMAP_CAP;
    for (int i = 0; i < HASHMAP_CAP; i++) {
        uint32_t p = (idx + i) % HASHMAP_CAP;
        if (!m->entries[p].occupied) return -1;
        if (strcmp(m->entries[p].key, key) == 0) {
            *value = m->entries[p].value;
            return 0;
        }
    }
    return -1;
}

/* ═══════════════  ScalarEncoder  ═══════════════ */

void scalar_encoder_init(ScalarEncoder *e, int sdr_size, int w,
                         double min_v, double max_v) {
    e->sdr_size    = sdr_size;
    e->w           = w;
    e->min_val     = min_v;
    e->max_val     = max_v;
    e->num_buckets = sdr_size - w + 1;
}

void scalar_encoder_encode(const ScalarEncoder *e, double value, uint8_t *out) {
    memset(out, 0, e->sdr_size);
    double val = value;
    if (val < e->min_val) val = e->min_val;
    if (val > e->max_val) val = e->max_val;

    double pct = 0.0;
    if (e->max_val != e->min_val)
        pct = (val - e->min_val) / (e->max_val - e->min_val);

    int start = (int)(pct * (e->num_buckets - 1));
    for (int i = start; i < start + e->w && i < e->sdr_size; i++)
        out[i] = 1;
}

/* ═══════════════  SDREncoder  ═══════════════ */

void sdr_encoder_init(SDREncoder *e, int sdr_size, double sparsity) {
    e->sdr_size        = sdr_size;
    e->sparsity        = sparsity;
    e->num_active_bits = (int)(sdr_size * sparsity);
    scalar_encoder_init(&e->scalar_enc, sdr_size, e->num_active_bits, 0.0, 100.0);

    hashmap_init(&e->token_to_id);
    memset(e->id_to_token, 0, sizeof(e->id_to_token));
    memset(e->token_sdrs, 0, sizeof(e->token_sdrs));
    memset(e->bit_freqs, 0, sizeof(e->bit_freqs));
    e->next_node_id = 0;
}

void sdr_encoder_free(SDREncoder *e) {
    for (int i = 0; i < e->next_node_id; i++) {
        free(e->token_sdrs[i]);
        free(e->bit_freqs[i]);
    }
}

static int is_numeric(const char *tok) {
    const char *p = tok;
    if (*p == '-') p++;
    if (!*p) return 0;
    int has_dot = 0;
    while (*p) {
        if (*p == '.') { if (has_dot) return 0; has_dot = 1; }
        else if (!isdigit((unsigned char)*p)) return 0;
        p++;
    }
    return 1;
}

uint8_t *sdr_encoder_encode(SDREncoder *e, const char *token) {
    int existing_id;
    if (hashmap_get(&e->token_to_id, token, &existing_id) == 0)
        return e->token_sdrs[existing_id];

    if (e->next_node_id >= MAX_NODES) return NULL;

    int id = e->next_node_id;
    uint8_t *sdr = sdr_create(e->sdr_size);

    int *indices = (int *)malloc(e->sdr_size * sizeof(int));
    sdr_random_sample(e->sdr_size, e->num_active_bits, indices);
    for (int i = 0; i < e->num_active_bits; i++)
        sdr[indices[i]] = 1;
    free(indices);

    e->token_sdrs[id] = sdr;
    hashmap_put(&e->token_to_id, token, id);
    strncpy(e->id_to_token[id], token, MAX_TOKEN_LEN - 1);
    e->id_to_token[id][MAX_TOKEN_LEN - 1] = '\0';
    e->next_node_id++;
    return sdr;
}

int sdr_encoder_get_node_id(SDREncoder *e, const char *token) {
    int id;
    if (hashmap_get(&e->token_to_id, token, &id) == 0)
        return id;
    sdr_encoder_encode(e, token);
    hashmap_get(&e->token_to_id, token, &id);
    return id;
}

void sdr_encoder_record_bits(SDREncoder *e, const char *token, const uint8_t *sdr) {
    int id;
    if (hashmap_get(&e->token_to_id, token, &id) != 0) return;
    if (!e->bit_freqs[id])
        e->bit_freqs[id] = (int *)calloc(e->sdr_size, sizeof(int));
    for (int i = 0; i < e->sdr_size; i++)
        if (sdr[i]) e->bit_freqs[id][i]++;
}

void sdr_encoder_mutate(SDREncoder *e, const char *a, const char *b, int bits) {
    int id_a, id_b;
    if (hashmap_get(&e->token_to_id, a, &id_a) != 0) return;
    if (hashmap_get(&e->token_to_id, b, &id_b) != 0) return;

    uint8_t *sa = e->token_sdrs[id_a];
    uint8_t *sb = e->token_sdrs[id_b];
    int sz = e->sdr_size;

    /* Find bits unique to A and unique to B */
    int ua[2048], ub[2048];
    int nua = 0, nub = 0;
    for (int i = 0; i < sz; i++) {
        if (sa[i] && !sb[i] && nua < 2048) ua[nua++] = i;
        if (sb[i] && !sa[i] && nub < 2048) ub[nub++] = i;
    }
    if (nua == 0 || nub == 0) return;

    int actual = bits;
    if (actual > nua) actual = nua;
    if (actual > nub) actual = nub;
    if (actual <= 0) return;

    /* Pick random bits to copy from A → B, and turn off from B */
    int *idx_a = (int *)malloc(nua * sizeof(int));
    int *idx_b = (int *)malloc(nub * sizeof(int));
    for (int i = 0; i < nua; i++) idx_a[i] = i;
    for (int i = 0; i < nub; i++) idx_b[i] = i;

    for (int i = 0; i < actual; i++) {
        int j = i + rand() % (nua - i);
        int t = idx_a[i]; idx_a[i] = idx_a[j]; idx_a[j] = t;
        j = i + rand() % (nub - i);
        t = idx_b[i]; idx_b[i] = idx_b[j]; idx_b[j] = t;
    }

    /* Apply mutation IN-PLACE to avoid invalidating external pointers */
    for (int i = 0; i < actual; i++) {
        sb[ua[idx_a[i]]] = 1;  /* copy bit from A */
        sb[ub[idx_b[i]]] = 0;  /* turn off a unique-to-B bit */
    }

    free(idx_a);
    free(idx_b);
}
