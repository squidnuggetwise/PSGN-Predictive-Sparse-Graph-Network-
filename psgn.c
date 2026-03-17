#include "psgn.h"

/* ═══════════════  Tokenizer  ═══════════════ */

int psgn_tokenize(const char *text, char ***tokens_out) {
    int cap = 1024, n = 0;
    char **toks = (char **)malloc(cap * sizeof(char *));
    const char *p = text;
    while (*p) {
        if (isspace((unsigned char)*p)||*p==','||*p==';'||*p=='('||*p==')'||
            *p=='['||*p==']'||*p=='{'||*p=='}'||*p==':'||*p=='!'||
            *p=='?'||*p=='"'||*p=='\'') { p++; continue; }
        if (*p=='='||*p=='+'||*p=='-'||*p=='*'||*p=='/'||*p=='.') {
            if(n>=cap){cap*=2;toks=(char**)realloc(toks,cap*sizeof(char*));}
            char buf[2]={*p,'\0'}; toks[n++]=strdup(buf); p++; continue;
        }
        if (isalnum((unsigned char)*p) || *p == '_' || *p == ':') {
            const char *s=p;
            while(*p && (isalnum((unsigned char)*p) || *p == '_' || *p == ':')) p++;
            int len=(int)(p-s); if(len>=MAX_TOKEN_LEN) len=MAX_TOKEN_LEN-1;
            if(n>=cap){cap*=2;toks=(char**)realloc(toks,cap*sizeof(char*));}
            char *t=(char*)malloc(len+1); memcpy(t,s,len); t[len]='\0';
            toks[n++]=t; continue;
        }
        p++;
    }
    *tokens_out = toks;
    return n;
}

static void free_tokens(char **tokens, int n) {
    for (int i = 0; i < n; i++) free(tokens[i]);
    free(tokens);
}

/* ═══════════════  PSGN Create/Free  ═══════════════ */

PSGN *psgn_create(int l1_size, double l1_sp, int l2_size, double l2_sp, float decay) {
    PSGN *p = (PSGN *)calloc(1, sizeof(PSGN));
    hierarchy_init(&p->hierarchy, l1_size, l1_sp, l2_size, l2_sp, decay);
    return p;
}

void psgn_free(PSGN *p) {
    if (!p) return;
    hierarchy_free(&p->hierarchy);
    free(p);
}

/* ═══════════════  read_text (with deferred clearing)  ═══════════════ */

void psgn_read_text(PSGN *p, const char *text) {
    char **words; int n = psgn_tokenize(text, &words);
    for (int i = 0; i < n; i++) {
        char concept[1024]; int trigger=0;
        concept[0] = '\0';
        /* Process ALL tokens including '.' to learn the Stop-State */
        hierarchy_process_token(&p->hierarchy, words[i], concept, &trigger, 1);

        /* AFTER processing the delimiter, wipe context for the next equation */
        if (strcmp(words[i], ".") == 0) {
            wm_clear(&p->hierarchy.scratchpad);
            hierarchy_reset_sequence(&p->hierarchy);
            continue;
        }

        if (i>0 && i%1000==0) {
            int *l1n=(int*)malloc(MAX_NODES*sizeof(int));
            int nl1=graph_get_all_nodes(&p->hierarchy.l1_graph,l1n,MAX_NODES);
            printf("Progress %d/%d: L1 Nodes: %d\n",i,n,nl1);
            free(l1n);
        }
    }
    free_tokens(words, n);
}

/* ═══════════════  generate_text (fully general, no arithmetic branch)  ═══════════════ */

char *psgn_generate_text(PSGN *p, const char *seq, int length, int full) {
    char **words; int nw = psgn_tokenize(seq, &words);
    int gen_cap = nw+length+1;
    char **generated = (char **)malloc(gen_cap * sizeof(char *));
    int gen_n = 0;
    for (int i=0;i<nw;i++) generated[gen_n++]=strdup(words[i]);

    printf("\n--- Priming State with Sequence: '%s' ---\n", seq);
    wm_clear(&p->hierarchy.scratchpad);
    hierarchy_reset_sequence(&p->hierarchy);

    /* Prime the entire sequence to align VSA signature */
    for (int i=0; i<nw; i++) {
        char c[MAX_TOKEN_LEN*4]; int t;
        hierarchy_process_token(&p->hierarchy, words[i], c, &t, 0);
    }

    int cur_id = sdr_encoder_get_node_id(&p->hierarchy.l1_encoder, words[nw-1]);
    float temperature = 0.0f;
    float *fatigue = (float*)calloc(MAX_NODES, sizeof(float));

    printf("\n--- Generating ---\n");
    for (int step=0; step<length; step++) {
        uint8_t *dyn=(uint8_t*)malloc(p->hierarchy.l2_encoder.sdr_size);
        hierarchy_get_dynamic_context(&p->hierarchy, dyn);
        uint8_t *scr=wm_read(&p->hierarchy.scratchpad);

        Prediction preds[512];
        int np = graph_get_predictions(&p->hierarchy.l1_graph, cur_id,
                                       dyn, scr, preds, 512);
        free(dyn);
        if (np==0) break;

        int next_id = preds[0].node_id;
        float weight = preds[0].score;

        char next_word[MAX_TOKEN_LEN];
        strncpy(next_word, p->hierarchy.l1_encoder.id_to_token[next_id], MAX_TOKEN_LEN-1);
        next_word[MAX_TOKEN_LEN-1] = '\0';
        
        if (strcmp(next_word, ".") == 0) break;

        printf("   Predicted '%s' (Score: %.2f)\n", next_word, weight);
        generated[gen_n++]=strdup(next_word);

        /* Feedback: update Markov context with our own prediction, 
         * but SKIP scratchpad XOR (record_in_wm=0) to prevent history drift. */
        char c[1024]; int t;
        hierarchy_process_token(&p->hierarchy, next_word, c, &t, 0);
        cur_id = next_id;
    }

    int si=full?0:nw; int tl=0;
    for(int i=si;i<gen_n;i++) tl+=(int)strlen(generated[i])+1;
    char *result=(char*)malloc(tl+1); result[0]='\0';
    for(int i=si;i<gen_n;i++){if(i>si)strcat(result," ");strcat(result,generated[i]);}
    for(int i=0;i<gen_n;i++)free(generated[i]); free(generated);
    free(fatigue);
    free_tokens(words,nw);
    return result;
}
