import numpy as np

from random import shuffle

# Description: The sorted ranked data is provided as an argument along with an equal length list of 1's and 0's 
#              that indicate whether the corresponding position is included in the enrichment plot
def get_enrichment_score(ranked_data, enrichment_bars):

    N   = len(ranked_data)
    N_H = sum(enrichment_bars)
    p = 0
    ES = 0

    for i in range(0, len(ranked_data)):

        P_hit  = None
        P_miss = None
        N_R    = 0

        for j in range(0, i + 1):
    
            if enrichment_bars[j] == 1:
                # Hit
                r_J = abs(ranked_data[j]['value']) ** p
                N_R += r_J
                P_hit = r_J / N_R if P_hit == None else P_hit + (r_J / N_R)

            else:
                # Miss
                P_miss = 1 / (N - N_H) if P_miss == None else P_miss + (1 /(N - N_H))


            if (P_hit != None) and (P_miss != None) and (P_hit - P_miss > ES):
                ES = P_hit - P_miss

    return ES

# Description: Performs GSEA KS test then returns ks stat and p value
def ks_test(ranked_data, enrichment_bars):


    bootstrap_num = 1000
    bootstrap_enrichment_bars = enrichment_bars.copy()

    ES = get_enrichment_score(ranked_data, enrichment_bars)
    ES_Null = []

    for i in range(bootstrap_num):
        shuffle(bootstrap_enrichment_bars)
        ES_Null.append(get_enrichment_score(ranked_data, bootstrap_enrichment_bars))

    p_value = sum([1 if x >= ES else 0 for x in ES_Null]) / bootstrap_num

    return ES, p_value
