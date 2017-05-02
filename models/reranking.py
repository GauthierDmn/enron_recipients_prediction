from  collections import Counter

def reranking(preds1,preds2):

    reranked_preds = {}
    for (k1 ,v1), (k2 ,v2) in zip(preds1.items(), preds2.items()):
        reranked_preds[k1] = [v1[0] ,[]]
        for l in range(len(v1[1])):
            temp_dict1 = {}
            temp_dict2 = {}
            for i in range(len(v1[1][l])):
                a = v1[1][l].index(v1[1][l][i])
                b = v2[1][l].index(v2[1][l][i])
                temp_dict1[v1[1][l][i]] = 1 / (a + 1)
                temp_dict2[v2[1][l][i]] = 1 / (b + 1)

            for i in range(len(v1[1][l])):
                temp_dict1[v2[1][l][i]] = temp_dict1.get(v2[1][l][i], 0) + temp_dict2[v2[1][l][i]]

            d = Counter(temp_dict1)
            preds = []
            for k, v in d.most_common():
                preds.append(k)

                reranked_preds[k1][1].append(preds)

    return reranked_preds