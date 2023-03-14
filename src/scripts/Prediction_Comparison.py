import pandas as pd

baseline_bert_data = pd.read_csv("../../data/preds_withtokens_submit.csv", delimiter=",")
# model_data = pd.read_csv("../data/bert_tf_predicts.csv", delimiter=",")
model_data = pd.read_csv("../../data/Roberta_Without_Tokens.csv", delimiter=",")

# print(baseline_bert_data.head())
# print(model_data.head())

base_relation_count = baseline_bert_data.Relation.value_counts()
print(base_relation_count)
model_relation_count = model_data.predicted_relations.value_counts()
print(model_relation_count)

# Checking Count Mismatch

print(sum(base_relation_count))
print(sum(model_relation_count))

total_difference = 0
for i in range(len(model_relation_count)):
    for j in range(len(base_relation_count)):
        if base_relation_count.index[j] == model_relation_count.index[i]:
            difference = abs(base_relation_count[j] - model_relation_count[i])
            print(f' Relation: {base_relation_count.index[j]} --- Base Count: {base_relation_count[j]} --- Model Count: {model_relation_count[i]} --- Difference: {difference}')
            total_difference = total_difference + difference

print(f'Total Difference is : {total_difference}')
print(f'Percentage of Difference is : {(total_difference/sum(model_relation_count))*100}%')
