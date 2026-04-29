import json
import os
import random
import re
import pandas as pd


def tokenization(text):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '/', ':', ';', '<', '=', '>','\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    ls = [i.strip() for i in text.split()]
    for i, w in enumerate(ls):
        w = re.sub(r'\.$', '', w)
        ls[i] = w
    return ls


def reduce_genome_to_length(gene_list, target_length, rng):
    if len(gene_list) <= target_length:
        return gene_list.copy()

    result = gene_list.copy()

    while len(result) > target_length:
        genes_to_remove_now = len(result) - target_length
        start = rng.randint(0, len(result) - 1)
        max_segment_length = min(genes_to_remove_now, len(result) - start)

        if max_segment_length <= 0:
            del result[rng.randint(0, len(result) - 1)]
            continue

        segment_length = rng.randint(1, max_segment_length)
        del result[start:start + segment_length]

    return result


def get_incomplete_data(phenotype_id, vocab_path, seed=231500012563):
    rng = random.Random(seed)

    test_data_path = f"/data/benchmark_data/train_model/test_data_{phenotype_id}.csv"
    df = pd.read_csv(test_data_path)

    output_path = f"./simulate_incomplete_genomes/test_data/{phenotype_id}/seed_{seed}"
    os.makedirs(output_path, exist_ok=True)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    metadata = []

    for _, row in df.iterrows():
        path = row["path"]
        label = row[phenotype_id]
        assembly_accession = row["#assembly_accession_x"]

        with open(path, encoding="utf-8") as f:
            content = f.read()

        genome_tokens = tokenization(content)
        genome_tokens = [i for i in genome_tokens if i != "Cluster"]

        sentence = [vocab[gene] for gene in genome_tokens if gene in vocab]

        if len(sentence) <= 100:
            metadata.append({
                "assembly_accession": assembly_accession,
                "label": label,
                "status": "filtered_short",
                "original_token_length": len(sentence)
            })
            continue

        original_length = len(sentence)
        cumulative_sentence = sentence.copy()

        for target_percentage in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]:
            target_length = int(original_length * target_percentage / 100)

            cumulative_sentence = reduce_genome_to_length(
                cumulative_sentence,
                target_length,
                rng
            )

            file_name = f"{assembly_accession}_{target_percentage}_seed{seed}.txt"
            file_path = os.path.join(output_path, file_name)

            with open(file_path, "w") as f_out:
                f_out.write(" ".join(map(str, cumulative_sentence)))

            metadata.append({
                "assembly_accession": assembly_accession,
                "label": label,
                "target_percentage": target_percentage,
                "original_token_length": original_length,
                "retained_token_length": len(cumulative_sentence),
                "retained_token_ratio": len(cumulative_sentence) / original_length,
                "file_name": file_name,
                "status": "kept"
            })

    pd.DataFrame(metadata).to_csv(
        os.path.join(output_path, "metadata.csv"),
        index=False
    )

    print(f"Processing complete for phenotype: {phenotype_id}, seed: {seed}")


df = pd.read_csv("phenotypes.csv")
phenotype_ids = df["Object"].tolist()

for phenotype_id in phenotype_ids:
    vocab_path = f"/GELATO/vocab/{phenotype_id}_w2i.json"

    for seed in [1, 2, 3, 4, 5]:
        get_incomplete_data(phenotype_id, vocab_path, seed=seed)