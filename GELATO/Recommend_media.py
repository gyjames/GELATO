import os
import json
import argparse
import yaml
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pointbiserialr
from utils import merge, min_max_normalize

def check_isolation_match(query_id, recommend_df, medium_info, isolation_df):
    isolation = isolation_df[isolation_df['query_id'] == query_id]['isolation_source'].values[0]
    match_list = []
    for idx, row in recommend_df.iterrows():
        media = row['Medium']
        if media in medium_info:
            media_isolation = medium_info[media].get('Isolation Sources', [])
            if isolation in media_isolation:
                match_list.append('yes')
            else:
                match_list.append('no')
        else:
            match_list.append('no')
    recommend_df['Isolation Match'] = match_list
    return recommend_df

def recommend_medium(config):
    output_dir = config['output_dir']
    genome_input_dir = config['genome_input_dir']
    
    r_m_path = os.path.join(output_dir, 'intermediate_r_m.csv')
    if not os.path.exists(r_m_path):
        raise FileNotFoundError(f"Result from Step 1 not found at {r_m_path}. Please run step1_predict.py first.")
    r_m = pd.read_csv(r_m_path)

    species2medium = json.load(open(config['species2medium_json']))
    medium_info = json.load(open('/data/workdir/zhangyk/gelato/resources/medium.json'))
    map_dict = json.load(open(config['map_dict_json']))
    medium_vec_df = pd.read_csv(config['medium_vec_csv'])
    medium_cluster_df = pd.read_csv(config['medium_cluster_csv'])
    temp_dir = os.path.join(output_dir, "temp")
    medium_output_dir = os.path.join(output_dir, "results")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(medium_output_dir, exist_ok=True)

    print("Generating Mash sketch for each .fna file")
    for fna_file in os.listdir(genome_input_dir):
        if fna_file.endswith((".fa", ".fna")):
            fna_path = os.path.join(genome_input_dir, fna_file)
            sketch_output = os.path.join(temp_dir, f"{os.path.splitext(fna_file)[0]}.msh")
            subprocess.run([
                "mash", "sketch",
                "-k", "31", "-s", "10000",
                "-o", sketch_output, fna_path
            ], check=True, capture_output=True, text=True)

    print("Merging all generated .msh files")
    merged_sketch = os.path.join(temp_dir, "genome_sketches.msh")
    msh_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".msh")]
    if not msh_files:
        raise FileNotFoundError(f"No .msh files found in {temp_dir}.")
    subprocess.run(["mash", "paste", merged_sketch] + msh_files, check=True, capture_output=True, text=True)

    print("Calculating Mash distances between all genomes")
    mash_distance_output = os.path.join(temp_dir, "new_vs_db_distances.tab")
    with open(mash_distance_output, 'w') as f:
        subprocess.run([
            "mash", "dist",
            config['sketch_db'],
            merged_sketch,
        ], stdout=f)

    query_refs = {}
    db_base_path = config.get('medium_genome_db', '')
    
    with open(mash_distance_output, 'r') as f:
        for line in f:
            fields = line.strip().split()
            ref_filename = os.path.basename(fields[0])
            ref_genome = os.path.join(db_base_path, ref_filename)
            # ref_genome = fields[0]
            query_genome = fields[1]
            mash_distance = float(fields[2])
            if ref_genome == query_genome:
                continue
            if query_genome not in query_refs:
                query_refs[query_genome] = []
            if mash_distance < 0.25:
                query_refs[query_genome].append(ref_genome)

    print("Running fastANI for each query genome")
    for query_genome, ref_genomes in tqdm(query_refs.items(), total=len(query_refs), desc="Running fastANI"):
        if not ref_genomes:
            continue
        query_basename = os.path.basename(query_genome)
        ref_list_file = os.path.join(temp_dir, f"{query_basename}_ref_list.txt")
        with open(ref_list_file, 'w') as f_ref_list:
            for ref_path in ref_genomes:
                f_ref_list.write(f"{ref_path}\n")
        fastani_output_file = os.path.join(temp_dir, f"{query_basename}_fastani_output.txt")
        fastani_cmd = [
            "fastANI", 
            "-q", query_genome, 
            "--rl", ref_list_file, 
            "-o", fastani_output_file, 
            "-t", str(config['threads'])
        ]
        try:
            subprocess.run(fastani_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running fastANI for {query_basename}: {e.stderr}")

    combined_file_path = os.path.join(temp_dir, "mash_fastani_result.txt")
    with open(combined_file_path, 'w') as outfile:
        for filename in os.listdir(temp_dir):
            if filename.endswith('_fastani_output.txt'):
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'r') as infile:
                    lines = infile.readlines()
                    if not lines: continue
                    outfile.writelines(lines)

    def extract_name_local(text):
        last_part = text.split('/')[-1]
        return last_part[:last_part.rfind('.')]

    if os.path.exists(combined_file_path) and os.path.getsize(combined_file_path) > 0:
        ani_result = pd.read_csv(combined_file_path, sep="\t", header=None, names=["Query", "Reference", "ANI", "Matches", "Total_Fragments"])
        ani_result["Query"] = ani_result["Query"].apply(extract_name_local)
        ani_result["Reference"] = ani_result["Reference"].apply(lambda x: x.split("/")[-1][:-4])
        ref_species = [map_dict.get(i, '') for i in ani_result['Reference']]
        ani_result['Ref_species'] = ref_species
        ani_result.to_csv(os.path.join(output_dir, "fastani_results.csv"), index=False)
    else:
        print("Warning: No FastANI results generated. Creating empty DataFrame.")
        ani_result = pd.DataFrame(columns=["Query", "Reference", "ANI", "Ref_species"])

    nutrients_pred = r_m.copy()
    if 'Query' in nutrients_pred.columns:
        query_names = nutrients_pred['Query'].values
        query_matrix = nutrients_pred.drop(columns=['Query']).values
    else:
        raise ValueError("r_m csv missing 'Query' column")

    medium_matrix = medium_vec_df.iloc[:, 1:-1].values
    medium_names = medium_vec_df['Medium'].values
    correlations = np.empty((query_matrix.shape[0], medium_matrix.shape[0]))

    for i in tqdm(range(query_matrix.shape[0]), desc="Recommend medium"):
        current_query_vec = query_matrix[i, :]
        for j in range(medium_matrix.shape[0]):
            current_medium_vec = medium_matrix[j, :]
            corr, _ = pointbiserialr(current_medium_vec, current_query_vec)
            correlations[i, j] = corr

    cluster_dict = medium_cluster_df.set_index('Medium')['small_cluster_id'].to_dict()
    isolation_df = pd.read_csv(os.path.join(config['genome_input_dir'], 'isolation_info.csv'))
    
    
    for i, query_name in enumerate(query_names):
        query_correlations = correlations[i, :]
        df = pd.DataFrame({'Medium': medium_names, 'Point Biserial Correlation': query_correlations})
        predict_result = df.sort_values(by='Point Biserial Correlation', ascending=False)
        predict_result.dropna(inplace=True)
        
        num_df = medium_vec_df[['Medium', 'num_of_compounds']]
        predict_result = pd.merge(predict_result, num_df, on='Medium', how='left')
        
        λ_r = predict_result.copy()
        λ_r['Score'] = λ_r['Point Biserial Correlation'] - config['lambda_reg'] * λ_r['num_of_compounds']
        λ_r = λ_r[λ_r['Score'] > 0]
        λ_r.sort_values(by='Score', ascending=False, inplace=True)
        
        pred_dict = dict(zip(λ_r['Medium'].tolist(), λ_r['Score'].tolist()))
        
        ani_pred = ani_result[ani_result['Query'] == query_name]
        ani_medium, ani_ls = [], []
        if not ani_pred.empty:
            for idx, row in ani_pred.iterrows():
                s = row['Ref_species']
                ani = row['ANI'] * 0.01
                ani_medium.extend(species2medium.get(s, []))
                ani_ls.extend([ani] * len(species2medium.get(s, [])))
        
        ani_pred_dict = dict(zip(ani_medium, ani_ls))
        result = merge(pred_dict, ani_pred_dict)
        
        sorted_items = sorted(result.items(), key=lambda item: item[1], reverse=True)
        result = dict(sorted_items)
        result = min_max_normalize(result)
        
        result_df = pd.DataFrame({'Medium': result.keys(), 'Recommendation Score': result.values()})
        result_df = result_df.head(config['top_k'])
        result_df['Cluster'] = [cluster_dict.get(m, -1) for m in result_df['Medium'].tolist()]
        result_df = check_isolation_match(query_name, result_df, medium_info, isolation_df)
        
        result_df.to_csv(os.path.join(medium_output_dir, f"{query_name}_recommendation.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Recommend media.")
    parser.add_argument("--config", required=True, help="Path to config.yaml file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found.")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    recommend_medium(config)
