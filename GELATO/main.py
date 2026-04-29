import os
import sys
import shutil
import argparse
import yaml
import subprocess
from pathlib import Path
from Metabolite_utilization_predicte import predict_phenotypes
from Recommend_media import recommend_medium

def run_annotation(input_dir, output_dir, config):
    print("\n" + "="*50)
    print(">>> Step 1: Genome Annotation Running...")
    print("="*50)

    resources = config.get('annot_resources', {})
    cmd = [
        sys.executable, "annot_genome.py",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-t", str(config['threads']),
        "--ko_list", str(resources.get('ko_list')),
        "--profiles", str(resources.get('profiles')),
        "--cdb", str(resources.get('diamond_db')),
        "--kofam_exec", str(resources.get('kofam_exec')),
        "--tmp_dir", str(resources.get('tmp_dir', '/dev/shm/tmp'))
    ]
    
    if resources.get('mem_cdb', False): cmd.append("--mem_cdb")
    if resources.get('mem_profiles', False): cmd.append("--mem_profiles")

    try:
        subprocess.run(cmd, check=True)
        print(">>> Annotation finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during annotation step. Exit code: {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Microbial Phenotype Prediction & Media Recommendation Pipeline"
    )

    parser.add_argument('-i', '--input', required=True, type=Path, help='Input directory containing genome files (.fna/.fa)')
    parser.add_argument('-o', '--output', required=True, type=Path, help='Main output directory')
    parser.add_argument('-c', '--config', required=True, type=Path, help='Path to config.yaml')

    parser.add_argument('--mode', choices=['phenotype', 'full'], default='full',
                        help="Mode: 'phenotype' (Predict phenotype only) or 'full' (Predict + Recommend Media)")

    args = parser.parse_args()


    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    input_dir = args.input.resolve()
    base_output_dir = args.output.resolve()
    annot_output_dir = base_output_dir / "01_annotations"  
    predict_output_dir = base_output_dir / "02_predictions" 

    os.makedirs(annot_output_dir, exist_ok=True)
    os.makedirs(predict_output_dir, exist_ok=True)


    run_annotation(input_dir, annot_output_dir, config)

    config['genome_input_dir'] = str(input_dir)      
    config['annotation_dir'] = str(annot_output_dir / "results") 
    config['output_dir'] = str(predict_output_dir)  
    
    
    print("\n" + "="*50)
    print(">>> Step 2: Phenotype Prediction Running...")
    print("="*50)
    
    try:
        predict_phenotypes(config)
        print(">>> Phenotype prediction finished.")
    except Exception as e:
        print(f"Error during phenotype prediction: {e}")
        sys.exit(1)

    if args.mode == 'full':
        print("\n" + "="*50)
        print(">>> Step 3: Media Recommendation Running...")
        print("="*50)
        
        try:
            intermediate_file = predict_output_dir / 'intermediate_r_m.csv'
            if not intermediate_file.exists():
                print(f"Error: Step 2 did not generate {intermediate_file}. Cannot proceed to recommendation.")
                sys.exit(1)
            
            recommend_medium(config)
            print(">>> Media recommendation finished.")
            
            if intermediate_file.exists():
                try:
                    os.remove(intermediate_file)
                    print(f"   [Clean] Removed file: {intermediate_file.name}")
                except OSError as e:
                    print(f"   [Warning] Failed to remove {intermediate_file.name}: {e}")

            temp_dir = predict_output_dir / "temp"
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    print(f"   [Clean] Removed directory: {temp_dir.name}")
                except OSError as e:
                    print(f"   [Warning] Failed to remove temp directory: {e}")
                    
                    
        except Exception as e:
            print(f"Error during media recommendation: {e}")
            sys.exit(1)
    
    print("\n" + "="*50)
    print(f"Pipeline Completed! Results are in: {base_output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()