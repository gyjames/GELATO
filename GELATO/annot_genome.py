import subprocess
import sys
import argparse
from pathlib import Path
import pandas as pd 
import time 
import os
import shutil
import atexit


def run_command(cmd):
    cmd_str = ' '.join(str(x) for x in cmd)
    print(f"[Run]: {cmd_str}", flush=True) 
    my_env = os.environ.copy()
    
    try:
        subprocess.run(
            cmd, 
            check=True, 
            text=True, 
            stdin=subprocess.DEVNULL, 
            env=my_env,
            close_fds=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd_str}", flush=True)
        print(f"Return code: {e.returncode}", flush=True)
        sys.exit(1)
        

def load_path_to_memory(src_path):
    """
    Copy a file OR a directory to /dev/shm.
    Returns the path in memory.
    """
    src_path = Path(src_path)
    name = src_path.name
    shm_path = Path("/dev/shm") / name
    
    print(f"[System] Loading {name} into memory (/dev/shm)...")
    start_load = time.time()
    
    if shm_path.exists():
        print(f"[System] {name} already in memory. Skipping copy.")
        return shm_path

    try:
        if src_path.is_dir():
            shutil.copytree(src_path, shm_path)
        else:
            shutil.copy2(src_path, shm_path)
    except Exception as e:
        print(f"Error loading to memory: {e}")
        print("Falling back to disk path.")
        return src_path

    print(f"[System] Loaded in {time.time() - start_load:.2f}s")
    return shm_path

def cleanup_memory_path(path):
    if path and path.exists() and str(path).startswith("/dev/shm"):
        print(f"[System] Cleaning up memory: {path}")
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError as e:
            print(f"Error cleaning up: {e}")


def get_gene_ids_from_faa(file_path):
    gene_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                header = line.strip()
                if len(header) > 1:
                    gene_id = header[1:].split()[0]
                    gene_ids.append(gene_id)
    return gene_ids

def parse_kofam_output(file_path, threshold=1e-6):
    if not os.path.exists(file_path):
        print(f"Warning: KofamScan output not found at {file_path}")
        return {}

    try:
        df = pd.read_csv(
            file_path, 
            sep='\t', 
            comment='#', 
            header=None, 
            usecols=[1, 2, 5], 
            names=['gene_id', 'ko_id', 'e_value'],
            dtype={'gene_id': str, 'ko_id': str} 
        )

        df['e_value'] = pd.to_numeric(df['e_value'], errors='coerce')
        df = df.dropna(subset=['e_value'])
        df = df[df['e_value'] < threshold]
        df = df.sort_values('e_value')
        df = df.drop_duplicates('gene_id', keep='first')

        return df.set_index('gene_id')['ko_id'].to_dict()

    except pd.errors.EmptyDataError:
        return {}
    except Exception as e:
        print(f"Error parsing KofamScan output: {e}")
        return {}


def get_best_annotation_diamond(file_path):
    try:
        df = pd.read_csv(
            file_path, 
            sep=r'\s+', 
            header=None, 
            comment='#', 
            engine='python'
        )
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    except pd.errors.EmptyDataError:
        print("Error: The file is empty or only contains comments.")
        return {}
    
    df.rename(columns={0: 'Query_ID', 1: 'Target_ID', 10: 'E_value'}, inplace=True)
    df['E_value'] = pd.to_numeric(df['E_value'], errors='coerce')
    best_hit_indices = df.groupby('Query_ID')['E_value'].idxmin()
    best_hits_df = df.loc[best_hit_indices]
    final_annotations = best_hits_df.set_index('Query_ID')['Target_ID'].to_dict()
    
    return final_annotations


def pipeline(input_genome, ko_list, profiles_path, kofam_exec, diamond_db_path, threads, final_dir, inter_dir, tmp_root):
    """
    Run pipeline and return breakdown times.
    Changed: output_path removed, replaced by final_dir and inter_dir
    """
    
    total_start = time.time()

    input_path = Path(input_genome)
    final_dir = Path(final_dir)
    inter_dir = Path(inter_dir)
    
    sample_name = input_path.stem 
    kofam_tmp_dir = Path(tmp_root) / sample_name
    
    print(f"\n>>> Processing sample: {sample_name}")

    prokka_outdir = inter_dir / f"prokka_{sample_name}"
    prokka_faa = prokka_outdir / f"{sample_name}.faa"
    kofam_output_file = inter_dir / f"{sample_name}_kofam.tsv"
    diamond_output_cluster = inter_dir / f"{sample_name}_vs_cdb.tsv"

    # --- 1. Prokka ---
    t_start = time.time()
    if prokka_faa.exists():
        print(f"   [Skip] Prokka output already exists.")
    else:
        cmd_prokka = [
            "prokka",
            "--outdir", str(prokka_outdir),
            "--prefix", sample_name, 
            "--cpus", str(threads),
            "--force", 
            str(input_path),
            "--norrna",
            "--notrna",
            "--compliant"
        ]
        run_command(cmd_prokka)
    time_prokka = time.time() - t_start

    # --- 2. KofamScan (HMM) ---
    t_start = time.time()
    print(f"   Starting KofamScan (HMM): {sample_name}")
    if kofam_output_file.exists():
        print(f"   [Skip] KofamScan output already exists.")
    else:
        kofam_tmp_dir.mkdir(parents=True, exist_ok=True)
        cmd_kofam = [
            kofam_exec,
            "-f", "detail-tsv",
            "--ko-list", str(ko_list),
            "--profile", str(profiles_path),
            "--tmp-dir", str(kofam_tmp_dir),
            "-o", str(kofam_output_file),
            "--cpu", str(threads),
            str(prokka_faa)
        ]
        try:
            run_command(cmd_kofam)
        finally:
            if kofam_tmp_dir.exists():
                shutil.rmtree(kofam_tmp_dir, ignore_errors=True)
    time_hmm = time.time() - t_start


    # --- 3. Diamond (Cluster) ---
    t_start = time.time()
    
    if not diamond_output_cluster.exists():
        cmd_diamond = [
            "diamond", "blastp", 
            "-q", str(prokka_faa), 
            "-d", str(diamond_db_path),
            "-o", str(diamond_output_cluster), 
            "-p", str(threads), 
            "--evalue", "1e-6"
                    ]
        run_command(cmd_diamond)
    time_diamond = time.time() - t_start
    
    
    # --- 4. Integration ---
    print(f"   Integrating annotations...")
    gene_ids = get_gene_ids_from_faa(prokka_faa)
    ko_annot_dict = parse_kofam_output(kofam_output_file)
    cluster_annot_dict = get_best_annotation_diamond(diamond_output_cluster)
    
    final_annotations = {}
    for gene_id in gene_ids:
        annotation = None
        if gene_id in ko_annot_dict and ko_annot_dict[gene_id]:
            annotation = ko_annot_dict[gene_id]
        elif gene_id in cluster_annot_dict:
            annotation = cluster_annot_dict[gene_id] 
        else:
            annotation = "<oov>" 
        final_annotations[gene_id] = annotation

    result_path = final_dir / f"{sample_name}"
    with open(result_path, 'w') as f: 
        f.write('\t'.join(list(final_annotations.values())) + '\n')
        
    total_time = time.time() - total_start
    
    return {
        "name": sample_name,
        "prokka": time_prokka,
        "hmm": time_hmm,
        "diamond": time_diamond,
        "total": total_time
    }

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline with detailed timer for Prokka, KofamScan(HMM) and DIAMOND",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--input_dir', required=True, type=Path, help='Input directory')
    parser.add_argument('-o', '--output_dir', required=True, type=Path, help='Output directory root')
    parser.add_argument('-e', '--ext', type=str, default="fna", help='File extension')
    parser.add_argument('-t', '--threads', type=int, default=8, help='Threads')
    
    parser.add_argument('--ko_list', type=str, help='Path to ko_list file')
    parser.add_argument('--profiles', type=str, help='Path to profiles directory')
    parser.add_argument('--kofam_exec', type=str, default="exec_annotation", help='Path to exec_annotation')
    parser.add_argument('--tmp_dir', type=str, default="/dev/shm/tmp", help='Temp dir for KofamScan')
    parser.add_argument('--cdb', required=True, type=str, help='Cluster DB (.dmnd)')
    
    parser.add_argument('--mem_cdb', action='store_true', help='Load Diamond DB into /dev/shm before running')
    parser.add_argument('--mem_profiles', action='store_true', help='Load KofamScan profiles into /dev/shm before running')
    parser.add_argument('--keep_mem_db', action='store_true', help='Do not delete DB from memory at exit (if loaded)')
    
    args = parser.parse_args()


    disk_cdb = Path(args.cdb)
    disk_profiles = Path(args.profiles)
    
    if args.mem_cdb:
        mem_cdb = load_path_to_memory(disk_cdb)
    else:
        print(f"[System] Using Diamond DB from disk: {disk_cdb}")
        mem_cdb = disk_cdb

    if args.mem_profiles:
        mem_profiles = load_path_to_memory(disk_profiles)
    else:
        print(f"[System] Using Profiles from disk: {disk_profiles}")
        mem_profiles = disk_profiles
    
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist -> {args.input_dir}")
        sys.exit(1)

    genome_files = list(args.input_dir.glob(f"*.{args.ext}"))

    if not genome_files:
        print(f"No files found with extension .{args.ext} in {args.input_dir}")
        sys.exit(1)
    else:
        print(f"Found {len(genome_files)} genome files.")
        
        final_output_dir = args.output_dir / "results"
        intermediate_output_dir = args.output_dir / "intermediate"
        
        print("-" * 60)
        print(f"[Output] Final annotation results will be in:       {final_output_dir}")
        print(f"[Output] Intermediate files will be in:  {intermediate_output_dir}")
        print("-" * 60)
        
        final_output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_output_dir.mkdir(parents=True, exist_ok=True)
        # ----------------------------

        if not args.keep_mem_db:
            if args.mem_cdb:
                atexit.register(cleanup_memory_path, mem_cdb)
            if args.mem_profiles:
                atexit.register(cleanup_memory_path, mem_profiles)
        
        results = []
        overall_start = time.time()
        
        for genome in genome_files:
            stats = pipeline(
                input_genome=genome, 
                ko_list=args.ko_list,
                profiles_path=mem_profiles,
                kofam_exec=args.kofam_exec,
                diamond_db_path=mem_cdb,
                threads=args.threads, 
                final_dir=final_output_dir,
                inter_dir=intermediate_output_dir,
                tmp_root=args.tmp_dir
            )
            results.append(stats)
            print(f">>> Finished {stats['name']}: Prokka={stats['prokka']:.1f}s, HMM={stats['hmm']:.1f}s, Diamond={stats['diamond']:.1f}s")

        total_elapsed = time.time() - overall_start
        
        print("\n" + "=" * 80)
        print(">>> PROCESSING SUMMARY")
        print("=" * 80)
        
        header = f"{'Sample':<25} | {'Prokka (s)':<10} | {'HMM (s)':<10} | {'Diamond (s)':<12} | {'Total (s)':<10}"
        print(header)
        print("-" * 80)
        
        for r in results:
            line = f"{r['name']:<25} | {r['prokka']:<10.1f} | {r['hmm']:<10.1f} | {r['diamond']:<12.1f} | {r['total']:<10.1f}"
            print(line)
            
        print("-" * 80)
        print(f"Grand Total Time: {total_elapsed:.2f} seconds")
        
if __name__ == "__main__":
    main()