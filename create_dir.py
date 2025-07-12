from pathlib import Path

base_dir = Path("data/sample_data")

main_dirs = [
    "processed",
    "result",
    "train",
    "test",
    "finetuned_model_DNABERT_2",
    "finetuned_DNABERT_2",
    "embedding_vector"
]

for dir_name in main_dirs:
    dir_path = base_dir / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {dir_path}")

for sub in ["train", "test"]:
    kmer_dir = base_dir / sub / "kmer"
    kmer_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created: {kmer_dir}")

    result_dir = base_dir / sub / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created: {result_dir}")

