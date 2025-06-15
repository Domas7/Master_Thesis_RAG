import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# This class is not relevant for the thesis, it is a helper file to compare semantic and recursive chunking 
# It is kept here for reference in case I will work with it later.


def load_chunks(file_path):
    """Load chunks from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_chunks(recursive_chunks, semantic_chunks):
    """Analyze and compare chunks from both methods."""
    # Count chunks per document
    recursive_counts = Counter()
    semantic_counts = Counter()
    
    for chunk in recursive_chunks:
        file_name = chunk['metadata']['file_name']
        recursive_counts[file_name] += 1
    
    for chunk in semantic_chunks:
        file_name = chunk['metadata']['file_name']
        semantic_counts[file_name] += 1
    
    # Get common documents
    common_docs = set(recursive_counts.keys()) & set(semantic_counts.keys())
    
    # Calculate statistics
    recursive_total = len(recursive_chunks)
    semantic_total = len(semantic_chunks)
    
    recursive_avg = np.mean(list(recursive_counts.values()))
    semantic_avg = np.mean(list(semantic_counts.values()))
    
    recursive_std = np.std(list(recursive_counts.values()))
    semantic_std = np.std(list(semantic_counts.values()))
    
    # Calculate chunk length statistics
    recursive_lengths = [len(chunk['content']) for chunk in recursive_chunks]
    semantic_lengths = [len(chunk['content']) for chunk in semantic_chunks]
    
    recursive_len_avg = np.mean(recursive_lengths)
    semantic_len_avg = np.mean(semantic_lengths)
    
    recursive_len_std = np.std(recursive_lengths)
    semantic_len_std = np.std(semantic_lengths)
    
    # Print statistics
    print(f"Recursive Chunking:")
    print(f"  Total chunks: {recursive_total}")
    print(f"  Average chunks per document: {recursive_avg:.2f} ± {recursive_std:.2f}")
    print(f"  Average chunk length: {recursive_len_avg:.2f} ± {recursive_len_std:.2f} characters")
    print()
    print(f"Semantic Chunking:")
    print(f"  Total chunks: {semantic_total}")
    print(f"  Average chunks per document: {semantic_avg:.2f} ± {semantic_std:.2f}")
    print(f"  Average chunk length: {semantic_len_avg:.2f} ± {semantic_len_std:.2f} characters")
    
    # Create comparison visualizations
    create_visualizations(recursive_counts, semantic_counts, 
                         recursive_lengths, semantic_lengths,
                         common_docs)

def create_visualizations(recursive_counts, semantic_counts, 
                         recursive_lengths, semantic_lengths,
                         common_docs):
    """Create comparison visualizations."""
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Histogram of chunks per document
    ax1.hist(list(recursive_counts.values()), bins=20, alpha=0.5, label='Recursive', color='blue')
    ax1.hist(list(semantic_counts.values()), bins=20, alpha=0.5, label='Semantic', color='green')
    ax1.set_title('Chunks per Document')
    ax1.set_xlabel('Number of Chunks')
    ax1.set_ylabel('Number of Documents')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of chunk lengths
    ax2.hist(recursive_lengths, bins=20, alpha=0.5, label='Recursive', color='blue')
    ax2.hist(semantic_lengths, bins=20, alpha=0.5, label='Semantic', color='green')
    ax2.set_title('Chunk Lengths')
    ax2.set_xlabel('Characters')
    ax2.set_ylabel('Number of Chunks')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot comparing chunks per document
    common_recursive = [recursive_counts[doc] for doc in common_docs]
    common_semantic = [semantic_counts[doc] for doc in common_docs]
    
    ax3.scatter(common_recursive, common_semantic, alpha=0.5)
    ax3.set_title('Chunks per Document Comparison')
    ax3.set_xlabel('Recursive Chunks')
    ax3.set_ylabel('Semantic Chunks')
    
    # Add diagonal line for reference
    max_val = max(max(common_recursive), max(common_semantic))
    ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot of chunk lengths
    data = [recursive_lengths, semantic_lengths]
    ax4.boxplot(data, labels=['Recursive', 'Semantic'])
    ax4.set_title('Chunk Length Distribution')
    ax4.set_ylabel('Characters')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chunking_comparison.png', dpi=300)
    print("Saved comparison visualizations to chunking_comparison.png")

if __name__ == "__main__":
    # Load chunks
    recursive_chunks = load_chunks("recursive_chunks.json")
    semantic_chunks = load_chunks("semantic_chunks.json")
    
    # Analyze and compare
    analyze_chunks(recursive_chunks, semantic_chunks)