import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from glob import glob

# Create output directory if it doesn't exist
os.makedirs('statistics', exist_ok=True)

# Load multiple JSON files
json_files = [
    'reprocessed_section_chunks/reprocessed_section_chunks_final.json',
    'reprocessed_section_chunks_2/reprocessed_section_chunks_2_final.json',
    'reprocessed_section_chunks_3/reprocessed_section_chunks_3_final.json'
]

# Initialize data structures
all_data = []
all_data_by_file = {}  # Store data separately for each file
documents = {}
documents_by_file = {}  # Store document counts separately for each file
chunk_lengths = []
chunk_lengths_by_file = {}  # Store chunk lengths separately for each file
section_levels = []
section_levels_by_file = {}  # Store section levels separately for each file

# Load and process each file
for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
            
            # Store data separately for this file
            file_key = os.path.basename(json_file)
            all_data_by_file[file_key] = data
            documents_by_file[file_key] = {}
            chunk_lengths_by_file[file_key] = []
            section_levels_by_file[file_key] = []
            
            print(f"Loaded {len(data)} chunks from {json_file}")
            
            # Process chunks in this file
            for chunk in data:
                # Get file name from metadata
                file_name = chunk.get('metadata', {}).get('file_name', 'Unknown')
                
                # Count chunks per document (overall)
                if file_name in documents:
                    documents[file_name] += 1
                else:
                    documents[file_name] = 1
                
                # Count chunks per document (per file)
                if file_name in documents_by_file[file_key]:
                    documents_by_file[file_key][file_name] += 1
                else:
                    documents_by_file[file_key][file_name] = 1
                
                # Get content length for histogram (overall)
                content = chunk.get('content', '')
                chunk_lengths.append(len(content))
                
                # Get content length for histogram (per file)
                chunk_lengths_by_file[file_key].append(len(content))
                
                # Get section level if available (overall)
                section_level = chunk.get('section_level', None)
                if section_level is not None:
                    section_levels.append(section_level)
                    
                    # Get section level if available (per file)
                    section_levels_by_file[file_key].append(section_level)
    except FileNotFoundError:
        print(f"Warning: File {json_file} not found, skipping.")
    except json.JSONDecodeError:
        print(f"Warning: File {json_file} contains invalid JSON, skipping.")

# Documents by chunk count (bar chart)
plt.figure(figsize=(10, 6))
plt.title('Distribution of Documents by Chunk Count', fontsize=16)

# Process combined data
chunk_count_distribution = {}
for doc, count in documents.items():
    # Cap at 80 chunks per document
    count = min(count, 80)
    if count in chunk_count_distribution:
        chunk_count_distribution[count] += 1
    else:
        chunk_count_distribution[count] = 1

# Convert to list of tuples and sort numerically
sorted_items = sorted(chunk_count_distribution.items(), key=lambda x: int(x[0]))
chunk_counts, doc_counts = zip(*sorted_items)

plt.bar(chunk_counts, doc_counts)
plt.xlabel('Number of Chunks per Document (capped at 80)')
plt.ylabel('Number of Documents')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('statistics/documents_by_chunk_count_bar.png')
plt.close()

# Documents by chunk count (line graph)
plt.figure(figsize=(10, 6))
plt.title('Distribution of Documents by Chunk Count', fontsize=16)

plt.plot(chunk_counts, doc_counts, marker='o', linestyle='-', linewidth=2, markersize=4)
plt.fill_between(chunk_counts, doc_counts, alpha=0.3)
plt.xlabel('Number of Chunks per Document (capped at 80)')
plt.ylabel('Number of Documents')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('statistics/documents_by_chunk_count_line.png')
plt.close()

# Token distribution with cleaner presentation
plt.figure(figsize=(12, 8))
plt.title('Tokens per Chunk', fontsize=16)

# Calculate token lengths and apply filters
token_lengths = [length / 4 for length in chunk_lengths]  # Estimate tokens (4 chars ≈ 1 token)
filtered_token_lengths = [length for length in token_lengths if length >= 40]  # Changed to 40
displayed_token_lengths = [length for length in filtered_token_lengths if length <= 2000]

# Create histogram
plt.hist(displayed_token_lengths, bins=30, alpha=0.7, color='teal')
plt.xlabel('Number of Tokens per Chunk (capped at 2,000)', fontsize=12)
plt.ylabel('Number of Chunks', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('statistics/chunk_token_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a summary dataframe for more detailed analysis
summary_data = []
for i, chunk in enumerate(all_data):
    metadata = chunk.get('metadata', {})
    stats = metadata.get('statistics', {})
    
    summary_data.append({
        'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
        'file_name': metadata.get('file_name', 'Unknown'),
        'content_length': len(chunk.get('content', '')),
        'section_level': chunk.get('section_level', None),
        'section_number': chunk.get('section_number', None),
        'total_sections': chunk.get('total_sections', None),
        'word_count': stats.get('word_count', 0),
        'character_count': stats.get('character_count', 0),
        'average_word_length': stats.get('average_word_length', 0),
    })

df = pd.DataFrame(summary_data)

# Save summary statistics to CSV
df.describe().to_csv('statistics/chunk_summary_stats.csv')

# Also create individual summary statistics for each file
for file_key, data in all_data_by_file.items():
    file_summary_data = []
    for i, chunk in enumerate(data):
        metadata = chunk.get('metadata', {})
        stats = metadata.get('statistics', {})
        
        file_summary_data.append({
            'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
            'file_name': metadata.get('file_name', 'Unknown'),
            'content_length': len(chunk.get('content', '')),
            'section_level': chunk.get('section_level', None),
            'section_number': chunk.get('section_number', None),
            'total_sections': chunk.get('total_sections', None),
            'word_count': stats.get('word_count', 0),
            'character_count': stats.get('character_count', 0),
            'average_word_length': stats.get('average_word_length', 0),
        })
    
    file_df = pd.DataFrame(file_summary_data)
    file_df.describe().to_csv(f'statistics/chunk_summary_stats_{file_key}.csv')

# Print overall statistics
print(f"Total number of chunks across all files: {len(all_data)}")
print(f"Number of unique documents: {len(documents)}")
print(f"Average chunk length: {sum(chunk_lengths)/len(chunk_lengths):.2f} characters")

# Print statistics for each file
for file_key, data in all_data_by_file.items():
    lengths = chunk_lengths_by_file[file_key]
    print(f"\nFile: {file_key}")
    print(f"  Number of chunks: {len(data)}")
    print(f"  Number of unique documents: {len(documents_by_file[file_key])}")
    print(f"  Average chunk length: {sum(lengths)/len(lengths):.2f} characters")

print(f"\nStatistics saved to the 'statistics' directory") 