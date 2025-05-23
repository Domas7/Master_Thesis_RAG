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

# Combined figure: Number of documents by chunk count (bar chart)
plt.figure(figsize=(18, 6))

# Create subplots for each file plus one for combined data
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle('Distribution of Documents by Chunk Count (Bar Chart)', fontsize=16)

# Process each file separately
for i, (file_key, docs) in enumerate(documents_by_file.items()):
    # Count how many documents have each chunk count
    chunk_count_distribution = {}
    for doc, count in docs.items():
        # Cap at 80 chunks per document
        count = min(count, 80)
        if count in chunk_count_distribution:
            chunk_count_distribution[count] += 1
        else:
            chunk_count_distribution[count] = 1
    
    # Convert to list of tuples and sort numerically
    if chunk_count_distribution:
        sorted_items = sorted(chunk_count_distribution.items(), key=lambda x: int(x[0]))
        # Unpack into separate lists for plotting
        chunk_counts, doc_counts = zip(*sorted_items)
        
        # Plot on the corresponding subplot
        axes[i].bar(chunk_counts, doc_counts)
        axes[i].set_xlabel('Number of Chunks per Document')
        axes[i].set_ylabel('Number of Documents')
        axes[i].set_title(f'File {i+1}: {file_key}')
        axes[i].grid(True, alpha=0.3, axis='y')

# Process combined data for the last subplot
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
# Unpack into separate lists for plotting
chunk_counts, doc_counts = zip(*sorted_items)

# Plot on the last subplot
axes[3].bar(chunk_counts, doc_counts)
axes[3].set_xlabel('Number of Chunks per Document (capped at 80)')
axes[3].set_ylabel('Number of Documents')
axes[3].set_title('All Files Combined')
axes[3].grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
plt.savefig('statistics/documents_by_chunk_count_combined_bar.png')
plt.close()

# Combined figure: Number of documents by chunk count (line graph)
plt.figure(figsize=(18, 6))

# Create subplots for each file plus one for combined data
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle('Distribution of Documents by Chunk Count (Line Graph)', fontsize=16)

# Process each file separately
for i, (file_key, docs) in enumerate(documents_by_file.items()):
    # Count how many documents have each chunk count
    chunk_count_distribution = {}
    for doc, count in docs.items():
        # Cap at 80 chunks per document
        count = min(count, 80)
        if count in chunk_count_distribution:
            chunk_count_distribution[count] += 1
        else:
            chunk_count_distribution[count] = 1
    
    # Convert to list of tuples and sort numerically
    if chunk_count_distribution:
        sorted_items = sorted(chunk_count_distribution.items(), key=lambda x: int(x[0]))
        # Unpack into separate lists for plotting
        chunk_counts, doc_counts = zip(*sorted_items)
        
        # Plot on the corresponding subplot
        axes[i].plot(chunk_counts, doc_counts, marker='o', linestyle='-', linewidth=2, markersize=4)
        axes[i].fill_between(chunk_counts, doc_counts, alpha=0.3)
        axes[i].set_xlabel('Number of Chunks per Document (capped at 80)')
        axes[i].set_ylabel('Number of Documents')
        axes[i].set_title(f'File {i+1}: {file_key}')
        axes[i].grid(True, alpha=0.3)

# Process combined data for the last subplot
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
# Unpack into separate lists for plotting
chunk_counts, doc_counts = zip(*sorted_items)

# Plot on the last subplot
axes[3].plot(chunk_counts, doc_counts, marker='o', linestyle='-', linewidth=2, markersize=4)
axes[3].fill_between(chunk_counts, doc_counts, alpha=0.3)
axes[3].set_xlabel('Number of Chunks per Document (capped at 80)')
axes[3].set_ylabel('Number of Documents')
axes[3].set_title('All Files Combined')
axes[3].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
plt.savefig('statistics/documents_by_chunk_count_combined_line.png')
plt.close()

# Combined figure: Token distribution
plt.figure(figsize=(20, 6))
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle('Distribution of Chunk Lengths in Tokens', fontsize=16)

# Process each file separately
for i, (file_key, lengths) in enumerate(chunk_lengths_by_file.items()):
    # Convert to tokens
    token_lengths = [length / 4 for length in lengths]  # Estimate tokens (4 chars ≈ 1 token)
    
    # Filter out chunks with fewer than 50 tokens (likely errors)
    filtered_token_lengths = [length for length in token_lengths if length >= 50]
    
    # Filter to only include tokens up to 2000 (instead of capping)
    displayed_token_lengths = [length for length in filtered_token_lengths if length <= 2000]
    
    # Count how many were above the cap
    above_cap_count = len(filtered_token_lengths) - len(displayed_token_lengths)
    
    # Plot on the corresponding subplot
    axes[i].hist(displayed_token_lengths, bins=20, alpha=0.7, color='teal')
    axes[i].set_xlabel('Estimated Tokens (up to 2,000)')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'File {i+1}: {file_key}')
    axes[i].grid(True, alpha=0.3)
    
    # Add text showing how many chunks were filtered
    filtered_count = len(token_lengths) - len(filtered_token_lengths)
    if filtered_count > 0 or above_cap_count > 0:
        filter_text = []
        if filtered_count > 0:
            filter_text.append(f"Filtered out {filtered_count} chunks with <50 tokens")
        if above_cap_count > 0:
            filter_text.append(f"Not shown: {above_cap_count} chunks with >2000 tokens")
        
        axes[i].text(0.05, 0.95, "\n".join(filter_text), 
                    transform=axes[i].transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Process combined data for the last subplot
token_lengths = [length / 4 for length in chunk_lengths]  # Estimate tokens (4 chars ≈ 1 token)

# Filter out chunks with fewer than 10 tokens (likely errors)
filtered_token_lengths = [length for length in token_lengths if length >= 50]

# Filter to only include tokens up to 2000 (instead of capping)
displayed_token_lengths = [length for length in filtered_token_lengths if length <= 2000]

# Count how many were above the cap
above_cap_count = len(filtered_token_lengths) - len(displayed_token_lengths)

# Plot on the last subplot
axes[3].hist(displayed_token_lengths, bins=20, alpha=0.7, color='teal')
axes[3].set_xlabel('Estimated Tokens (up to 2,000)')
axes[3].set_ylabel('Frequency')
axes[3].set_title('All Files Combined')
axes[3].grid(True, alpha=0.3)

# Add text showing how many chunks were filtered
filtered_count = len(token_lengths) - len(filtered_token_lengths)
if filtered_count > 0 or above_cap_count > 0:
    filter_text = []
    if filtered_count > 0:
        filter_text.append(f"Filtered out {filtered_count} chunks with <50 tokens")
    if above_cap_count > 0:
        filter_text.append(f"Not shown: {above_cap_count} chunks with >2000 tokens")
    
    axes[3].text(0.05, 0.95, "\n".join(filter_text), 
                transform=axes[3].transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
plt.savefig('statistics/chunk_token_distribution_combined.png')
plt.close()

# Create additional figures for combined data with different token caps
token_caps = [500, 1000, 1500, 2000]
token_lengths = [length / 4 for length in chunk_lengths]  # Estimate tokens (4 chars ≈ 1 token)
filtered_token_lengths = [length for length in token_lengths if length >= 50]

# Create a figure with 4 subplots for different token caps
plt.figure(figsize=(20, 12))
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Distribution of Chunk Lengths in Tokens (All Files Combined)', fontsize=16)

# Flatten axes for easier iteration
axes = axes.flatten()

for i, cap in enumerate(token_caps):
    # Filter to only include tokens up to the current cap
    displayed_token_lengths = [length for length in filtered_token_lengths if length <= cap]
    
    # Count how many were above the cap
    above_cap_count = len(filtered_token_lengths) - len(displayed_token_lengths)
    
    # Plot on the corresponding subplot
    axes[i].hist(displayed_token_lengths, bins=20, alpha=0.7, color='teal')
    axes[i].set_xlabel(f'Estimated Tokens (up to {cap})')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Token Cap: {cap}')
    axes[i].grid(True, alpha=0.3)
    
    # Add text showing how many chunks were filtered
    filtered_count = len(token_lengths) - len(filtered_token_lengths)
    if filtered_count > 0 or above_cap_count > 0:
        filter_text = []
        if filtered_count > 0:
            filter_text.append(f"Filtered out {filtered_count} chunks with <50 tokens")
        if above_cap_count > 0:
            filter_text.append(f"Not shown: {above_cap_count} chunks with >{cap} tokens")
        
        axes[i].text(0.05, 0.95, "\n".join(filter_text), 
                    transform=axes[i].transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
plt.savefig('statistics/chunk_token_distribution_different_caps.png')
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