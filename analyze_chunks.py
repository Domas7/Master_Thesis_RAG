import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Load the JSON file
with open('RAG/SPACECRAFT22_section_chunks_CHECKTHIS.json', 'r') as f:
    data = json.load(f)

# Extract document names and create a dictionary to count chunks per document
documents = {}
chunk_lengths = []
section_levels = []

for chunk in data:
    # Get file name from metadata
    file_name = chunk.get('metadata', {}).get('file_name', 'Unknown')
    
    # Count chunks per document
    if file_name in documents:
        documents[file_name] += 1
    else:
        documents[file_name] = 1
    
    # Get content length for histogram
    content = chunk.get('content', '')
    chunk_lengths.append(len(content))
    
    # Get section level if available
    section_level = chunk.get('section_level', None)
    if section_level is not None:
        section_levels.append(section_level)

# Create output directory if it doesn't exist
os.makedirs('statistics', exist_ok=True)

# 1. Histogram of chunks per document
plt.figure(figsize=(12, 6))
# Sort the documents by name for better readability
sorted_docs = dict(sorted(documents.items()))
plt.bar(sorted_docs.keys(), sorted_docs.values())
plt.xlabel('Document Name')
plt.ylabel('Number of Chunks')
plt.title('Number of Chunks per Document')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('statistics/chunks_per_document.png')
plt.close()

# New histogram: Number of documents by chunk count
plt.figure(figsize=(12, 6))
# Count how many documents have each chunk count
chunk_count_distribution = {}
for doc, count in documents.items():
    if count in chunk_count_distribution:
        chunk_count_distribution[count] += 1
    else:
        chunk_count_distribution[count] = 1

# Convert to list of tuples and sort numerically
sorted_items = sorted(chunk_count_distribution.items(), key=lambda x: int(x[0]))
# Unpack into separate lists for plotting
chunk_counts, doc_counts = zip(*sorted_items)

plt.bar(chunk_counts, doc_counts)
plt.xlabel('Number of Chunks per Document')
plt.ylabel('Number of Documents')
plt.title('Distribution of Documents by Chunk Count')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('statistics/documents_by_chunk_count.png')
plt.close()

# Line graph: Number of documents by chunk count
plt.figure(figsize=(12, 6))
# Use the same sorted data from the histogram
plt.plot(chunk_counts, doc_counts, marker='o', linestyle='-', linewidth=2, markersize=6, color='blue')
plt.fill_between(chunk_counts, doc_counts, alpha=0.3, color='blue')
plt.xlabel('Number of Chunks per Document')
plt.ylabel('Number of Documents')
plt.title('Distribution of Documents by Chunk Count (Line Graph)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('statistics/documents_by_chunk_count_line.png')
plt.close()

# 1b. Line graph of chunks per document
plt.figure(figsize=(12, 6))
# Sort the documents by number of chunks for better visualization
sorted_docs = dict(sorted(documents.items(), key=lambda item: item[1], reverse=True))
plt.plot(sorted_docs.keys(), sorted_docs.values(), marker='o', linestyle='-', linewidth=2, markersize=8)
plt.fill_between(sorted_docs.keys(), sorted_docs.values(), alpha=0.3)
plt.xlabel('Document Name')
plt.ylabel('Number of Chunks')
plt.title('Number of Chunks per Document (Line Graph)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('statistics/chunks_per_document_line.png')
plt.close()

# 2e. Token-based histogram (capped at 5,000 tokens)
token_lengths = [length / 4 for length in chunk_lengths]  # Estimate tokens (4 chars ≈ 1 token)
plt.figure(figsize=(10, 6))
token_cap = 2000  # Cap at 5,000 tokens (equivalent to 20,000 characters)
capped_token_lengths = [min(length, token_cap) for length in token_lengths]
plt.hist(capped_token_lengths, bins=20, alpha=0.7, color='teal')
plt.xlabel('Estimated Tokens per Chunk (capped at 2,000)')
plt.ylabel('Frequency')
plt.title('Distribution of Chunk Lengths in Tokens')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('statistics/chunk_token_distribution.png')
plt.close()

# 5. Create a summary dataframe for more detailed analysis
summary_data = []
for i, chunk in enumerate(data):
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

# Estimate tokens using a simple approximation (4 characters ≈ 1 token)
df['estimated_tokens'] = df['content_length'] / 4

# 9b. Capped histogram of tokens per chunk
plt.figure(figsize=(10, 6))
token_cap = 1000  # Cap at 10,00 tokens
df['estimated_tokens_capped'] = df['estimated_tokens'].apply(lambda x: min(x, token_cap))
plt.hist(df['estimated_tokens_capped'], bins=30, alpha=0.7, color='orange')
plt.xlabel(f'Estimated Tokens per Chunk (capped at {token_cap})')
plt.ylabel('Number of Chunks')
plt.title('Distribution of Estimated Tokens per Chunk (Capped)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('statistics/tokens_per_chunk_distribution_capped.png')
plt.close()

print(f"Total number of chunks: {len(data)}")
print(f"Number of unique documents: {len(documents)}")
print(f"Average chunk length: {sum(chunk_lengths)/len(chunk_lengths):.2f} characters")
print(f"Statistics saved to the 'statistics' directory") 