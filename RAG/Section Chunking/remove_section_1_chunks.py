import json
import os

# Load the file
input_file = "RAG/SPACECRAFT22_section_chunks_CHECKTHIS.json"
output_file = "RAG/Section Chunking/skipped_files_filtered_3.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter out entries with total_sections = 1
filtered_data = [chunk for chunk in data if chunk.get("total_sections", 0) != 1]

# Save the filtered data
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"Original count: {len(data)}")
print(f"Filtered count: {len(filtered_data)}")
print(f"Removed {len(data) - len(filtered_data)} entries")