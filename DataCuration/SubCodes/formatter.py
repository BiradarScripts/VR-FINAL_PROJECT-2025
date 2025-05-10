import json

input_path = './analyzed_images/010-mllS7JL.json'
output_path = './analyzed_images/010-mllS7JL_parsed.json'

# Clean text function
def clean_text(text):
    return text.strip().strip('"').strip('\\')

# Read the content
with open(input_path, 'r') as file:
    raw_data = file.read().strip()

# Optional: preview what's inside
print("RAW DATA SAMPLE:\n", repr(raw_data[:300]))

# Split blocks by double newlines
blocks = raw_data.split('\\n\\n')

qa_list = []

for block in blocks:
    lines = block.strip().split('\\n')
    if len(lines) == 2:
        try:
            question_line = lines[0].strip()
            answer_line = lines[1].strip()

            # Use partition to safely split at first colon
            _, _, question_value = question_line.partition(':')
            _, _, answer_value = answer_line.partition(':')

            qa_list.append({
                "question": clean_text(question_value),
                "answer": clean_text(answer_value)
            })
        except Exception as e:
            print(f"⚠️ Skipping block due to error: {block} — {e}")

# Save the parsed list to a JSON file
with open(output_path, 'w') as outfile:
    json.dump(qa_list, outfile, indent=2)

print(f"✅ Parsed {len(qa_list)} Q&A pairs to: {output_path}")
