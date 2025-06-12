import os
import json

def extract_unique_words_from_folder(folder_path):
    all_words = set()

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue

            text_fields = [v for k, v in data.items() if k.endswith("_q") or k.endswith("_a")]
            all_sentences = sum(text_fields, [])
            
            for sentence in all_sentences:
                words = sentence.split()
                for word in words:
                    cleaned = word.lower().strip(".,'\"")
                    all_words.add(cleaned)

    return sorted(all_words)

if __name__ == '__main__':
    dataset = 'training'
    folder_path = f'/data/womd-reasoning/{dataset}/{dataset}'
    unique_words = extract_unique_words_from_folder(folder_path)

    print(unique_words)
