import json
import os

path = r'c:\CAWU4GROUP3\projects\projectRoodio\machineLearning\comparisson_model\exploratory_data_analysis\lyrics_eda_en.ipynb'

if not os.path.exists(path):
    print(f'Error: File not found at {path}')
    exit(1)

with open(path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Translation dictionary
translations = {
    '## EDA SONG LYRICS': '## SONG LYRICS EDA',
    '### 01. Library Setup': '### 01. Library Setup',
    '### 02. Import Datasets': '### 02. Import Datasets',
    '### 03. Data Cleaning': '### 03. Data Cleaning',
    '### 04. Data Distribution': '### 04. Data Distribution',
    '✅ Setup EDA selesai': '✅ EDA Setup complete',
    '✅ Informasi Dataset:': '✅ Dataset Information:',
    '📊 Informasi Dataset:': '📊 Dataset Information:',
    'Kolom:': 'Columns:',
    'Missing values:': 'Missing values:',
    '🔍 5 Sample Pertama:': '🔍 First 5 Samples:',
    '✅ Data cleaning selesai': '✅ Data cleaning complete',
    '📝 Contoh hasil cleaning:': '📝 Cleaning results example:',
    '📊 Statistik awal:': '📊 Initial statistics:',
    'Rata-rata jumlah kata:': 'Average word count:',
    'Min kata:': 'Min words:',
    'Max kata:': 'Max words:',
    '# MLflow untuk tracking EDA': '# MLflow for EDA tracking',
    '# Setup visualisasi': '# Setup visualization',
    "pd.read_excel('lyrics.xlsx')": "pd.read_excel('data/lyrics/lyrics.xlsx')",
    '# Terapkan cleaning': '# Apply cleaning',
    '# Tambah kolom analisis': '# Add analysis columns',
    'hapus angka': 'remove digits',
    'hapus punctuation': 'remove punctuation',
    'hapus whitespace berlebih': 'remove extra whitespace'
}

def translate_content(source):
    result = []
    for line in source:
        new_line = line
        for id_text, en_text in translations.items():
            new_line = new_line.replace(id_text, en_text)
        result.append(new_line)
    return result

for cell in notebook['cells']:
    if 'source' in cell:
        cell['source'] = translate_content(cell['source'])
    if 'outputs' in cell:
        for output in cell['outputs']:
            if 'text' in output:
                output['text'] = translate_content(output['text'])

with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('Translation complete.')
