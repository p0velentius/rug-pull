# 🛠 Data Preprocessing Pipeline

## 📊 Input/Output Overview

| **File** | **Input** | **Output** | **Rows** | **Processing Time** |
|----------|-----------|------------|----------|-------------------|
| `preprocessing.py` | `questions_clean.csv` | `questions_preprocessed.csv` | 6,977 | < 1 second |

## ⚙️ Processing Parameters

| **Parameter** | **Value** | **Description** |
|---------------|-----------|------------------|
| `do_lower` | `True` | Convert text to lowercase |
| `remove_emojis` | `True` | Remove all emoji characters |
| `remove_punct` | `True` | Remove punctuation marks |
| `remove_polite` | `True` | Remove polite phrases |
| `do_lemmatize` | `False` | **Disabled** lemmatization |
| `remove_short_tokens` | `True` | Remove short tokens |

## 📁 Output Structure

The processed file `questions_preprocessed.csv` contains:

| Column | Description |
|--------|-------------|
| `q_id` | Question identifier |
| `query_clean` | Cleaned and processed text query |

## 🎯 Key Processing Features

### 🔢 Anonymous Number Handling
- **Patterns**: `0000`, `XXXX`, `0`, `XX`, etc.
- **Replacement**: `⟨ANON_NUM⟩`
- **Benefit**: Vectorizer recognizes single token instead of multiple variants

### 😊 Emoji Removal
- **Method**: Regex based on Unicode ranges
- **Advantage**: More reliable than manual pattern matching

### 📝 Punctuation Cleaning
- **Approach**: Careful removal while preserving special tokens
- **Preserved**: Words and special tokens like `⟨ANON_NUM⟩`

## 🚀 Quick Start

```bash
python preprocessing.py
