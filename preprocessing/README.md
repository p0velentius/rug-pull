# Data Preprocessing Pipeline

## Overview

### Text Cleaning Pipeline
| File | Input | Output | Rows | Processing Time |
|------|-------|--------|------|-----------------|
| `preprocessing.py` | `questions_clean.csv` | `questions_preprocessed.csv` | 6,977 | < 1 second |
| `lemmatization.py` | `questions_preprocessed.csv` | `questions_lemmatizated.csv` | 6,977 | < 112 seconds |

### Website Data Processing
| File | Input | Output | Rows | Columns |
|------|-------|--------|------|---------|
| `preprocessing.py` | - | `websites_preprocessed.csv` | 1,938 | 4 |
| `lemmatization.py` | `websites_preprocessed.csv` | `websites_lemmatizated.csv` | 1,938 | 4 |

## Processing Parameters

### Preprocessing Settings
- `do_lower`: True
- `remove_emojis`: True  
- `remove_punct`: True
- `remove_polite`: True
- `do_lemmatize`: False
- `remove_short_tokens`: True

## Output Files Structure

### Questions Data
**questions_preprocessed.csv** (6,977 x 2)
- `q_id` - Question identifier
- `query_clean` - Cleaned text query

**questions_lemmatizated.csv** (6,977 x 2)
- `q_id` - Question identifier
- `query_clean` - Lemmatized text query

### Websites Data
**websites_preprocessed.csv** (1,938 x 4)
- `web_id` - Website identifier
- `url` - Website URL
- `kind` - Content type/category
- `title_clean` - Cleaned title text
- `text_clean` - Cleaned body text

**websites_lemmatizated.csv** (1,938 x 4)
- `web_id` - Website identifier
- `url` - Website URL
- `kind` - Content type/category
- `title_lemmas` - Lemmatized title text
- `text_lemmas` - Lemmatized body text

## Processing Features

### Text Normalization
- Anonymous number patterns (0000, XXXX, 0, XX, etc.) replaced with `<ANON_NUM>`
- Consistent token representation for vectorization
- Unicode-based emoji removal for reliability
- Careful punctuation removal preserving special tokens

### Lemmatization
- Converts words to their base/dictionary forms
- Applied to both questions and website content
- Maintains original data structure while enhancing text analysis

## Execution

```bash
# Run preprocessing
python preprocessing.py

# Run lemmatization  
python lemmatization.py
