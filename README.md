# Natural Language to SQL

Natural Language to SQL is a project that enables users to convert plain English queries into executable SQL statements. This can help users interact with relational databases without needing to know SQL syntax, making data querying more accessible for non-technical users.

- Fine-tuned CodeT5 on a subset of the Spider dataset to translate natural language questions into executable SQL queries.

- Implemented basic schema linking by combining database schema with input questions for better context understanding.

- Built training & evaluation pipeline using Hugging Face Transformers, with exact-match metric for SQL correctness.

- Added inference module to generate SQL from user queries and a Gradio interface for interactive testing.

## Getting Started

### Prerequisites

- Python 3.7+
- Required libraries (see `requirements.txt`)
### Dataset

Spider dataset (Subset used in this project)

### Installation

```bash
git clone https://github.com/Ayeshazaf/Natural_language_to_SQL.git
cd Natural_language_to_SQL
pip install -r requirements.txt
```

### Usage

1. **Run the application**:  
   ```bash
   python main.py
   ```
2. **Enter a natural language question**:  
   Example:  
   ```
   Show all customers who placed orders in August.
   ```
3. **View the generated SQL query and results**.

## Example

Input:  
```
List all employees with a salary greater than $50,000.
```

Output SQL:
```sql
SELECT * FROM employees WHERE salary > 50000;
```

## Project Structure

```
Natural_language_to_SQL/
├── app.py
├── requirements.txt
├── README.md
├── models/
├── utils/
```

## Technologies Used

- Python
- Natural Language Processing (NLP)
- SQL

## Contributing

Pull requests are welcome! 
