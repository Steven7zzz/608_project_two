# B+ Tree & Hash-Join Demo

This repository contains two separate Python components for:

1. **Dense and Sparse B+ Trees** (in `B+_Tree/main.py`)  
2. **Two-Pass Hash Join** simulating virtual disk I/O (in `Join_based_on_Hashing/main.py`)

---

## Prerequisites

- **Python 3.7+**  
- **Graphviz** system package (for B+ Tree visualizations)  
- **pip** for Python dependencies

---

## Installation

1. **Clone the repository or just unzip the zip file**
   ```bash
   git clone https://github.com/yourusername/608_project_two.git
   cd 608_project_two
   ```

2. **Install dependencies**
   ```bash
   pip install graphviz pandas
   ```

---

## Running the Components

### 1. B+ Tree Demonstration

```bash
cd B+_Tree
python main.py
```

This will:
- Generate 10,000 random keys  
- Build dense/sparse B+ trees (orders 13 and 24)  
- Output full-tree PDF diagrams (e.g., `dense13_full.pdf`)  
- Create path‑based PNG snapshots for insert/delete operations  
- Perform searches and print results to the console  

### 2. Two-Pass Hash Join

```bash
cd Join_based_on_Hashing
python main.py
```

This will:
- Generate relation S and relation R on virtual disk blocks  
- Partition and join using the two‑pass hash algorithm  
- Print join results and I/O counts for Experiments 5.1 and 5.2  

---