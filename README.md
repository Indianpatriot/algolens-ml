# AlgoLens — Machine Learning & Structural Code Analysis Engine 🔍⚡

> **Structural AST Feature Extraction, Random Forest Algorithm Classification & Rule-Based Big-O Complexity Estimation.**

🌐 **Live Demo**: [https://algo-lens-viz.lovable.app](https://algo-lens-viz.lovable.app)

<!-- TODO: Record and add demo.gif to docs/demo.gif -->
![AlgoLens Demo](docs/demo.gif)

AlgoLens is powered by a real, deterministic machine learning and static code analysis pipeline — **not an LLM wrapper**. It parses source code directly into AST structural feature vectors, classifies algorithmic paradigms using a trained Random Forest model (100% cross-validation accuracy on canonical algorithmic signatures), and deterministically computes exact Big-O time and space complexity bounds with human-readable rationale.

---

## 🌟 Capabilities

AlgoLens analyzes code across 10+ core algorithm families, detecting structural patterns, loop bounds, recursion trees, and data structure semantics without executing untrusted code or relying on generative AI hallucinations.

- **AST Feature Extractor**: Extracts structural signals (loop depth, recursion branching, halving patterns, pointer movements, stack/queue operations, DP table updates, graph adjacency).
- **Random Forest Classifier**: Classifies algorithm categories with confidence scores.
- **Rule-Based Complexity Engine**: Derives formal Time and Space Big-O bounds ($O(1)$, $O(\log N)$, $O(N)$, $O(N \log N)$, $O(N^2)$, $O(V + E)$).
- **Dynamic Parameter & Trace Dispatch**: Extracts arrays, targets, k-values, intervals, trees, and graphs for downstream visual trace generation.

<details>
<summary><strong>Full list of 35+ supported algorithms & patterns</strong></summary>

- **Dynamic Programming**: Climbing Stairs (Fibonacci), Coin Change, 0/1 Knapsack, Longest Increasing Subsequence (LIS), Word Break, House Robber II (Circular), Longest Palindromic Subsequence (LPS 2D Interval Table).
- **Two Pointers & Hash Map**: Two Sum (Hash Map complement lookup), Two Sum II (Sorted Array converging), 3Sum (Triplets), Container With Most Water, Move Zeroes, Remove Duplicates, Merge Sorted Arrays, Fast & Slow Pointer Cycle Detection.
- **Intervals**: Insert Interval (3-phase sweep), Interval List Intersections (Range overlaps), Interval Scheduling (Greedy earliest finish time).
- **Binary Trees**: Inorder, Preorder, Postorder, Level-Order (BFS queue), Invert Binary Tree, Lowest Common Ancestor (LCA).
- **Heaps & Priority Queues**: Top K Frequent Elements, Kth Largest Element in Array, Heap Sort.
- **Monotonic Stack**: Daily Temperatures, Next Greater Element, Largest Rectangle in Histogram, Online Stock Span, Evaluate Reverse Polish Notation (RPN).
- **Graphs**: Dijkstra Shortest Path, Kruskal's MST (DSU), Kahn's Algorithm (BFS In-Degree), Topological Sort (DFS Post-Order).
- **Linked Lists**: Singly Linked List (Insert, Delete, Reverse), Doubly Linked List, Linked List Merge Sort.
- **Searching & Bounds**: Binary Search, Search in Rotated Sorted Array, Lower Bound, Upper Bound, Linear Search.
- **Backtracking**: N-Queens, Permutations, Subsets, Combination Sum.
- **String Matching**: Horspool's Algorithm (Bad-Symbol Shift Table).

</details>

---

## 🛠️ Tech Stack

| Component | Technologies |
|---|---|
| **API Framework** | Python 3.10+, FastAPI, Uvicorn, Pydantic |
| **Machine Learning** | scikit-learn (Random Forest), pandas, joblib |
| **Code Analysis** | Python standard `ast` module, structural tokenizers |
| **Frontend Visualizer** | React 19, TanStack Start, Tailwind CSS, D3.js ([algo-lens-viz](https://github.com/Indianpatriot/algo-lens-viz)) |

---

## 🚀 Getting Started

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Indianpatriot/algolens-ml.git
cd algolens-ml

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the FastAPI Server

```bash
uvicorn main:app --reload --port 8000
```

API is available at `http://127.0.0.1:8000`.

### 3. API Usage Example

```bash
curl -X POST http://127.0.0.1:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"code": "def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target - x], i]\n        seen[x] = i\n    return []"}'
```

---

## 🧪 Testing

Run internal analysis and validation test suites:

```bash
# Verify AST feature extractor
python test_extractor.py

# Verify execution tracer
python test_tracer.py

# Run independent classification tests
python independent_test.py

# Check model cross-validation accuracy
python cv_check.py
```

---

## 🤝 Contributing

Contributions, issue reports, and algorithm submissions are welcome. Please open an issue or pull request.

---

## 👤 About

Created by [Indianpatriot](https://github.com/Indianpatriot) — building intelligent developer and computer science education tools.

---

## 📄 License

This project is licensed under the MIT License (or your specified license). See `LICENSE` for details.

