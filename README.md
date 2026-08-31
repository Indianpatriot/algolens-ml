# AlgoLens 🔍⚡

> **Intelligent Algorithm Identification, Structural AST Analysis, Complexity Estimation & Interactive Step-by-Step Visualization Engine.**

[![Live Demo](https://img.shields.io/badge/Live%20Deployable%20App-algo--lens--viz.lovable.app-6366f1?style=for-the-badge&logo=lovable&logoColor=white)](https://algo-lens-viz.lovable.app)

🌐 **Live Deployable Application**: [https://algo-lens-viz.lovable.app](https://algo-lens-viz.lovable.app)

---

AlgoLens is an end-to-end platform that analyzes algorithmic source code (Python & JavaScript), extracts structural AST signals, classifies the algorithm category using Machine Learning and rule-based inference, estimates Time/Space complexity bounds, and generates step-by-step interactive visual traces.

---

## 🌟 Key Features

### 1. 🧠 Machine Learning & Structural Code Analysis
- **AST Feature Extractor**: Parses code structure into feature vectors capturing loop depth, recursion branching, halving patterns, pointer movements, stack/queue operations, DP table updates, and graph adjacency structures.
- **Random Forest Classifier**: Machine-learned category predictions with confidence scores.
- **Rule-Based Complexity Estimator**: Derives exact Big-O Time and Space bounds (e.g. $O(N)$, $O(N \log N)$, $O(N^2)$, $O(V + E)$) with human-readable rationale.
- **Dynamic Parameter Extraction**: Automatically detects and extracts arrays, targets, k-values, intervals, trees, and graphs directly from code variables or function calls.

### 2. 🎬 Rich Interactive Visualization Suites
- **Dynamic Programming**:
  - 1D DP: Climbing Stairs, Coin Change, Longest Increasing Subsequence (LIS), Word Break, House Robber II.
  - 2D DP: 0/1 Knapsack, Longest Palindromic Subsequence (LPS 2D interval grid).
- **Two Pointers**:
  - Converging: Two Sum II (Sorted Array), 3Sum (Triplets with zero sum), Container With Most Water.
  - Hash Map Lookup: Two Sum (Complement formula $O(N)$ with interactive dictionary ribbon).
  - Same-Direction: Move Zeroes (in-place array compaction), Remove Duplicates.
  - Parallel: Merge Two Sorted Arrays, Array Intersection.
  - Fast & Slow: Floyd's Tortoise & Hare Cycle Detection.
- **Interval Algorithms**:
  - **Insert Interval**: Explicit 3-phase visualization showcasing the new interval insertion and merging.
  - **Interval List Intersections**: Horizontal range bars displaying start/end overlap calculations $[ \max(\text{start}), \min(\text{end}) ]$.
  - **Interval Scheduling / Activity Selection**: Greedy earliest-finish-time selection.
- **Binary Trees**:
  - Traversals: Inorder, Preorder, Postorder, Level-Order (BFS queue).
  - Tree Transformations: Invert Binary Tree (mirroring left/right subtrees), Lowest Common Ancestor (LCA).
- **Heaps & Priority Queues**:
  - Top K Frequent Elements (frequency hash map + bounded min-heap).
  - Kth Largest Element in Array (bounded min-heap stream selection).
  - Heap Sort (max-heapify and extraction).
- **Monotonic Stack**:
  - Daily Temperatures (next warmer day), Next Greater Element (NGE).
  - Largest Rectangle in Histogram (increasing stack boundary expansion).
  - Online Stock Span, Evaluate Reverse Polish Notation (RPN).
- **Graphs**:
  - Dijkstra Single-Source Shortest Path (min-priority queue).
  - Kruskal's Minimum Spanning Tree (Disjoint Set Union / DSU).
  - Topological Sort (Kahn's in-degree BFS and DFS post-order recursion).
- **Linked Lists**:
  - Singly Linked List: Step-by-step Node Insertion, Deletion by position/value, and In-Place Reversal.
  - Doubly Linked List: Bidirectional pointer updates.
  - Linked List Merge Sort.
- **Searching & Bounds**:
  - Binary Search, Search in Rotated Sorted Array, Lower Bound ($\ge \text{target}$), Upper Bound ($> \text{target}$), Linear Search.
- **Backtracking**:
  - N-Queens (chessboard placement with conflict detection), Permutations, Subsets, Combination Sum.
- **String Matching**:
  - Horspool's Algorithm (Bad-symbol shift table and right-to-left scan).

### 3. 🎯 Manual Algorithm Selection Fallback
- Accessible via **"Not finding what you wanted?"** after code analysis.
- Live search across supported problems with instant alias resolution (e.g. `LPS`, `LIS`, `LCA`, `3Sum`, `Top K`, `Rotated`, `Inorder`).
- Code preview and one-click loading of verified canonical Python/JavaScript implementations directly into the workspace editor.

---

## 🏗️ Repository Architecture

```text
algolens-ml/
├── main.py                     # FastAPI service exposing POST /analyze
├── feature_extractor.py        # Python AST structural signal extractor
├── complexity_estimator.py     # Rule-based Big-O Time/Space complexity engine
├── execution_tracer.py         # Dynamic runtime execution tracer
├── model.py                    # Random Forest loader and inference pipeline
├── model.pkl                   # Trained scikit-learn model artifact
├── feature_order.pkl           # Feature schema mapping
├── requirements.txt            # Python dependencies
│
├── algo-lens-viz/              # Frontend Web Application (TanStack Start + React)
│   ├── src/
│   │   ├── components/         # UI components & Visualizers
│   │   │   ├── viz/            # D3 / Canvas / React visualizer views
│   │   │   │   ├── dp/         # 1D & 2D Dynamic Programming visualizer
│   │   │   │   ├── two_pointers/# Two Pointers & Interval Intersections
│   │   │   │   ├── tree/       # Binary Tree & LCA visualizers
│   │   │   │   ├── heap/       # Heap & Top-K visualizers
│   │   │   │   ├── graph/      # Graph & Topological Sort visualizers
│   │   │   │   ├── stack/      # Monotonic Stack visualizers
│   │   │   │   └── linked_list/# Singly & Doubly Linked List visualizers
│   │   │   ├── CodeEditor.tsx  # Monaco Code Editor
│   │   │   └── ManualAlgorithmSelectionModal.tsx # Fallback modal
│   │   ├── lib/
│   │   │   └── analysis/       # TypeScript analysis, AST parser, trace builder
│   │   │       ├── structural.ts       # Structural feature detectors
│   │   │       ├── identify.ts         # Algorithm identification & scoring
│   │   │       ├── inputs.ts           # Dynamic input parameter parser
│   │   │       ├── trace.ts            # Trace generators for all 35+ algorithms
│   │   │       ├── patterns.ts         # Algorithm registry & canonical snippets
│   │   │       └── supportedProblems.ts# Searchable problem catalog with aliases
│   │   └── routes/             # TanStack file-based routes (Workspace, History, etc.)
│   └── package.json            # Node.js dependencies & scripts
│
└── working-examples/           # Curated canonical algorithm test snippets
    └── snippet.txt             # Reference snippet repository (read-only)
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** & **npm**

---

### Backend Setup (FastAPI ML Service)

1. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   The API will be live at `http://127.0.0.1:8000`.

4. **Test the endpoint**:
   ```bash
   curl -X POST http://127.0.0.1:8000/analyze \
        -H "Content-Type: application/json" \
        -d '{"code": "def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target - x], i]\n        seen[x] = i\n    return []"}'
   ```

---

### Frontend Setup (AlgoLens Visualization Platform)

1. **Navigate to the frontend directory**:
   ```bash
   cd algo-lens-viz
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```
   Open your browser at `http://localhost:5173`.

4. **Build for production**:
   ```bash
   npm run build
   ```

---

## 🧪 Testing & Verification

Run the test suites to verify classifier accuracy, trace generators, and input parsers:

```bash
# In algo-lens-viz directory:

# 1. Two Sum Hash Map vs Search Disambiguation Suite
npx tsx ../scratch/test_twosum_hashmap_suite.ts

# 2. Top-K, Linked List Sequence, Invert Tree & Traversals Suite
npx tsx ../scratch/test_topk_llseq_invert_traversals_suite.ts

# 3. Manual Algorithm Selection & Canonical Pipeline Suite
npx tsx ../scratch/test_manual_algorithm_selection_suite.ts
```

Run Python backend tests:
```bash
python test_extractor.py
python test_tracer.py
python independent_test.py
```

---

## 🛠️ Tech Stack

| Domain | Technologies |
|---|---|
| **Frontend Framework** | React 19, TanStack Start, TanStack Router, TanStack Query |
| **Styling & Components** | Tailwind CSS, Radix UI Primitives, Lucide Icons, Class Variance Authority |
| **Visualizations** | D3.js (Hierarchy, Curves, Layouts), HTML5 Canvas, SVG |
| **Code Editor** | Monaco Editor / CodeMirror |
| **Client-Side Analysis** | Babel Parser (`@babel/parser`), Structural Tokenizers |
| **Backend & ML** | Python 3, FastAPI, Uvicorn, scikit-learn (Random Forest), pandas, joblib |

---

## 📄 License

This project is licensed under the MIT License.
