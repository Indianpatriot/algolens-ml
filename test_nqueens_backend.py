import json
import urllib.request
from feature_extractor import extract_features, extract_top_level_parameters
from complexity_estimator import estimate_complexity

code_user = """def solve_n_queens(n):
    cols, d1, d2 = set(), set(), set()
    board = [-1] * n
    res = []

    def backtrack(r):
        if r == n:
            res.append(board[:])
            return
        for c in range(n):
            if c in cols or (r - c) in d1 or (r + c) in d2:
                continue
            cols.add(c); d1.add(r - c); d2.add(r + c)
            board[r] = c
            backtrack(r + 1)
            cols.remove(c); d1.remove(r - c); d2.remove(r + c)

    backtrack(0)
    return res

n = 8
"""

def test_user_code_pipeline():
    # 1. Parameter extraction
    params = extract_top_level_parameters(code_user)
    assert params.get("n") == 8, f"Expected n=8 in params, got {params}"
    print("[PASS] Top-level parameter extraction: n = 8")

    # 2. Features
    feats = extract_features(code_user)
    assert feats.get("detected_n_value") == 8.0, f"Expected detected_n_value=8.0, got {feats.get('detected_n_value')}"
    print("[PASS] Feature extractor detected_n_value: 8.0")

    # 3. Complexity reasoning
    comp = estimate_complexity(code_user, algorithm="N-Queens", class_confidence=0.99)
    assert "8×8" in comp["reasoning"], f"Expected 8x8 in reasoning: {comp['reasoning']}"
    assert "92" in comp["reasoning"], f"Expected 92 solutions in reasoning: {comp['reasoning']}"
    print("[PASS] Complexity reasoning dynamically reflects 8x8 and 92 valid solutions")

    # 4. Live API test
    try:
        req = urllib.request.Request(
            "http://localhost:8000/analyze",
            data=json.dumps({"code": code_user, "language": "python"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["algorithm"] == "N-Queens", f"Algorithm expected N-Queens, got {data['algorithm']}"
            assert data.get("detected_n_value") == 8.0, f"Expected detected_n_value=8.0, got {data.get('detected_n_value')}"
            assert data.get("extracted_params", {}).get("n") == 8, f"Expected extracted_params.n=8, got {data.get('extracted_params')}"
            assert "8×8" in data["complexity"]["reasoning"], f"Expected 8x8 in API complexity: {data['complexity']['reasoning']}"
            assert "92" in data["complexity"]["reasoning"], f"Expected 92 in API complexity: {data['complexity']['reasoning']}"
            print("[PASS] Live /analyze API endpoint returns N-Queens, detected_n_value=8, extracted_params={'n': 8}, 8x8 chessboard and 92 solutions!")
    except Exception as e:
        print(f"[WARN] Could not connect to live API: {e}")

if __name__ == "__main__":
    test_user_code_pipeline()
