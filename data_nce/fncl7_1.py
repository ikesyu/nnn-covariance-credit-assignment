"""
fncl7_1.py — 論文 §7.1「Resource Requirements」Table 6 の数値を算出する.

1 回の重み更新あたりの演算数とメモリ語数を、N (入力数), T (確率サンプル数), H (隠れ幅),
L (隠れ層数) から見積もり、標準出力に表として表示する。既定値 N=128, T=64, H=64, L=2
が論文の Table 6。列は

  deterministic NN (backprop, Adam) : 同じ大きさの決定論的ネットワーク (T=1, 活性は実数)
  NNN (backprop, Adam)              : 自動微分の sample-wise backward を NNN で行う参照
  NNN (cov_deriv, SGD)              : スカラー共分散 credit (§4.2)
  NNN (cov_jac, Adam)               : weight mirror の再帰 (§4.3)

行は forward サンプリング、乱数生成、学習の積和、推定量の除算、学習統計量・
オプティマイザ状態・生成器状態のメモリ、転置重みパスの要否。

  full-precision MACs : 乗算器を要する積和 (入力が実数の第 1 層、backprop の W^T delta,
                        cov_jac の credit 再帰 W_hat^T delta)
  binary-gated MACs   : 隠れ活性 z が二値 {0,1} のため AND ゲート + 加算器で済む積和
                        (学習統計量の累算、外積更新、アンサンブル平均の累算を含む)
  divisions           : 推定量の正規化 (Cov/Var)。オプティマイザの要素演算は含めない
  memory (words)      : 順方向の重み・バイアスを除く追加分

計数の前提 (Table 6 のキャプション): 交差活性は現サンプルと直前サンプルのノイズを使うので
新規の乱数生成はユニット・サンプルあたり 1 回、生成器状態はユニットあたり 16 ビット LFSR
1 本、NNN の backprop は全 NT サンプルの活性と誤差信号を保持する sample-wise backward。
これは演算数と語数の計数であり、面積・電力・レイテンシの実測ではない。

実行例:
  python data_nce/fncl7_1.py
  python data_nce/fncl7_1.py --N 128 --T 64 --H 64 --layers 2
"""
import argparse


def fmt(n: float) -> str:
    """概数表示 (3 有効数字の k/M 表示)。Table 6 は表示桁をさらに丸めている."""
    if n == 0:
        return "0"
    for unit, s in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= unit:
            return f"{n / unit:.3g}{s}"
    return f"{n:.0f}"


def learning_counts(N: int, T: int, H: int, L: int, d_in: int = 1, d_out: int = 1) -> dict:
    """学習 (forward サンプリングを除く) の追加コストを手法ごとの dict で返す.

    memory は成分のタプル (backprop は per-sample buffer + 転置アクセス可能な重みの複製)。
    """
    w_sizes = d_in * H + (L - 1) * H * H + H * d_out   # 全重み行列の要素数
    inter = (L - 1) * H * H + H * d_out                # 層間 (隠れ間 + 読み出し) の重み数
    outer = N * T * (d_in * H + (L - 1) * H * H)       # 隠れ層の外積更新
    readout = N * H * d_out                            # アンサンブル平均読み出しの外積
    return {
        "backprop": {
            # W^T delta の逆方向行列積 (delta は連続量 -> 真の乗算)
            "full_mac": N * T * inter,
            # delta z_prev^T の外積 (z は二値)
            "gated_mac": N * T * w_sizes,
            "div": 0,
            # per-sample buffer (全 NT サンプルの活性と誤差信号) + 転置アクセス用の重み複製
            "memory": (N * T * H * L, w_sizes),
        },
        "cov_deriv": {
            "full_mac": 0,
            # 共分散統計 (L*z, z^2) + KDE slope のカウンタ差分 + 外積更新
            "gated_mac": 2 * N * T * H * L + N * T * H * L + outer + readout,
            # per-input credit の除算 Cov/Var
            "div": N * H * L,
            # ユニットあたり 4 本の累算器
            "memory": (4 * H * L,),
        },
        "cov_jac": {
            # credit 再帰 W_hat^T delta (T に依らず入力ごとに 1 回)
            "full_mac": N * inter,
            # mirror 統計 Cov(d_next, z), Var(z) + KDE slope のカウンタ差分 + 外積更新
            "gated_mac": N * T * (inter + 2 * H * L) + outer + readout,
            # mirror の除算 (入力プーリング後に 1 回)
            "div": inter,
            # mirror とその EMA の 2 セット
            "memory": (2 * w_sizes,),
        },
    }


def forward_counts(N: int, T: int, H: int, L: int, d_in: int = 1, d_out: int = 1) -> dict:
    """T 回の確率的 forward サンプリング 1 回分 (1 更新あたり)."""
    w_sizes = d_in * H + (L - 1) * H * H + H * d_out
    return {
        "full_mac": N * T * d_in * H,                       # 第 1 層は入力が実数
        "gated_mac": N * T * ((L - 1) * H * H + H * d_out), # 以降は z が二値
        "rng_draws": N * T * H * L,                         # 新規の乱数生成 (ユニット・サンプルあたり 1 回)
        "ensemble_acc": N * T * d_out,                      # アンサンブル平均の累算
        "weights": w_sizes,
    }


def table6(N: int, T: int, H: int, L: int) -> dict:
    """Table 6 の 4 列を列名 -> {行名: 値} で返す (値は整数の計数)."""
    learn = learning_counts(N, T, H, L)
    fwd = forward_counts(N, T, H, L)
    w = fwd["weights"]
    cols = {}
    for method, label, optimizer in (("backprop", "NNN (backprop, Adam)", "adam"),
                                     ("cov_deriv", "NNN (cov_deriv, SGD)", "sgd"),
                                     ("cov_jac", "NNN (cov_jac, Adam)", "adam")):
        r = learn[method]
        learn_mem = sum(r["memory"])
        opt_mem = 2 * w if optimizer == "adam" else 0          # Adam の m, v
        rng_state = H * L                                      # LFSR 1 本 / ユニット
        cols[label] = {
            "Forward-pass full-precision MACs": fwd["full_mac"],
            "Forward-pass binary-gated MACs": fwd["gated_mac"],
            "Random-number draws": fwd["rng_draws"],
            "Learning full-precision MACs": r["full_mac"],
            "Learning binary-gated MACs": r["gated_mac"],
            "Divisions (estimators)": r["div"],
            "Total full-precision MACs": fwd["full_mac"] + r["full_mac"],
            "Total binary-gated MACs": fwd["gated_mac"] + r["gated_mac"] + fwd["ensemble_acc"],
            "Memory for learning statistics (words)": learn_mem,
            "Memory for optimizer state (words)": opt_mem,
            "Memory for generator state (words)": rng_state,
            "Total additional memory (words)": learn_mem + opt_mem + rng_state,
            "Transposed weight path": "required" if method == "backprop" else "not required",
        }
    # 決定論的ネットワーク (T=1, backprop, Adam): 活性が実数なので全て真の MAC
    det_fwd = forward_counts(N, 1, H, L)
    det_bp = learning_counts(N, 1, H, L)["backprop"]
    det_learn_mem = sum(det_bp["memory"])
    det = {
        "Forward-pass full-precision MACs": det_fwd["full_mac"] + det_fwd["gated_mac"],
        "Forward-pass binary-gated MACs": 0,
        "Random-number draws": 0,
        "Learning full-precision MACs": det_bp["full_mac"] + det_bp["gated_mac"],
        "Learning binary-gated MACs": 0,
        "Divisions (estimators)": 0,
        "Total full-precision MACs": (det_fwd["full_mac"] + det_fwd["gated_mac"]
                                      + det_bp["full_mac"] + det_bp["gated_mac"]),
        "Total binary-gated MACs": 0,
        "Memory for learning statistics (words)": det_learn_mem,
        "Memory for optimizer state (words)": 2 * w,
        "Memory for generator state (words)": 0,
        "Total additional memory (words)": det_learn_mem + 2 * w,
        "Transposed weight path": "required",
    }
    return {"deterministic NN (backprop, Adam)": det, **cols}


def main() -> None:
    p = argparse.ArgumentParser(description="§7.1 Table 6: operation and memory counts per update.")
    p.add_argument("--N", type=int, default=128, help="入力点数")
    p.add_argument("--T", type=int, default=64, help="確率サンプル数")
    p.add_argument("--H", type=int, default=64, help="隠れ幅")
    p.add_argument("--layers", type=int, default=2, help="隠れ層数 L")
    args = p.parse_args()

    cols = table6(args.N, args.T, args.H, args.layers)
    names = list(cols)
    rows = list(next(iter(cols.values())))
    print(f"Table 6 — operation and memory counts per weight update "
          f"(N={args.N}, T={args.T}, H={args.H}, L={args.layers})\n")
    print("| | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for row in rows:
        cells = []
        for n in names:
            v = cols[n][row]
            cells.append(v if isinstance(v, str) else f"{v:,} ({fmt(v)})")
        print(f"| {row} | " + " | ".join(cells) + " |")
    print("\nNotes: counts include forward sampling, random-number generation, readout "
          "averaging, learning statistics, and optimizer state, but exclude optimizer "
          "arithmetic. Memory excludes the forward weights and biases. These are counts, "
          "not measurements of area, power, or latency.")


if __name__ == "__main__":
    main()
