import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Convert an adjacency matrix into a Markov matrix.
# ============================================================
def build_markov_from_adjacency(A):
    """
    Converts a raw adjacency matrix A into a column-stochastic Markov matrix P.

    A[i, j] = 1 means: page j has a hyperlink that points TO page i.
    (So columns are "from" pages, rows are "to" pages.)

    The Markov matrix P must have columns that sum to 1.
    If a column has sum 0 (a "dangling page"), we replace it with 1/n.
    """

    A = np.array(A, dtype=float)
    n = A.shape[0]                       # number of pages (matrix is nxn)
    P = np.zeros_like(A, dtype=float)

    # Sum of each column = total outgoing links from that page.
    col_sums = A.sum(axis=0)

    for j in range(n):
        if col_sums[j] == 0:
            # ----------------------------------------------------------
            # This page has NO outgoing links.
            # In PageRank theory, a dangling page redistributes evenly.
            # So each row gets 1/n in this column.
            # ----------------------------------------------------------
            P[:, j] = 1.0 / n
        else:
            # ----------------------------------------------------------
            # Normal case: divide each link by total outgoing links.
            # Makes the column sum to 1.
            # ----------------------------------------------------------
            P[:, j] = A[:, j] / col_sums[j]

    return P



# ============================================================
# 2. PageRank via power iteration
# ============================================================
def pagerank(P, alpha=0.85, tol=1e-10, max_iter=100):
    """
    Performs PageRank using the power method.

    P : Markov matrix from adjacency
    alpha : damping factor (probability the surfer follows a link)
    (1-alpha) : teleport probability (jumps to random page)

    Returns:
        steady : steady-state probability vector
        history : iterations for plotting
        G : Google matrix
    """

    n = P.shape[0]

    # ----------------------------------------------------------
    # Create the Google matrix:
    # G = αP + (1−α)(1/n) * (matrix of ones)
    # This ensures every column still sums to 1.
    # ----------------------------------------------------------
    G = alpha * P + (1 - alpha) / n * np.ones((n, n))

    # Start with uniform probability: each page = 1/n
    x = np.ones(n) / n
    history = [x.copy()]  # track iterations for plotting

    for iteration in range(max_iter):
        # Multiply by G to perform one "step" of surfing
        x_new = G @ x
        history.append(x_new.copy())

        # Stop if vector barely changes → steady state reached
        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break

        x = x_new

    return x, np.array(history), G



# ============================================================
# 3. Main Program
# ============================================================
def main():
    # ----------------------------------------------------------
    # Define a web of 6 pages using an adjacency matrix.
    # You may change these — this is just an example structure.
    #
    # A[i, j] = 1 means j → i (j links to i)
    # ----------------------------------------------------------
    A = np.array([
        # 0  1  2  3  4  5   (from page j)
        [0, 0, 0, 1, 0, 0],  # page 3 links to page 0
        [1, 0, 0, 0, 0, 0],  # page 0 → page 1
        [0, 1, 0, 0, 0, 0],  # page 1 → page 2
        [0, 1, 0, 0, 0, 0],  # page 1 → page 3
        [0, 0, 1, 0, 0, 0],  # page 2 → page 4
        [0, 0, 1, 0, 0, 0],  # page 2 → page 5
    ], dtype=float)

    # ----------------------------------------------------------
    # Step 1: Build Markov matrix (normalize columns)
    # ----------------------------------------------------------
    P = build_markov_from_adjacency(A)

    # ----------------------------------------------------------
    # Step 2: Run PageRank (power iteration)
    # ----------------------------------------------------------
    steady, history, G = pagerank(P, alpha=0.85, tol=1e-12, max_iter=200)

    # Names for plotting + ranking (optional)
    page_names = ["Google", "Facebook", "YouTube", "X", "Wikipedia", "Amazon"]

    # ----------------------------------------------------------
    # Print final steady-state vector
    # ----------------------------------------------------------
    print("Final PageRank probabilities:\n")
    for i, prob in enumerate(steady):
        print(f"{page_names[i]} (page {i}): {prob:.6f}")

    # ----------------------------------------------------------
    # Rank pages from most to least important
    # ----------------------------------------------------------
    print("\nRanked pages:\n")
    ranking = np.argsort(-steady)  # sort in descending order
    for rank, idx in enumerate(ranking, 1):
        print(f"{rank}. {page_names[idx]} — PR = {steady[idx]:.6f}")


    # ============================================================
    # 4. Plot probability convergence
    # ============================================================
    iterations = np.arange(history.shape[0])

    plt.figure(figsize=(8, 5))
    for i in range(history.shape[1]):
        plt.plot(iterations, history[:, i], marker="o", linewidth=2,
                 label=f"{page_names[i]}")

    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.title("PageRank Convergence (Power Iteration)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pagerank_convergence.png", dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Optional: Final bar chart of PageRank values
    # ----------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.bar(page_names, steady)
    plt.ylabel("PageRank")
    plt.title("Final PageRank Scores")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("pagerank_final.png", dpi=300)
    plt.show()



# ============================================================
# RUN PROGRAM
# ============================================================
if __name__ == "__main__":
    main()
