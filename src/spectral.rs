//! Spectral graph theory: Laplacian spectrum, algebraic connectivity, Fiedler vector, Cheeger inequality.

use crate::graph::Graph;
use nalgebra::{DMatrix, DVector};

/// Compute the graph Laplacian matrix L = D - A.
pub fn laplacian_matrix(g: &Graph) -> DMatrix<f64> {
    let n = g.n;
    let mut a = vec![0.0f64; n * n];
    for u in 0..n {
        for &(v, w) in &g.adj[u] {
            a[u * n + v] += w;
        }
    }
    let mut d = vec![0.0f64; n * n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += a[i * n + j];
        }
        d[i * n + i] = row_sum;
    }
    let mut l_data = vec![0.0f64; n * n];
    for i in 0..n * n {
        l_data[i] = d[i] - a[i];
    }
    DMatrix::from_row_slice(n, n, &l_data)
}

/// Compute eigenvalues of the Laplacian (sorted ascending).
/// Returns a vector of eigenvalue real parts.
pub fn laplacian_spectrum(g: &Graph) -> Vec<f64> {
    let l = laplacian_matrix(g);
    let eigen = l.symmetric_eigen();
    let mut vals: Vec<f64> = eigen.eigenvalues.iter().map(|x| *x).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals
}

/// Algebraic connectivity: second-smallest eigenvalue of the Laplacian.
pub fn algebraic_connectivity(g: &Graph) -> f64 {
    let spectrum = laplacian_spectrum(g);
    if spectrum.len() >= 2 {
        spectrum[1]
    } else {
        0.0
    }
}

/// Fiedler vector: eigenvector corresponding to the second-smallest eigenvalue.
/// Returns the Fiedler vector and the corresponding eigenvalue.
pub fn fiedler_vector(g: &Graph) -> (DVector<f64>, f64) {
    let l = laplacian_matrix(g);
    let eigen = l.symmetric_eigen();
    let n = g.n;

    // Find eigenvalues and sort by value
    let mut indexed: Vec<(usize, f64)> = eigen.eigenvalues.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if n < 2 {
        return (DVector::zeros(n), 0.0);
    }

    let fiedler_idx = indexed[1].0;
    let fiedler_val = indexed[1].1;
    let fiedler_vec = eigen.eigenvectors.column(fiedler_idx).into_owned();
    (fiedler_vec, fiedler_val)
}

/// Compute the Cheeger constant (edge expansion) of a graph.
/// h(G) = min over all subsets S with |S| <= n/2 of (edges crossing cut) / |S|.
/// This is expensive (exponential) for large graphs; we use a spectral approximation.
pub fn cheeger_constant_approx(g: &Graph) -> f64 {
    let (fiedler, _) = fiedler_vector(g);
    let n = g.n;
    if n < 2 { return 0.0; }

    // Sort vertices by Fiedler vector value
    let mut indexed: Vec<(usize, f64)> = fiedler.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Try all prefix cuts (Swarup/Spielman-Teng approach)
    let mut best = f64::INFINITY;
    let mut s = std::collections::HashSet::new();
    for (i, &(v, _)) in indexed.iter().enumerate() {
        s.insert(v);
        if s.len() > n / 2 { break; }

        // Count edges crossing the cut
        let mut crossing = 0.0f64;
        for &u in &s {
            for &(w, weight) in &g.adj[u] {
                if !s.contains(&w) {
                    crossing += weight;
                }
            }
        }
        let h = crossing / s.len() as f64;
        best = best.min(h);
    }
    best
}

/// Cheeger inequality bounds: lambda_2/2 <= h(G) <= sqrt(2 * lambda_2).
/// Returns (lower_bound, upper_bound) for the Cheeger constant.
pub fn cheeger_inequality_bounds(g: &Graph) -> (f64, f64) {
    let lambda2 = algebraic_connectivity(g);
    let lower = lambda2 / 2.0;
    let upper = (2.0 * lambda2).sqrt();
    (lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_complete() {
        // K3: Laplacian is [[2,-1,-1],[-1,2,-1],[-1,-1,2]]
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(1, 2, 1.0);
        let l = laplacian_matrix(&g);
        assert!((l[(0, 0)] - 2.0).abs() < 1e-9);
        assert!((l[(0, 1)] - (-1.0)).abs() < 1e-9);
        assert!((l[(1, 0)] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_spectrum_connected() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let spectrum = laplacian_spectrum(&g);
        assert!(spectrum[0].abs() < 1e-9); // smallest eigenvalue is 0
        assert!(spectrum[1] > 0.0); // connected => lambda2 > 0
    }

    #[test]
    fn test_algebraic_connectivity_disconnected() {
        let g = Graph::new(3); // 3 isolated vertices
        let ac = algebraic_connectivity(&g);
        assert!(ac.abs() < 1e-9);
    }

    #[test]
    fn test_algebraic_connectivity_connected() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let ac = algebraic_connectivity(&g);
        assert!(ac > 0.0);
    }

    #[test]
    fn test_fiedler_vector_sums() {
        // For a path graph 0-1-2, Fiedler vector should separate {0} from {2}
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let (vec, val) = fiedler_vector(&g);
        assert!(val > 0.0);
        // Fiedler vector is orthogonal to the all-ones vector => sums to ~0
        let sum: f64 = vec.iter().sum();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn test_cheeger_bounds() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(3, 0, 1.0);
        let (lower, upper) = cheeger_inequality_bounds(&g);
        assert!(lower > 0.0);
        assert!(upper >= lower);
    }

    #[test]
    fn test_spectrum_complete_graph() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i+1)..4 {
                g.add_edge(i, j, 1.0);
            }
        }
        let spectrum = laplacian_spectrum(&g);
        assert!(spectrum[0].abs() < 1e-9); // 0
        assert!((spectrum[1] - 4.0).abs() < 1e-9); // K4 => lambda2 = n = 4
    }
}
