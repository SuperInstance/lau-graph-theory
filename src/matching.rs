//! Matching algorithms: maximum bipartite matching, Hungarian algorithm.

use crate::graph::Graph;

/// Maximum bipartite matching using augmenting paths (Hopcroft-Karp style BFS+DFS).
/// `n_left` = number of vertices on the left partition (vertices 0..n_left-1).
/// `n_right` = number on the right (vertices n_left..n_left+n_right-1).
/// The graph should be undirected bipartite with edges only between left and right.
/// Returns a vector of matched pairs (left_vertex, right_vertex).
pub fn bipartite_matching(g: &Graph, n_left: usize, n_right: usize) -> Vec<(usize, usize)> {
    assert_eq!(g.n, n_left + n_right);
    let nil = usize::MAX;
    let mut pair_u = vec![nil; n_left];
    let mut pair_v = vec![nil; n_right];
    let mut dist = vec![0usize; n_left];

    fn bfs(
        g: &Graph, n_left: usize,
        pair_u: &[usize], pair_v: &[usize],
        dist: &mut Vec<usize>,
    ) -> bool {
        use std::collections::VecDeque;
        let mut queue = VecDeque::new();
        let nil = usize::MAX;
        for u in 0..n_left {
            if pair_u[u] == nil {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = usize::MAX;
            }
        }
        let mut found = false;
        while let Some(u) = queue.pop_front() {
            for &(v_rel, _) in &g.adj[u] {
                if v_rel < n_left { continue; }
                let v = v_rel - n_left;
                if pair_v[v] == nil {
                    found = true;
                } else if dist[pair_v[v]] == usize::MAX {
                    dist[pair_v[v]] = dist[u] + 1;
                    queue.push_back(pair_v[v]);
                }
            }
        }
        found
    }

    fn dfs(
        g: &Graph, u: usize, n_left: usize,
        pair_u: &mut Vec<usize>, pair_v: &mut Vec<usize>,
        dist: &mut Vec<usize>,
    ) -> bool {
        let nil = usize::MAX;
        for &(v_rel, _) in &g.adj[u] {
            if v_rel < n_left { continue; }
            let v = v_rel - n_left;
            if pair_v[v] == nil || (dist[pair_v[v]] == dist[u] + 1 &&
                dfs(g, pair_v[v], n_left, pair_u, pair_v, dist))
            {
                pair_u[u] = v;
                pair_v[v] = u;
                return true;
            }
        }
        dist[u] = usize::MAX;
        false
    }

    let mut matching = 0usize;
    while bfs(g, n_left, &pair_u, &pair_v, &mut dist) {
        for u in 0..n_left {
            if pair_u[u] == usize::MAX {
                if dfs(g, u, n_left, &mut pair_u, &mut pair_v, &mut dist) {
                    matching += 1;
                }
            }
        }
    }

    let mut result = Vec::new();
    for u in 0..n_left {
        if pair_u[u] != usize::MAX {
            result.push((u, pair_u[u] + n_left));
        }
    }
    result
}

/// Hungarian algorithm for minimum weight perfect matching in a bipartite graph.
/// `cost[i][j]` = cost of matching left vertex i with right vertex j.
/// Requires n_left <= n_right. Returns a vector of (left, right) pairs and total cost.
pub fn hungarian(cost: &Vec<Vec<f64>>) -> (Vec<(usize, usize)>, f64) {
    let n = cost.len();
    if n == 0 { return (vec![], 0.0); }
    let m = cost[0].len();
    assert!(n <= m, "Hungarian requires n_left <= n_right");

    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 { break; }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 { break; }
        }
    }

    let mut result = Vec::new();
    let mut total_cost = 0.0;
    for j in 1..=m {
        if p[j] != 0 {
            result.push((p[j] - 1, j - 1));
            total_cost += cost[p[j] - 1][j - 1];
        }
    }
    (result, total_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bipartite_matching_simple() {
        // Left: 0,1,2  Right: 3,4,5
        let mut g = Graph::new(6);
        g.add_edge_unit(0, 3);
        g.add_edge_unit(0, 4);
        g.add_edge_unit(1, 3);
        g.add_edge_unit(2, 5);
        let matching = bipartite_matching(&g, 3, 3);
        assert_eq!(matching.len(), 3);
    }

    #[test]
    fn test_bipartite_matching_partial() {
        // Left: 0,1  Right: 2,3,4
        let mut g = Graph::new(5);
        g.add_edge_unit(0, 2);
        g.add_edge_unit(0, 3);
        g.add_edge_unit(1, 2);
        let matching = bipartite_matching(&g, 2, 3);
        assert_eq!(matching.len(), 2);
    }

    #[test]
    fn test_bipartite_matching_no_match() {
        let mut g = Graph::new(4);
        // No edges
        let matching = bipartite_matching(&g, 2, 2);
        assert!(matching.is_empty());
    }

    #[test]
    fn test_hungarian() {
        let cost = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 1.0, 2.0],
            vec![2.0, 3.0, 1.0],
        ];
        let (matching, total) = hungarian(&cost);
        assert_eq!(matching.len(), 3);
        assert!((total - 3.0).abs() < 1e-9); // 1+1+1 = 3
    }

    #[test]
    fn test_hungarian_rect() {
        let cost = vec![
            vec![1.0, 3.0],
            vec![3.0, 1.0],
        ];
        let (matching, total) = hungarian(&cost);
        assert_eq!(matching.len(), 2);
        assert!((total - 2.0).abs() < 1e-9); // 1+1
    }
}
