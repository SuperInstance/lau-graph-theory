//! Minimum spanning tree: Kruskal and Prim.

use crate::graph::{Graph, Edge};

/// Union-Find (Disjoint Set Union) with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry { return false; }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
        true
    }
}

/// Kruskal's MST algorithm. Returns the MST edges and total weight.
pub fn kruskal(g: &Graph) -> (Vec<Edge>, f64) {
    let mut edges: Vec<Edge> = g.edge_list();
    edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal));
    let mut uf = UnionFind::new(g.n);
    let mut mst = Vec::new();
    let mut total = 0.0;
    for e in edges {
        if uf.union(e.from, e.to) {
            total += e.weight;
            mst.push(e.clone());
            if mst.len() == g.n - 1 { break; }
        }
    }
    (mst, total)
}

/// Prim's MST algorithm starting from vertex `start`.
pub fn prim(g: &Graph, start: usize) -> (Vec<Edge>, f64) {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    #[derive(Debug, Copy, Clone)]
    struct EdgeState {
        weight: f64,
        from: usize,
        to: usize,
    }

    impl PartialEq for EdgeState {
        fn eq(&self, other: &Self) -> bool {
            self.weight == other.weight
        }
    }

    impl Eq for EdgeState {}

    impl PartialOrd for EdgeState {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            other.weight.partial_cmp(&self.weight)
        }
    }

    impl Ord for EdgeState {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap_or(Ordering::Equal)
        }
    }

    let mut in_mst = vec![false; g.n];
    let mut heap = BinaryHeap::new();
    let mut mst = Vec::new();
    let mut total = 0.0;
    in_mst[start] = true;
    for &(v, w) in &g.adj[start] {
        heap.push(EdgeState { weight: w, from: start, to: v });
    }
    while let Some(EdgeState { weight, from, to }) = heap.pop() {
        if in_mst[to] { continue; }
        in_mst[to] = true;
        mst.push(Edge { from, to, weight });
        total += weight;
        for &(v, w) in &g.adj[to] {
            if !in_mst[v] {
                heap.push(EdgeState { weight: w, from: to, to: v });
            }
        }
    }
    (mst, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kruskal() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 2.0);
        g.add_edge(1, 2, 3.0);
        g.add_edge(1, 3, 4.0);
        g.add_edge(2, 3, 5.0);
        let (mst, total) = kruskal(&g);
        assert_eq!(mst.len(), 3);
        assert!((total - 7.0).abs() < 1e-9); // 1+2+4
    }

    #[test]
    fn test_prim() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 2.0);
        g.add_edge(1, 2, 3.0);
        g.add_edge(1, 3, 4.0);
        g.add_edge(2, 3, 5.0);
        let (mst, total) = prim(&g, 0);
        assert_eq!(mst.len(), 3);
        assert!((total - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_mst_kruskal_prim_agree() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1, 2.0);
        g.add_edge(0, 3, 6.0);
        g.add_edge(1, 2, 3.0);
        g.add_edge(1, 3, 8.0);
        g.add_edge(1, 4, 5.0);
        g.add_edge(2, 4, 7.0);
        g.add_edge(3, 4, 9.0);
        let (_, tw) = kruskal(&g);
        let (_, tp) = prim(&g, 0);
        assert!((tw - tp).abs() < 1e-9);
        assert!((tw - 16.0).abs() < 1e-9); // 2+3+5+6
    }
}
