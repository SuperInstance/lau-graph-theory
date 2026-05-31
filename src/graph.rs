//! Core graph representations and conversions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A weighted edge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

/// Graph representation with adjacency list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// Number of vertices.
    pub n: usize,
    /// Adjacency list: adj[u] = vec of (v, weight).
    pub adj: Vec<Vec<(usize, f64)>>,
    /// Whether the graph is directed.
    pub directed: bool,
}

impl Graph {
    /// Create a new undirected graph with `n` vertices.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![vec![]; n],
            directed: false,
        }
    }

    /// Create a new directed graph with `n` vertices.
    pub fn new_directed(n: usize) -> Self {
        Self {
            n,
            adj: vec![vec![]; n],
            directed: true,
        }
    }

    /// Add an edge (u, v) with given weight.
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        self.adj[u].push((v, weight));
        if !self.directed {
            self.adj[v].push((u, weight));
        }
    }

    /// Add an unweighted edge (weight = 1.0).
    pub fn add_edge_unit(&mut self, u: usize, v: usize) {
        self.add_edge(u, v, 1.0);
    }

    /// Get the edge list.
    pub fn edge_list(&self) -> Vec<Edge> {
        let mut edges = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for u in 0..self.n {
            for &(v, w) in &self.adj[u] {
                let key = if !self.directed {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    (a, b, w.to_bits())
                } else {
                    (u, v, w.to_bits())
                };
                if !seen.contains(&key) {
                    seen.insert(key);
                    edges.push(Edge { from: u, to: v, weight: w });
                }
            }
        }
        edges
    }

    /// Convert to adjacency matrix (n x n). Missing edges are f64::INFINITY.
    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let mut mat = vec![vec![f64::INFINITY; self.n]; self.n];
        for i in 0..self.n {
            mat[i][i] = 0.0;
        }
        for u in 0..self.n {
            for &(v, w) in &self.adj[u] {
                mat[u][v] = w;
            }
        }
        mat
    }

    /// Get the degree of vertex v.
    pub fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    /// Number of edges (counting undirected edges once).
    pub fn edge_count(&self) -> usize {
        if self.directed {
            self.adj.iter().map(|l| l.len()).sum()
        } else {
            self.adj.iter().map(|l| l.len()).sum::<usize>() / 2
        }
    }

    /// Get neighbors of vertex v.
    pub fn neighbors(&self, v: usize) -> &[(usize, f64)] {
        &self.adj[v]
    }

    /// Build a graph from an edge list.
    pub fn from_edge_list(n: usize, edges: &[(usize, usize, f64)], directed: bool) -> Self {
        let mut g = if directed { Self::new_directed(n) } else { Self::new(n) };
        for &(u, v, w) in edges {
            g.add_edge(u, v, w);
        }
        g
    }
}

/// Degree matrix as a flat vector (diagonal entries).
pub fn degree_matrix(g: &Graph) -> Vec<f64> {
    let mut deg = vec![0.0; g.n];
    for u in 0..g.n {
        for &(_, w) in &g.adj[u] {
            deg[u] += w;
        }
    }
    // For undirected, each edge counted twice from adjacency list
    if !g.directed {
        // Already correct: each neighbor entry contributes to degree
    }
    deg
}
