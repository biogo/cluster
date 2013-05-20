// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cluster provides interfaces and types for data clustering.
package cluster

// Indices is a list of indexes into a array or slice of Values.
type Indices []int

// Clusterer is the common interface implemented by clustering types.
type Clusterer interface {
	// Cluster the data.
	Cluster() error

	// Centers returns a slice of centers of the clusters.
	Centers() []Center

	// Values returns the internal representation of the original data.
	Values() []Value
}

// Interface is a type, typically a collection, that satisfies cluster.Interface can be clustered
// by an ℝⁿ Clusterer. The Clusterer requires that the elements of the collection be enumerated by
// an integer index.
type Interface interface {
	Len() int               // Return the length of the data slice.
	Values(i int) []float64 // Return the data values for element i as float64.
}

// Weighter is an extension of the Interface that allows values represented by the Interface to be
// differentially weighted.
type Weighter interface {
	Weight(i int) float64 // Return the weight for element i.
}

type Point interface {
	V() []float64
}

// A Value is the representation of a data point within the clustering object.
type Value interface {
	Point
	Cluster() int
}

// A Center is a representation of a cluster center in ℝⁿ.
type Center interface {
	Point
	Cluster() Indices
}
