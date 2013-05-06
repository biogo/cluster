// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cluster provides interfaces and types for data clustering.
//
// At this stage only Lloyd's k-means clustering of ℝ² data is supported in subpackages.
package cluster

// Indices is a list of indexes into a array or slice of Values.
type Indices []int

// Clusterer is the common interface implemented by clustering types.
type Clusterer interface {
	// Cluster the data.
	Cluster() error

	// Clusters returns a slice of Indices representing the indices of
	// the original data grouped into clusters.
	Clusters() []Indices

	// Within returns a slice of sum of squares distances for the clusters.
	Within() []float64
	// total returns the total sum of squares distance for the original data.
	Total() float64
}

// R2 is the interrogation interface implemented by ℝ² Clusterers.
type R2 interface {
	// Centers returns a slice of centers of the clusters.
	Centers() []Center
	// Values returns the internal representation of the original data.
	Values() []Value
}

// RN is the interrogative interface implemented by ℝⁿ Clusterers.
type RN interface {
	// Centers returns a slice of centers of the clusters.
	Centers() []NCenter
	// Values returns the internal representation of the original data.
	Values() []NValue
}

// Interface is a type, typically a collection, that satisfies cluster.Interface can be clustered
// by an ℝ² Clusterer. The Clusterer requires that the elements of the collection be enumerated by
// an integer index.
type Interface interface {
	Len() int                    // Return the length of the data slice.
	Values(i int) (x, y float64) // Return the data values for element i as float64.
}

// Weighter is an extension of the Interface that allows values represented by the Interface to be
// differentially weighted.
type Weighter interface {
	Weight(i int) float64 // Return the weight for element i.
}

type val struct {
	x, y, w float64
}

// X returns the x-coordinate of the point.
func (v val) X() float64 { return v.x }

// Y returns the y-coordinate of the point.
func (v val) Y() float64 { return v.y }

// Weight returns the weight of the point.
func (v val) Weight() float64 { return v.w }

// A Value is the representation of a data point within the clustering object.
type Value struct {
	val
	cluster int
}

// Cluster returns the cluster membership of the Value.
func (v Value) Cluster() int { return v.cluster }

// A Center is a representation of a cluster center in ℝ².
type Center struct {
	val
	count int
}

// NewCenter returns a new Center with the coordinates x and y.
func NewCenter(x, y, w float64) Center {
	return Center{val: val{x: x, y: y}}
}

// Count returns the number of members of the Center's cluster.
func (c Center) Count() int { return c.count }

// A type, typically a collection, that satisfies cluster.Interface can be clustered by an ℝⁿ Clusterer.
// The Clusterer requires that the elements of the collection be enumerated by an integer index.
type NInterface interface {
	Len() int                   // Return the length of the data slice.
	Values(i int) (v []float64) // Return the data values for element i as []float64.
}

type nval struct {
	coord []float64
	w     float64
}

// V returns the ith coordinate of the point.
func (v nval) V(i int) float64 { return v.coord[i] }

// Weight returns the weight of the point.
func (v nval) Weight() float64 { return v.w }

// A Value is the representation of a data point within the clustering object.
type NValue struct {
	nval
	cluster int
}

// Cluster returns the cluster membership of the NValue.
func (v NValue) Cluster() int { return v.cluster }

// An NCenter is a representation of a cluster center in ℝⁿ.
type NCenter struct {
	nval
	count int
}

// NewNCenter returns a new Center with the coordinates held in v.
func NewNCenter(v []float64, w float64) NCenter {
	return NCenter{nval: nval{coord: v, w: w}}
}

// Count returns the number of members of the NCenter's cluster.
func (v NCenter) Count() int { return v.count }
