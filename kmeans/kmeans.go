// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package kmeans provides Lloyd's k-means clustering for ℝ² data.
package kmeans

import (
	"code.google.com/p/biogo.cluster"
	"errors"
	"math/rand"
	"unsafe"
)

// These types mirror the definitions in cluster.
type (
	val struct {
		x, y float64
	}
	value struct {
		val
		cluster int
	}
	center struct {
		val
		count int
	}
)

// A Kmeans clusters ℝ² data according to the Lloyd k-means algorithm.
type Kmeans struct {
	values []value
	means  []center
}

// NewKmeans creates a new k-means Clusterer object populated with data from an Interface value, data.
func NewKmeans(data cluster.Interface) *Kmeans {
	return &Kmeans{
		values: convert(data),
	}
}

// Convert the data to the internal float64 representation.
func convert(data cluster.Interface) []value {
	va := make([]value, data.Len())
	for i := 0; i < data.Len(); i++ {
		x, y := data.Values(i)
		va[i] = value{val: val{x: x, y: y}}
	}

	return va
}

// Seed generates the initial means for the k-means algorithm.
func (km *Kmeans) Seed(k int) {
	km.means = make([]center, k)

	km.means[0].val = km.values[rand.Intn(len(km.values))].val
	if k == 1 {
		return
	}
	d := make([]float64, len(km.values))
	for i := 1; i < k; i++ {
		sum := 0.
		for j, v := range km.values {
			_, min := km.nearest(v.val)
			d[j] = min
			sum += d[j]
		}
		target := rand.Float64() * sum
		j := 0
		for sum = d[0]; sum < target; sum += d[j] {
			j++
		}
		km.means[i].val = km.values[j].val
	}
}

// Find the nearest center to the point v. Returns c, the index of the nearest center
// and min, the square of the distance from v to that center.
func (km *Kmeans) nearest(v val) (c int, min float64) {
	xd, yd := v.x-km.means[0].x, v.y-km.means[0].y
	min = xd*xd + yd*yd

	for i := 1; i < len(km.means); i++ {
		xd, yd = v.x-km.means[i].x, v.y-km.means[i].y
		d := xd*xd + yd*yd
		if d < min {
			min = d
			c = i
		}
	}

	return c, min
}

// Cluster the data using the standard k-means algorithm.
func (km *Kmeans) Cluster() error {
	if len(km.means) == 0 {
		return errors.New("kmeans: no centers")
	}
	for i, v := range km.values {
		n, _ := km.nearest(v.val)
		km.values[i].cluster = n
	}

	for {
		for i := range km.means {
			km.means[i] = center{}
		}
		for _, v := range km.values {
			km.means[v.cluster].x += v.x
			km.means[v.cluster].y += v.y
			km.means[v.cluster].count++
		}
		for i := range km.means {
			inv := 1 / float64(km.means[i].count)
			km.means[i].x *= inv
			km.means[i].y *= inv
		}

		deltas := 0
		for i, v := range km.values {
			if n, _ := km.nearest(v.val); n != v.cluster {
				deltas++
				km.values[i].cluster = n
			}
		}
		if deltas == 0 {
			break
		}
	}
	return nil
}

// Within calculates the total sum of squares for the data relative to the data mean.
func (km *Kmeans) Total() (ss float64) {
	var x, y float64

	for _, v := range km.values {
		x += v.x
		y += v.y
	}
	inv := 1 / float64(len(km.values))
	x *= inv
	y *= inv

	for _, v := range km.values {
		dx, dy := x-v.x, y-v.y
		ss += dx*dx + dy*dy
	}

	return
}

// Within calculates the sum of squares within each cluster.
// Returns nil if Cluster has not been called.
func (km *Kmeans) Within() (ss []float64) {
	if km.means == nil {
		return
	}
	ss = make([]float64, len(km.means))

	for _, v := range km.values {
		dx, dy := km.means[v.cluster].x-v.x, km.means[v.cluster].y-v.y
		ss[v.cluster] += dx*dx + dy*dy
	}

	return
}

// Means returns the k-means.
func (km *Kmeans) Means() (c []cluster.Center) {
	return *(*[]cluster.Center)(unsafe.Pointer(&km.means))
}

// Features returns a slice of the values in the Kmeans.
func (km *Kmeans) Values() (v []cluster.Value) {
	return *(*[]cluster.Value)(unsafe.Pointer(&km.values))
}

// Clusters returns the k clusters.
// Returns nil if Cluster has not been called.
func (km *Kmeans) Clusters() (c [][]int) {
	if km.means == nil {
		return
	}
	c = make([][]int, len(km.means))

	for i := range c {
		c[i] = make([]int, 0, km.means[i].count)
	}
	for i, v := range km.values {
		c[v.cluster] = append(c[v.cluster], i)
	}

	return
}
