// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package meanshift provides mean shift clustering for ℝⁿ data.
package meanshift

import (
	"code.google.com/p/biogo.cluster"
	"code.google.com/p/biogo.kdtree"
	"fmt"
)

type pnt []float64

func (p pnt) V() []float64 { return p }

type value struct {
	pnt
	w       float64
	cluster int
}

func (v *value) Weight() float64 { return v.w }
func (v *value) Cluster() int    { return v.cluster }

type center struct {
	pnt
	w     float64
	count int
}

func (c *center) Count() int { return c.count }

type Shifter interface {
	Init(cluster.Interface)
	Shift() float64
	Bandwidth() float64
	Centers() kdtree.Interface
}

// A MeanShift clusters ℝⁿ data according to the mean shift algorithm.
type MeanShift struct {
	k       Shifter
	tol     float64
	maxIter int
	values  []value
	centers []center
	ci      []cluster.Indices
}

// New creates a new mean shift Clusterer object populated with data from an Interface value, data
// and using the Kernel k.
func New(data cluster.Interface, k Shifter, tol float64, maxIter int) *MeanShift {
	k.Init(data)
	return &MeanShift{
		k:       k,
		tol:     tol,
		maxIter: maxIter,
		values:  convert(data),
	}
}

// Convert the data to the internal float64 representation.
func convert(data cluster.Interface) []value {
	va := make([]value, data.Len())
	for i := 0; i < data.Len(); i++ {
		va[i] = value{pnt: append(pnt(nil), data.Values(i)...)}
	}
	if w, ok := data.(cluster.Weighter); ok {
		for i := 0; i < data.Len(); i++ {
			va[i].w = w.Weight(i)
		}
	} else {
		for i := 0; i < data.Len(); i++ {
			va[i].w = 1
		}
	}

	return va
}

// Cluster the data using the mean shift algorithm.
func (ms *MeanShift) Cluster() error {
	for i := 0; ; i++ {
		delta := ms.k.Shift()
		if delta <= ms.tol {
			break
		}
		if i > ms.maxIter {
			return fmt.Errorf("meanshift: exceeded max iterations: delta=%f", delta)
		}
	}

	var (
		kc        = ms.k.Centers()
		ct        = kdtree.New(kc, false)
		neighbors = kdtree.NewDistKeeper(ms.k.Bandwidth())
		centers   kdtree.Tree
	)
	for i := 0; i < kc.Len(); i++ {
		ct.NearestSet(neighbors, kc.Index(i))

		wp := &ShiftPoint{Point: make(kdtree.Point, len(ms.values[0].pnt))}
		for _, c := range neighbors.Heap[:len(neighbors.Heap)-1] {
			p := c.Comparable.(*ShiftPoint)
			if p.ID >= 0 {
				wp.Members = append(wp.Members, p.ID)
				p.ID = -1
			}
			for j := range wp.Point {
				wp.Point[j] += p.Point[j] / float64(len(neighbors.Heap)-1)
			}
		}

		if _, d := centers.Nearest(wp); d != 0 {
			centers.Insert(wp, false)
		}

		neighbors.Heap[0] = kdtree.ComparableDist{Comparable: nil, Dist: ms.k.Bandwidth()}
		neighbors.Heap = neighbors.Heap[:1]
	}

	ms.ci = make([]cluster.Indices, 0, centers.Len())
	ms.centers = make([]center, 0, centers.Len())
	centers.Do(func(c kdtree.Comparable, _ *kdtree.Bounding, _ int) (done bool) {
		p := c.(*ShiftPoint)
		if len(p.Members) == 0 {
			return
		}
		for _, cm := range p.Members {
			ms.values[cm].cluster = len(ms.ci)
		}
		ms.ci = append(ms.ci, p.Members)
		ms.centers = append(ms.centers, center{pnt: p.Point, count: len(p.Members)})
		return
	})

	return nil
}

// Total calculates the total sum of squares for the data relative to the data mean.
func (ms *MeanShift) Total() float64 {
	p := make([]float64, len(ms.values[0].pnt))

	for _, v := range ms.values {
		for i := range p {
			p[i] += v.pnt[i]
		}
	}
	inv := 1 / float64(len(ms.values))
	for i := range p {
		p[i] *= inv
	}

	var ss float64
	for _, v := range ms.values {
		for i := range p {
			d := p[i] - v.pnt[i]
			ss += d * d
		}
	}

	return ss
}

// Within calculates the sum of squares within each cluster.
// Returns nil if Cluster has not been called.
func (ms *MeanShift) Within() []float64 {
	if ms.centers == nil {
		return nil
	}
	ss := make([]float64, len(ms.centers))

	for _, v := range ms.values {
		for i := range ms.centers[0].pnt {
			d := ms.centers[v.cluster].pnt[i] - v.pnt[i]
			ss[v.cluster] += d * d
		}
	}

	return ss
}

// Centers returns the centers.
func (ms *MeanShift) Centers() []cluster.Center {
	cs := make([]cluster.Center, len(ms.centers))
	for i := range ms.centers {
		cs[i] = &ms.centers[i]
	}
	return cs
}

// Features returns a slice of the values in the MeanShift.
func (ms *MeanShift) Values() []cluster.Value {
	vs := make([]cluster.Value, len(ms.values))
	for i := range ms.values {
		vs[i] = &ms.values[i]
	}
	return vs
}

// Clusters returns the k clusters.
// Returns nil if Cluster has not been called.
func (ms *MeanShift) Clusters() []cluster.Indices {
	return ms.ci
}
