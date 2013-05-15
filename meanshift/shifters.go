// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package meanshift

import (
	"code.google.com/p/biogo.cluster"
	"code.google.com/p/biogo.kdtree"
	"math"
)

// A point is a weighted point.
type point struct {
	ID      int
	Point   []float64
	Weight  float64
	Members []int
}

func (p *point) Clone() kdtree.Comparable {
	return &point{Point: append(kdtree.Point(nil), p.Point...), Weight: p.Weight}
}
func (p *point) Compare(c kdtree.Comparable, d kdtree.Dim) float64 {
	q := c.(*point)
	return p.Point[d] - q.Point[d]
}
func (p *point) Dims() int { return len(p.Point) }
func (p *point) Distance(c kdtree.Comparable) float64 {
	q := c.(*point)
	var sum float64
	for dim, c := range p.Point {
		d := c - q.Point[dim]
		sum += d * d
	}
	return sum
}

// A points is a collection of point values that satisfies the kdtree.Interface.
type points []*point

func (p points) Index(i int) kdtree.Comparable         { return p[i] }
func (p points) Len() int                              { return len(p) }
func (p points) Pivot(d kdtree.Dim) int                { return plane{points: p, Dim: d}.Pivot() }
func (p points) Slice(start, end int) kdtree.Interface { return p[start:end] }
func (p points) Values(i int) []float64                { return p[i].Point }

// An plane is a wrapping type that allows a points type be pivoted on a dimension.
type plane struct {
	kdtree.Dim
	points
}

func (p plane) Less(i, j int) bool                     { return p.points[i].Point[p.Dim] < p.points[j].Point[p.Dim] }
func (p plane) Pivot() int                             { return kdtree.Partition(p, kdtree.MedianOfRandoms(p, kdtree.Randoms)) }
func (p plane) Slice(start, end int) kdtree.SortSlicer { p.points = p.points[start:end]; return p }
func (p plane) Swap(i, j int)                          { p.points[i], p.points[j] = p.points[j], p.points[i] }

type Uniform struct {
	centers []*point
	cn      []float64
	tree    *kdtree.Tree
	hits    *kdtree.DistKeeper
}

func NewUniform(h float64) *Uniform {
	return &Uniform{
		hits: kdtree.NewDistKeeper(h * h),
	}
}

func (s *Uniform) Init(data cluster.NInterface) {
	w, isWeighter := data.(cluster.Weighter)

	s.centers = make([]*point, data.Len())
	vals := make(points, data.Len())

	for i := 0; i < data.Len(); i++ {
		s.centers[i] = &point{ID: i}
		s.centers[i].Point = append([]float64(nil), data.Values(i)...)
		v := &point{Point: data.Values(i)}
		if isWeighter {
			v.Weight = w.Weight(i)
		} else {
			v.Weight = 1
		}
		vals[i] = v
	}

	s.tree = kdtree.New(vals, false)
	s.cn = make([]float64, len(s.centers[0].Point))
}

func (s *Uniform) Bandwidth() float64 { return s.hits.Heap[len(s.hits.Heap)-1].Dist }

func (s *Uniform) Shift() (delta float64) {
	for i, c := range s.centers {
		s.tree.NearestSet(s.hits, c)

		div := 0.
		for _, hit := range s.hits.Heap[:len(s.hits.Heap)-1] {
			h := hit.Comparable.(*point)
			div += h.Weight
			for j := range s.cn {
				s.cn[j] += h.Point[j] * h.Weight
			}
		}
		for j := range s.cn {
			s.cn[j] /= div
			delta += (c.Point[j] - s.cn[j]) * (c.Point[j] - s.cn[j])
		}
		copy(s.centers[i].Point, s.cn)

		for j := range s.cn {
			s.cn[j] = 0
		}
		s.hits.Heap[0] = kdtree.ComparableDist{Comparable: nil, Dist: s.hits.Heap[len(s.hits.Heap)-1].Dist}
		s.hits.Heap = s.hits.Heap[:1]
	}

	return delta
}

func (s *Uniform) Centers() kdtree.Interface {
	return points(s.centers)
}

type TruncGauss struct {
	h       float64
	centers []*point
	cn      []float64
	tree    *kdtree.Tree
	hits    *kdtree.DistKeeper
}

func NewTruncGauss(h, oversample float64) *TruncGauss {
	return &TruncGauss{
		h:    h,
		hits: kdtree.NewDistKeeper(h * h * oversample),
	}
}

func (s *TruncGauss) Init(data cluster.NInterface) {
	w, isWeighter := data.(cluster.Weighter)

	s.centers = make([]*point, data.Len())
	vals := make(points, data.Len())

	for i := 0; i < data.Len(); i++ {
		s.centers[i] = &point{ID: i}
		s.centers[i].Point = append([]float64(nil), data.Values(i)...)
		v := &point{Point: data.Values(i)}
		if isWeighter {
			v.Weight = w.Weight(i)
		} else {
			v.Weight = 1
		}
		vals[i] = v
	}

	s.tree = kdtree.New(vals, false)
	s.cn = make([]float64, len(s.centers[0].Point))
}

func (s *TruncGauss) Bandwidth() float64 { return s.h }

func (s *TruncGauss) Shift() (delta float64) {
	inv := 1 / (2 * s.h * s.h)
	for i, c := range s.centers {
		s.tree.NearestSet(s.hits, c)

		div := 0.
		for _, hit := range s.hits.Heap[:len(s.hits.Heap)-1] {
			h := hit.Comparable.(*point)
			kfn := h.Weight * math.Exp(hit.Comparable.Distance(c)*inv)
			div += kfn
			for j := range s.cn {
				s.cn[j] += h.Point[j] * kfn
			}
		}
		for j := range s.cn {
			s.cn[j] /= div
			delta += (c.Point[j] - s.cn[j]) * (c.Point[j] - s.cn[j])
		}
		copy(s.centers[i].Point, s.cn)

		for j := range s.cn {
			s.cn[j] = 0
		}
		s.hits.Heap[0] = kdtree.ComparableDist{Comparable: nil, Dist: s.hits.Heap[len(s.hits.Heap)-1].Dist}
		s.hits.Heap = s.hits.Heap[:1]
	}

	return delta
}

func (s *TruncGauss) Centers() kdtree.Interface {
	return points(s.centers)
}
