// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package meanshift

import (
	"code.google.com/p/biogo.cluster"
	"code.google.com/p/biogo.kdtree"
	"math"
)

// shiftPoint is a weighted point which carries group identity and membership information.
// shiftPoint satisfies the kdtree.Comparable interface.
type shiftPoint struct {
	Point   []float64
	Weight  float64
	ID      int
	Members []int
}

func (p *shiftPoint) Clone() kdtree.Comparable {
	return &shiftPoint{Point: append(kdtree.Point(nil), p.Point...), Weight: p.Weight}
}
func (p *shiftPoint) Compare(c kdtree.Comparable, d kdtree.Dim) float64 {
	q := c.(*shiftPoint)
	return p.Point[d] - q.Point[d]
}
func (p *shiftPoint) Dims() int { return len(p.Point) }
func (p *shiftPoint) Distance(c kdtree.Comparable) float64 {
	q := c.(*shiftPoint)
	var sum float64
	for dim, c := range p.Point {
		d := c - q.Point[dim]
		sum += d * d
	}
	return sum
}

// shiftPoints is a collection of shiftPoint values that satisfies the kdtree.Interface.
type shiftPoints []*shiftPoint

func (p shiftPoints) Index(i int) kdtree.Comparable         { return p[i] }
func (p shiftPoints) Len() int                              { return len(p) }
func (p shiftPoints) Pivot(d kdtree.Dim) int                { return plane{shiftPoints: p, Dim: d}.Pivot() }
func (p shiftPoints) Slice(start, end int) kdtree.Interface { return p[start:end] }
func (p shiftPoints) Values(i int) []float64                { return p[i].Point }

// plane wraps a shiftPoints type allowing it to be pivoted on a dimension.
type plane struct {
	kdtree.Dim
	shiftPoints
}

func (p plane) Less(i, j int) bool {
	return p.shiftPoints[i].Point[p.Dim] < p.shiftPoints[j].Point[p.Dim]
}
func (p plane) Pivot() int { return kdtree.Partition(p, kdtree.MedianOfRandoms(p, kdtree.Randoms)) }
func (p plane) Slice(start, end int) kdtree.SortSlicer {
	p.shiftPoints = p.shiftPoints[start:end]
	return p
}
func (p plane) Swap(i, j int) { p.shiftPoints[i], p.shiftPoints[j] = p.shiftPoints[j], p.shiftPoints[i] }

type Uniform struct {
	centers []*shiftPoint
	cn      []float64
	tree    *kdtree.Tree
	hits    *kdtree.DistKeeper
}

func NewUniform(h float64) *Uniform {
	return &Uniform{
		hits: kdtree.NewDistKeeper(h * h),
	}
}

func (s *Uniform) Init(data cluster.Interface) {
	w, isWeighter := data.(cluster.Weighter)

	s.centers = make([]*shiftPoint, data.Len())
	vals := make(shiftPoints, data.Len())

	for i := 0; i < data.Len(); i++ {
		s.centers[i] = &shiftPoint{ID: i}
		s.centers[i].Point = append([]float64(nil), data.Values(i)...)
		v := &shiftPoint{Point: data.Values(i)}
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
			h := hit.Comparable.(*shiftPoint)
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

func (s *Uniform) Centers() []cluster.Center {
	return collate(shiftPoints(s.centers), s.Bandwidth())
}

type TruncGauss struct {
	h       float64
	centers []*shiftPoint
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

func (s *TruncGauss) Init(data cluster.Interface) {
	w, isWeighter := data.(cluster.Weighter)

	s.centers = make([]*shiftPoint, data.Len())
	vals := make(shiftPoints, data.Len())

	for i := 0; i < data.Len(); i++ {
		s.centers[i] = &shiftPoint{ID: i}
		s.centers[i].Point = append([]float64(nil), data.Values(i)...)
		v := &shiftPoint{Point: data.Values(i)}
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
			h := hit.Comparable.(*shiftPoint)
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

func (s *TruncGauss) Centers() []cluster.Center {
	return collate(shiftPoints(s.centers), s.Bandwidth())
}

func collate(kc kdtree.Interface, h float64) []cluster.Center {
	var (
		ct        = kdtree.New(kc, false)
		neighbors = kdtree.NewDistKeeper(h)
		centers   kdtree.Tree
	)
	for i := 0; i < kc.Len(); i++ {
		ct.NearestSet(neighbors, kc.Index(i))

		wp := &shiftPoint{Point: make(kdtree.Point, kc.Index(0).Dims())}
		for _, c := range neighbors.Heap[:len(neighbors.Heap)-1] {
			p := c.Comparable.(*shiftPoint)
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

		neighbors.Heap[0] = kdtree.ComparableDist{Comparable: nil, Dist: h}
		neighbors.Heap = neighbors.Heap[:1]
	}

	cen := make([]cluster.Center, 0, centers.Len())
	centers.Do(func(c kdtree.Comparable, _ *kdtree.Bounding, _ int) (done bool) {
		p := c.(*shiftPoint)
		if len(p.Members) == 0 {
			return
		}
		cen = append(cen, &center{pnt: p.Point, indices: p.Members})
		return
	})

	return cen
}
