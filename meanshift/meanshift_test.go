// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package meanshift_test

import (
	"github.com/biogo/cluster/cluster"
	"github.com/biogo/cluster/meanshift"

	"math/rand"
	"strings"
	"testing"

	"gopkg.in/check.v1"
)

func Test(t *testing.T) { check.TestingT(t) }

type S struct{}

func (s *S) TearDownSuite(_ *check.C) { rand.Seed(1) } // Reset the seed for the example test. See note below.

var _ = check.Suite(&S{})

var (
	seq = Features{
		{ID: "0", Start: 0, End: 100},
		{ID: "1", Start: 100, End: 200},
		{ID: "2", Start: 200, End: 300},
		{ID: "3", Start: 300, End: 400},
		{ID: "4", Start: 400, End: 500},
		{ID: "5", Start: 500, End: 600},
		{ID: "6", Start: 600, End: 700},
		{ID: "7", Start: 700, End: 800},
		{ID: "8", Start: 800, End: 900},
		{ID: "9", Start: 900, End: 1000},
	}
	tests = []struct {
		set        Features
		bandwidth  float64
		oversample float64
		effort     int

		clusters []cluster.Indices

		total  int
		within []float64
	}{
		{
			feats,
			60, 3, 5,
			[]cluster.Indices{{5}, {0, 1}, {4, 3, 2}, {7, 6}, {9, 8, 10}},
			4747787,
			[]float64{0, 0.5, 52, 2500, 3833.1023809507415},
		},
		{
			feats,
			200, 3, 100,
			[]cluster.Indices{{1, 0}, {4, 3, 2, 5}, {6, 7}, {10, 8, 9}},
			4747787,
			[]float64{0.5, 15864.884101059888, 2500, 3829.3691610066735},
		},
		{
			seq,
			60, 3, 5,
			[]cluster.Indices{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
			1650000,
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			seq,
			500, 3, 500,
			[]cluster.Indices{{6, 7, 0, 5, 8, 1, 4, 9, 2, 3}},
			1650000,
			[]float64{1650000},
		},
	}
)

// Tests
func (s *S) TestMeanShift(c *check.C) {
	for i, t := range tests {
		// Note that though there does not appear to be any randomness in the approach used here, we use
		// kdtree for storing data. The data are inserted on mass at the creation of the tree based on
		// kdtree.MedianOfRandoms under the current implementation. So seed makes a difference.
		rand.Seed(1)
		c.Logf("Test %d: bandwidth = %.2f effort = %d", i, t.bandwidth, t.effort)
		ms := meanshift.New(t.set, meanshift.NewTruncGauss(t.bandwidth, t.oversample), 0.1, t.effort)
		err := ms.Cluster()
		c.Check(err, check.Equals, nil)
		clusters := ms.Centers()
		for ci, cl := range clusters {
			c.Logf("Cluster %d:", ci)
			for _, j := range cl.Members() {
				f := t.set[j]
				c.Logf("%2s %s%s",
					f.ID,
					strings.Repeat(" ", f.Start/20),
					strings.Repeat("-", f.Len()/20),
				)
			}
			// c.Logf("Values: %v\nCenters: %v", ms.Values(), ms.Centers())
		}
		c.Log()
		for ci, m := range clusters {
			c.Check(m.Members(), check.DeepEquals, t.clusters[ci])
		}
		c.Check(int(ms.Total()), check.Equals, t.total)
		c.Check(ms.Within(), check.DeepEquals, t.within)
	}
}

type bench [][2]float64

func (b bench) Len() int               { return len(b) }
func (b bench) Values(i int) []float64 { return b[i][:] }

var benchData bench = func() bench {
	b := make(bench, 1000)
	for i := 0; i < 20; i++ {
		x, y := float64(rand.Intn(10000)), float64(rand.Intn(10000))
		r := float64(rand.Intn(200))
		for j := range b {
			b[j] = [2]float64{x + r*rand.NormFloat64(), y + r*rand.NormFloat64()}
		}
	}
	return b
}()

func BenchmarkTruncGauss(b *testing.B) {
	s := meanshift.NewTruncGauss(800, 1)
	for i := 0; i < b.N; i++ {
		err := meanshift.New(benchData, s, 20, 5).Cluster()
		if err != nil {
			b.Log(err)
		}
	}
}

func BenchmarkUniform(b *testing.B) {
	s := meanshift.NewUniform(800)
	for i := 0; i < b.N; i++ {
		err := meanshift.New(benchData, s, 20, 5).Cluster()
		if err != nil {
			b.Log(err)
		}
	}
}
