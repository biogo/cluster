// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kmeans_test

import (
	check "launchpad.net/gocheck"
	"math/rand"
	"strings"
	"testing"
)

func Test(t *testing.T) { check.TestingT(t) }

type S struct{}

func (s *S) TearDownSuite(_ *check.C) { rand.Seed(1) } // Reset the seed for the example test.

var _ = check.Suite(&S{})

var (
	seq = []*Feature{
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
		set     []*Feature
		epsilon float64
		effort  int

		clusters [][]int

		// results determined with R
		total  int
		within []float64
	}{
		{
			feats,
			0.15, 5,
			[][]int{{0, 1}, {2, 3, 4, 5}, {6, 7}, {8, 9, 10}},
			4747787,
			[]float64{0.5, 15820.75, 2500, 3829.3333333333335},
		},
		{
			feats,
			0.1, 5,
			[][]int{{8, 9, 10}, {0, 1}, {6}, {2, 3, 4}, {5}, {7}},
			4747787,
			[]float64{3829.3333333333335, 0.5, 0, 52, 0, 0},
		},
		{
			seq,
			0.2, 5,
			[][]int{{3}, {7}, {9}, {1}, {6}, {0}, {5}, {4}, {8}, {2}},
			1650000,
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			seq,
			1, 5,
			[][]int{{4, 5}, {2, 3}, {8, 9}, {0, 1}, {6, 7}},
			1650000,
			[]float64{10000, 10000, 10000, 10000, 10000},
		},
	}
)

// Tests
func (s *S) TestKmeans(c *check.C) {
	for i, t := range tests {
		rand.Seed(1)
		km := ClusterFeatures(t.set, t.epsilon, t.effort)
		clusters := km.Clusters()
		c.Logf("Test %d: epsilon = %.2f effort = %d", i, t.epsilon, t.effort)
		for ci, cl := range clusters {
			c.Logf("Cluster %d:", ci)
			for _, j := range cl {
				f := t.set[j]
				c.Logf("%2s %s%s",
					f.ID,
					strings.Repeat(" ", f.Start/20),
					strings.Repeat("-", f.Len()/20),
				)
			}
		}
		c.Log()
		c.Check(clusters, check.DeepEquals, t.clusters)
		c.Check(int(km.Total()), check.Equals, t.total)
		c.Check(km.Within(), check.DeepEquals, t.within)
	}
}
