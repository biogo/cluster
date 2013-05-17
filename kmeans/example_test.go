// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kmeans_test

import (
	"code.google.com/p/biogo.cluster/kmeans"
	"fmt"
	"strings"
)

type Feature struct {
	ID    string
	Start int
	End   int
}

func (f *Feature) Len() int { return f.End - f.Start }

type Features []*Feature

func (f Features) Len() int               { return len(f) }
func (f Features) Values(i int) []float64 { return []float64{float64(f[i].Start), float64(f[i].End)} }

var feats = []*Feature{
	{ID: "0", Start: 1, End: 1700},
	{ID: "1", Start: 2, End: 1700},
	{ID: "2", Start: 3, End: 610},
	{ID: "3", Start: 2, End: 605},
	{ID: "4", Start: 1, End: 600},
	{ID: "5", Start: 2, End: 750},
	{ID: "6", Start: 650, End: 900},
	{ID: "7", Start: 700, End: 950},
	{ID: "8", Start: 1000, End: 1700},
	{ID: "9", Start: 950, End: 1712},
	{ID: "10", Start: 1000, End: 1650},
}

// Cluster feat.Features on the basis of location where:
//  epsilon is allowable error, and
//  effort is number of attempts to achieve error < epsilon for any k.
func ClusterFeatures(f []*Feature, epsilon float64, effort int) (*kmeans.Kmeans, error) {
	km, err := kmeans.New(Features(f))
	if err != nil {
		return nil, err
	}

	values := km.Values()
	cut := make([]float64, len(values))
	for i, v := range values {
		v := v.V()
		l := epsilon * (v[1] - v[0])
		cut[i] = l * l
	}

	for k := 1; k <= len(f); k++ {
	ATTEMPT:
		for attempt := 0; attempt < effort; attempt++ {
			km.Seed(k)
			km.Cluster()
			centers := km.Centers()
			for i, v := range values {
				cv := centers[v.Cluster()].V()
				vv := v.V()
				dx, dy := cv[0]-vv[0], cv[1]-vv[1]
				ok := dx*dx+dy*dy < cut[i]
				if !ok {
					continue ATTEMPT
				}
			}
			return km, nil
		}
	}

	panic("cannot reach")
}

func Example() {
	km, err := ClusterFeatures(feats, 0.15, 5)
	if err != nil {
		return
	}
	for ci, c := range km.Clusters() {
		fmt.Printf("Cluster %d:\n", ci)
		for _, i := range c {
			f := feats[i]
			fmt.Printf("%2s %s%s\n",
				f.ID,
				strings.Repeat(" ", f.Start/20),
				strings.Repeat("-", f.Len()/20),
			)
		}
		fmt.Println()
	}

	var within float64
	for _, ss := range km.Within() {
		within += ss
	}
	fmt.Printf("betweenSS / totalSS = %.6f\n", 1-(within/km.Total()))

	// Output:
	// Cluster 0:
	//  0 ------------------------------------------------------------------------------------
	//  1 ------------------------------------------------------------------------------------
	//
	// Cluster 1:
	//  2 ------------------------------
	//  3 ------------------------------
	//  4 -----------------------------
	//  5 -------------------------------------
	//
	// Cluster 2:
	//  6                                 ------------
	//  7                                    ------------
	//
	// Cluster 3:
	//  8                                                   -----------------------------------
	//  9                                                --------------------------------------
	// 10                                                   --------------------------------
	//
	// betweenSS / totalSS = 0.995335
}
