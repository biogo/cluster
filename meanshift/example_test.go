// Copyright ©2012 The bíogo.cluster Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package meanshift_test

import (
	"github.com/biogo/cluster/meanshift"

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

func Example() {
	ms := meanshift.New(Features(feats), meanshift.NewTruncGauss(60, 3), 0.1, 5)
	err := ms.Cluster()
	if err != nil {
		fmt.Println(err)
		return
	}

	for ci, c := range ms.Centers() {
		fmt.Printf("Cluster %d:\n", ci)
		for _, i := range c.Members() {
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
	for _, ss := range ms.Within() {
		within += ss
	}
	fmt.Printf("betweenSS / totalSS = %.6f\n", 1-(within/ms.Total()))

	// Output:
	// Cluster 0:
	//  5 -------------------------------------
	//
	// Cluster 1:
	//  0 ------------------------------------------------------------------------------------
	//  1 ------------------------------------------------------------------------------------
	//
	// Cluster 2:
	//  4 -----------------------------
	//  3 ------------------------------
	//  2 ------------------------------
	//
	// Cluster 3:
	//  7                                    ------------
	//  6                                 ------------
	//
	// Cluster 4:
	//  9                                                --------------------------------------
	//  8                                                   -----------------------------------
	// 10                                                   --------------------------------
	//
	// betweenSS / totalSS = 0.998655
}
