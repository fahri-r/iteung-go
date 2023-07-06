package lstm

import (
	"context"
	"io"
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestCost(t *testing.T) {
	hiddenSize := 10
	model := newModelFromBackends(testBackends(5, 5, hiddenSize))
	learnrate := 0.1
	l2reg := 1e-6
	clipVal := float64(5)
	var hiddenT, cellT tensor.Tensor
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))
	tset := &testSet{
		values: [][]float32{
			{1, 0, 0, 0, 0},
			{0, 1, 0, 0, 0},
			{0, 0, 1, 0, 0},
			{0, 0, 0, 1, 0},
			{0, 0, 0, 0, 1},
		},
		expectedValues: []int{1, 2, 3, 4, 0},
	}
	for i := 0; i < 20; i++ {
		tset := &testSet{
			values: [][]float32{
				{1, 0, 0, 0, 0},
				{0, 1, 0, 0, 0},
				{0, 0, 1, 0, 0},
				{0, 0, 0, 1, 0},
				{0, 0, 0, 0, 1},
			},
			expectedValues: []int{1, 2, 3, 4, 0},
		}
		if hiddenT == nil {
			hiddenT = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		}
		if cellT == nil {
			cellT = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		}
		l := model.newLSTM(hiddenT, cellT)
		_, _, hidden, cell, err := l.cost(tset)
		if err != nil {
			t.Fatal(err)
		}
		machine := G.NewLispMachine(l.g)
		if err := machine.RunAll(); err != nil {
			t.Fatalf("Pass: %v, error: %v", i, err)
		}
		solver.Step(G.Nodes{l.biasC, l.biasF, l.biasI, l.biasO, l.biasY,
			l.uc, l.uf, l.ui, l.uo,
			l.wc, l.wf, l.wi, l.wo, l.wy})
		copy(hiddenT.Data().([]float32), hidden.Value().Data().([]float32))
		copy(cellT.Data().([]float32), cell.Value().Data().([]float32))
		tset.flush()
		//l.g.UnbindAllNonInputs()
	}
	getMax := func(a []float32) int {
		max := float32(0)
		idx := 0
		for i, val := range a {
			if val > max {
				idx = i
				max = val
			}
		}
		return idx
	}
	for i, computedValue := range tset.outputValues {
		val := getMax(computedValue)
		if tset.expectedValues[i] != val {
			t.Log(computedValue)
			t.Fatal("Bad result")
		}

	}
}

func TestTrain(t *testing.T) {
	model := newModelFromBackends(testBackends(5, 5, 10))
	tset := &testSet{
		values: [][]float32{
			{1, 0, 0, 0, 0},
			{0, 1, 0, 0, 0},
			{0, 0, 1, 0, 0},
			{0, 0, 0, 1, 0},
			{0, 0, 0, 0, 1},
		},
		expectedValues: []int{1, 2, 3, 4, 0},
		maxEpoch:       10,
	}
	learnrate := 0.01
	l2reg := 1e-6
	clipVal := float64(5)
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))

	pause := make(chan struct{})
	infoChan, errc := model.Train(context.TODO(), tset, solver, pause)
	for infos := range infoChan {
		t.Log(infos)
		for _, computedVector := range tset.GetComputedVectors() {
			t.Log(computedVector.Value().Data().([]float32))
		}
	}
	err := <-errc
	if err == io.EOF {
		close(pause)
		return
	}
	if err != nil && err != io.EOF {
		t.Fatal(err)
	}

}
