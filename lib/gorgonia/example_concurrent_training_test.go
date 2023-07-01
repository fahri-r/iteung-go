package gorgonia_test

import (
	"fmt"
	"runtime"
	"sync"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const (
	// rows      = 373127
	// cols      = 53

	// We'll use a nice even sized batch size, instead of weird prime numbers
	rows      = 30000
	cols      = 5
	batchSize = 100
	epochs    = 10
)

type concurrentTrainer struct {
	g    *ExprGraph
	x, y *Node
	vm   VM
	// cost Value

	batchSize int
	epoch     int // number of epochs done
}

func newConcurrentTrainer() *concurrentTrainer {
	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(batchSize, cols), WithName("x"))
	y := NewVector(g, Float64, WithShape(batchSize), WithName("y"))
	xT := Must(Transpose(x))
	z := Must(Mul(xT, y))
	sz := Must(Sum(z))

	// Read(sz, &ct.cost)
	Grad(sz, x, y)
	vm := NewTapeMachine(g, BindDualValues(x, y))

	return &concurrentTrainer{
		g:  g,
		x:  x,
		y:  y,
		vm: vm,

		batchSize: batchSize,
		epoch:     -1,
	}
}

type cost struct {
	Nodes []ValueGrad
	VM    // placed for debugging purposes. In real life use you can just use a channel of Nodes

	// cost Value
}

func (t *concurrentTrainer) train(x, y Value, costChan chan cost, wg *sync.WaitGroup) {
	Let(t.x, x)
	Let(t.y, y)
	if err := t.vm.RunAll(); err != nil {
		panic("HELP")
	}

	costChan <- cost{
		[]ValueGrad{t.x, t.y},
		t.vm,
		// t.cost,
	}

	t.vm.Reset()
	wg.Done()
}

func trainEpoch(bs []batch, ts []*concurrentTrainer, threads int) {
	// costs := make([]float64, 0, len(bs))
	chunks := len(bs) / len(ts)
	for chunk := 0; chunk <= chunks; chunk++ {
		costChan := make(chan cost, len(bs))

		var wg sync.WaitGroup
		for i, t := range ts {
			idx := chunk*threads + i
			if idx >= len(bs) {
				break
			}
			b := bs[idx]

			wg.Add(1)
			go t.train(b.xs, b.ys, costChan, &wg)
		}
		wg.Wait()
		close(costChan)

		solver := NewVanillaSolver(WithLearnRate(0.01), WithBatchSize(batchSize))
		for cost := range costChan {
			// y := cost.Nodes[1].Value()
			// yG, _ := cost.Nodes[1].Grad()
			// c := cost.cost.Data().(float64)
			// costs = append(costs, c)
			solver.Step(cost.Nodes)
		}
	}

	// var avg float64
	// for _, c := range costs {
	// 	avg += c
	// }
	// avg /= float64(len(costs))
}

type batch struct {
	xs Value
	ys Value
}

func prep() (x, y Value, bs []batch) {
	xV := tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(tensor.Range(Float64, 0, cols*rows)))
	yV := tensor.New(tensor.WithShape(rows), tensor.WithBacking(tensor.Range(Float64, 0, rows)))

	// prep the data: y = ΣnX, where n = col ID, x ∈ X = colID / 100
	xData := xV.Data().([]float64)
	yData := yV.Data().([]float64)
	for r := 0; r < rows; r++ {
		var sum float64
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			fc := float64(c)
			v := fc * fc / 100
			xData[idx] = v
			sum += v
		}
		yData[r] = sum
	}

	// batch the examples up into their respective batchSize
	for i := 0; i < rows; i += batchSize {
		xVS, _ := xV.Slice(S(i, i+batchSize))
		yVS, _ := yV.Slice(S(i, i+batchSize))
		b := batch{xVS, yVS}
		bs = append(bs, b)
	}
	return xV, yV, bs
}

func concurrentTraining(xV, yV Value, bs []batch, es int) {
	threads := runtime.NumCPU()

	ts := make([]*concurrentTrainer, threads)
	for chunk := 0; chunk < threads; chunk++ {
		trainer := newConcurrentTrainer()
		ts[chunk] = trainer
		defer trainer.vm.Close()
	}

	for e := 0; e < es; e++ {
		trainEpoch(bs, ts, threads)
	}
}

func nonConcurrentTraining(xV, yV Value, es int) {
	g := NewGraph()
	x := NewMatrix(g, Float64, WithValue(xV))
	y := NewVector(g, Float64, WithValue(yV))
	xT := Must(Transpose(x))
	z := Must(Mul(xT, y))
	sz := Must(Sum(z))
	Grad(sz, x, y)
	vm := NewTapeMachine(g, BindDualValues(x, y))

	Let(x, xV)
	Let(y, yV)
	solver := NewVanillaSolver(WithLearnRate(0.01), WithBatchSize(batchSize))
	for i := 0; i < es; i++ {
		vm.RunAll()
		solver.Step([]ValueGrad{x, y})
		vm.Reset()
		runtime.GC()
	}
}

func Example_concurrentTraining() {
	xV, yV, bs := prep()
	concurrentTraining(xV, yV, bs, epochs)

	fmt.Printf("x:\n%1.1v", xV)
	fmt.Printf("y:\n%1.1v", yV)

	// Output:
	// x:
	// ⎡-0.0003     0.01     0.04     0.09      0.2⎤
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// .
	// .
	// .
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎣-0.0003     0.01     0.04     0.09      0.2⎦
	// y:
	// [0.3  0.3  0.3  0.3  ... 0.3  0.3  0.3  0.3]

}

func Example_nonConcurrentTraining() {
	xV, yV, _ := prep()
	nonConcurrentTraining(xV, yV, epochs)

	fmt.Printf("x:\n%1.1v", xV)
	fmt.Printf("y:\n%1.1v", yV)

	//Output:
	// x:
	// ⎡-0.0003     0.01     0.04     0.09      0.2⎤
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// .
	// .
	// .
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎢-0.0003     0.01     0.04     0.09      0.2⎥
	// ⎣-0.0003     0.01     0.04     0.09      0.2⎦
	// y:
	// [0.3  0.3  0.3  0.3  ... 0.3  0.3  0.3  0.3]

}
