package lstm

import (
	"fmt"
	"io"
	"math/rand"

	"github.com/owulveryck/lstm/datasetter"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// testBackends returns a backend with predictabale values for the test
// biais are zeroes and matrices are 1
func testBackends(inputSize, outputSize int, hiddenSize int) *backends {
	var back backends
	initValue := float32(1e-3)
	back.InputSize = inputSize
	back.OutputSize = outputSize
	back.HiddenSize = hiddenSize
	back.Wi = make([]float32, hiddenSize*inputSize)
	for i := 0; i < hiddenSize*inputSize; i++ {
		back.Wi[i] = initValue * rand.Float32()
	}
	back.Ui = make([]float32, hiddenSize*hiddenSize)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		back.Ui[i] = initValue * rand.Float32()
	}
	back.BiasI = make([]float32, hiddenSize)
	back.Wo = make([]float32, hiddenSize*inputSize)
	for i := 0; i < hiddenSize*inputSize; i++ {
		back.Wo[i] = initValue * rand.Float32()
	}
	back.Uo = make([]float32, hiddenSize*hiddenSize)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		back.Uo[i] = initValue * rand.Float32()
	}
	back.BiasO = make([]float32, hiddenSize)
	back.Wf = make([]float32, hiddenSize*inputSize)
	for i := 0; i < hiddenSize*inputSize; i++ {
		back.Wf[i] = initValue * rand.Float32()
	}
	back.Uf = make([]float32, hiddenSize*hiddenSize)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		back.Uf[i] = initValue * rand.Float32()
	}
	back.BiasF = make([]float32, hiddenSize)
	back.Wc = make([]float32, hiddenSize*inputSize)
	for i := 0; i < hiddenSize*inputSize; i++ {
		back.Wc[i] = initValue * rand.Float32()
	}
	back.Uc = make([]float32, hiddenSize*hiddenSize)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		back.Uc[i] = initValue * rand.Float32()
	}
	back.BiasC = make([]float32, hiddenSize)
	back.Wy = make([]float32, hiddenSize*outputSize)
	for i := 0; i < hiddenSize*outputSize; i++ {
		back.Wy[i] = initValue * rand.Float32()
	}
	back.BiasY = make([]float32, outputSize)
	return &back
}

type testSet struct {
	values         [][]float32
	expectedValues []int
	offset         int
	output         G.Nodes
	outputValues   [][]float32
	epoch          int
	maxEpoch       int
}

func (t *testSet) ReadInputVector(g *G.ExprGraph) (*G.Node, error) {
	if t.offset >= len(t.values) {
		return nil, io.EOF
	}
	size := len(t.values[t.offset])
	inputTensor := tensor.New(tensor.WithShape(size), tensor.WithBacking(t.values[t.offset]))
	node := G.NewVector(g, tensor.Float32, G.WithName(fmt.Sprintf("input_%v", t.offset)), G.WithShape(size), G.WithValue(inputTensor))
	t.offset++
	return node, nil
}

func (t *testSet) flush() error {
	t.outputValues = make([][]float32, len(t.output))
	for i, node := range t.output {
		t.outputValues[i] = node.Value().Data().([]float32)
	}
	return nil
}

func (t *testSet) WriteComputedVector(n *G.Node) error {
	t.output = append(t.output, n)
	return nil
}

func (t *testSet) GetComputedVectors() G.Nodes {
	return t.output
}

func (t *testSet) GetExpectedValue(offset int) (int, error) {
	return t.expectedValues[offset], nil
}

func (t *testSet) GetTrainer() (datasetter.Trainer, error) {
	if t.epoch <= t.maxEpoch {
		t.epoch++
		t.offset = 0
		t.output = make([]*G.Node, 0)
		return t, nil
	}
	return nil, io.EOF
}
