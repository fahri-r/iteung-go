package char

import (
	"io"
	// "fmt"
	"bytes"
)

// Prediction is the based type that can be used as a training dataset
type Prediction struct {
	input      *bytes.Buffer
	runeToIdx  func(r string) (int, error)
	sampleSize int
	generated  int
	vocabSize  int
	output     [][]float32
}

// NewPrediction return an object suitable for the LSTM
func NewPrediction(input string, runeToIdx func(r string) (int, error), sampleSize, vocabSize int) *Prediction {
	return &Prediction{
		input:      bytes.NewBufferString(input),
		runeToIdx:  runeToIdx,
		sampleSize: sampleSize,
		vocabSize:  vocabSize,
		output:     make([][]float32, 0),
	}
}

// Float32Read ...
func (p *Prediction) Read() ([]float32, error) {
	rn, err := p.input.ReadString(' ')

	// fmt.Println(rn)
	//
	// fmt.Println("after")

	if err != nil && err != io.EOF {
		return nil, err
	}
	if err == io.EOF && p.generated < p.sampleSize {
		p.generated++
		return p.output[len(p.output)-1], nil
	}
	if p.generated >= p.sampleSize {
		return nil, io.EOF
	}
	backend := make([]float32, p.vocabSize)
	idx, err := p.runeToIdx(rn)
	if err != nil {
		return nil, err
	}
	backend[idx] = 1
	return backend, nil
}

// Float32Write ...
func (p *Prediction) Write(val []float32) error {
	max := float32(0)
	idx := 0
	for i := range val {
		if val[i] >= max {
			max = val[i]
			idx = i
		}
	}
	output := make([]float32, len(val))
	output[idx] = 1
	p.output = append(p.output, output)
	return nil
}

// GetOutput ...
func (p *Prediction) GetOutput() [][]float32 {
	return p.output
}
