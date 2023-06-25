package gorgonia_test

import (
	"io"
	"runtime"
	"testing"

	"gorgonia.org/tensor"
)

func BenchmarkTrainingConcurrent(b *testing.B) {
	xV, yV, bs := prep()

	for i := 0; i < b.N; i++ {
		concurrentTraining(xV, yV, bs, 10)
	}

	runtime.GC()
}

func BenchmarkTrainingNonConcurrent(b *testing.B) {
	xV, yV, _ := prep()

	for i := 0; i < b.N; i++ {
		nonConcurrentTraining(xV, yV, 10)
	}

	runtime.GC()
}

func BenchmarkTapeMachineExecution(b *testing.B) {
	m, c, machine := linregSetup(tensor.Float64)
	for i := 0; i < b.N; i++ {
		linregRun(m, c, machine, 100, false)
	}
	machine.(io.Closer).Close()
}
