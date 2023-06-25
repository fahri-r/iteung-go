package nnops

import (
	"hash/fnv"
	"unsafe"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func simpleHash(op gorgonia.Op) uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func checkArity(op gorgonia.Op, inputs int) error {
	if inputs != op.Arity() && op.Arity() >= 0 {
		return errors.Errorf("%v has an arity of %d. Got %d instead", op, op.Arity(), inputs)
	}
	return nil
}

// nomem is a dummy type that implements cudnn.Memory, but returns 0 for all the pointer.
//
// It's essentially "nil" for CUDA memory
type nomem struct{}

func (nomem) Uintptr() uintptr           { return 0 }
func (nomem) Pointer() unsafe.Pointer    { return nil }
func (nomem) IsNativelyAccessible() bool { return false }

func calcMemSize(dt tensor.Dtype, s tensor.Shape) uintptr {
	var elemSize uintptr
	if s.IsScalar() {
		elemSize = 1
	} else {
		elemSize = uintptr(s.TotalSize())
	}
	dtSize := dt.Size()
	return elemSize * dtSize
}

func dtypeOf(t hm.Type) (retVal tensor.Dtype, err error) {
	switch p := t.(type) {
	case tensor.Dtype:
		retVal = p
	case gorgonia.TensorType:
		return dtypeOf(p.Of)
	case hm.TypeVariable:
		err = errors.Errorf("instance %v does not have a dtype", p)
	default:
		err = errors.Errorf("Not yet implemented: %v %v", "dtypeOf", p)
		return
	}

	return
}

func CheckConvolutionParams(pad, stride, dilation []int) error {
	// checks
	for _, s := range stride {
		if s <= 0 {
			return errors.Errorf("Cannot use strides of less than or equal 0: %v", stride)
		}
	}

	for _, p := range pad {
		if p < 0 {
			return errors.Errorf("Cannot use padding of less than 0: %v", pad)
		}
	}

	for _, d := range dilation {
		if d <= 0 {
			return errors.Errorf("Cannot use dilation less than or eq 0 %v", dilation)
		}
	}
	return nil
}
