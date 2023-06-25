// +build !cuda

package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

func finalizeTapeMachine(m *tapeMachine) {}

// UseCudaFor is an option for *tapeMachine. This function is NO-OP unless the program is built with the `cuda` tag.
func UseCudaFor(ops ...string) VMOpt {
	return func(m VM) {}
}

func (m *tapeMachine) getEngine(dev Device) tensor.Engine { return m.Engine }

func (instr *execOp) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v. Node is: %x", instr, instr.id)
	m.enterLogScope()
	defer m.leaveLogScope()

	// Read
	m.watchedLogf("Inputs:")
	m.enterLogScope()
	var inputs []Value
	for _, reg := range instr.readFrom {
		v := m.cpumem[reg.id]
		inputs = append(inputs, v)
		m.watchedLogf(m.valueFmt, v)
	}
	m.leaveLogScope()

	// check if the destination has already been allocated
	var usePrealloc bool
	dest := instr.writeTo.id
	if m.cpumem[dest] != nil {
		usePrealloc = true
	}

	// Execute
	var v Value
	switch {
	case instr.preAllocated:
		if pd, ok := instr.op.(UsePreallocDoer); ok {
			p := m.cpumem[instr.writeTo.id]
			if v, err = pd.UsePreallocDo(p, inputs...); err != nil {
				return errors.Wrapf(err, "Happened while attempting to execute %v. Node is %x. Register was: %v ", instr, instr.id, instr.writeTo.id)
			}
		} else {
			// TODO: maybe warn?
			if v, err = instr.op.Do(inputs...); err != nil {
				return errors.Wrap(err, opDoFail)
			}
		}
	case usePrealloc:
		if pd, ok := instr.op.(UsePreallocDoer); ok {
			p := m.cpumem[instr.writeTo.id]
			if v, err = pd.UsePreallocDo(p, inputs...); err != nil {
				if v, err = instr.op.Do(inputs...); err != nil {
					return errors.Wrap(err, opDoFail)
				}
			}
		} else {
			if v, err = instr.op.Do(inputs...); err != nil {
				return errors.Wrap(err, opDoFail)
			}
		}
	case instr.useUnsafe:
		if ud, ok := instr.op.(UnsafeDoer); ok {
			if v, err = ud.UnsafeDo(inputs...); err != nil {
				return errors.Wrap(err, "Failed to carry UnsafeDo()")
			}
		} else {
			// TODO: warn?
			if v, err = instr.op.Do(inputs...); err != nil {
				return errors.Wrap(err, opDoFail)
			}
		}
	default:
		if v, err = instr.op.Do(inputs...); err != nil {
			return errors.Wrap(err, opDoFail)
		}
	}

	m.watchedLogf("Result:")
	m.enterLogScope()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLogScope()
	// TODO: type and sohape checks

	// Write
	setEngine(v, m.Engine)

	m.cpumem[dest] = v
	node := m.p.g.Node(instr.id).(*Node)

	if m.trace() && (len(m.watchNodes) == 0 || m.watchNodes.Contains(node)) {
		if err = node.bindCopy(v); err != nil {
			return errors.Wrapf(err, "TraceExec failed to bind copy")
		}
		// Iop is special
		if node.op == (Iop{}) {
			v = node.Value()
			m.cpumem[dest] = v
		}

	} else {
		node.bind(v)
	}

	// this is a gradient node then, we should also bind the value to the node's dualValue
	if m.bindDV() && node.derivOf != nil {
		for _, src := range node.derivOf {
			if len(m.bindNodesDV) > 0 && !m.bindNodesDV.Contains(src) {
				continue
			}

			switch {
			case node.op == (Iop{}):
				// we'll need to put a closure into the closure queue
				closure := func() error {
					dv := dvUnit(src.boundTo)
					add := newEBOByType(addOpType, TypeOf(dv.d), TypeOf(v))
					if _, err := add.UnsafeDo(dv.d, v); err != nil {
						return err
					}
					return nil
				}
				m.closureQueue = append(m.closureQueue, closure)
			default:
				// TODO HERE
				/*
				   e.g. problem
				   z = y * (x + 1)

				   Here, 1 is a constant. But 1 comes early in the expression graph.
				   The final gradient is also 1, so 1 will also be the derivOf `z`
				   But because the graph is sorted, the 1 node will be walked before
				   the `z` node, and this part will cause a panic, as `z` will have no `Value`
				   associated with it yet.
				*/
				dv := dvUnit(src.boundTo)
				add := newEBOByType(addOpType, TypeOf(dv.d), TypeOf(v))

				if d, err := add.UnsafeDo(dv.d, v); err == nil {
					dv.SetDeriv(d)
					src.bind(dv)
				} else {
					return err
				}
			}

		}

	}

	m.watchedLogf("Written To: %v", instr.writeTo)
	m.enterLogScope()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLogScope()
	return nil
}

func (instr deviceTransport) exec(m *tapeMachine) error {
	return nil
}
