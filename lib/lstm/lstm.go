package lstm

import (
	"io"

	"github.com/owulveryck/lstm/datasetter"
	G "gorgonia.org/gorgonia"
)

// forwardStep as described here https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
// It returns the last hidden node and the last cell node
func (l *lstm) forwardStep(dataSet datasetter.ReadWriter, prevHidden, prevCell *G.Node, step int) (*G.Node, *G.Node, error) {
	// Read the current input vector
	inputVector, err := dataSet.ReadInputVector(l.g)

	switch {
	case err != nil && err != io.EOF:
		return prevHidden, prevCell, err
	case err == io.EOF:
		return prevHidden, prevCell, nil
	}
	// Helper function for clarity
	// r is a replacer that will change ₜ and ₜ₋₁ for step and step-1
	// this is to avoid conflict in the graph due to the recursion
	r := replace(`ₜ`, step)
	set := func(ident, equation string) *G.Node {
		//log.Printf("==> %v=%v", r.Replace(ident), r.Replace(equation))
		res, _ := l.parser.Parse(r.Replace(equation))
		l.parser.Set(r.Replace(ident), res)
		return res
	}

	l.parser.Set(r.Replace(`xₜ`), inputVector)
	if step == 0 {
		l.parser.Set(r.Replace(`hₜ₋₁`), prevHidden)
		l.parser.Set(r.Replace(`cₜ₋₁`), prevCell)

	}
	set(`iₜ`, `σ(Wᵢ·xₜ+Uᵢ·hₜ₋₁+Bᵢ)`)
	set(`fₜ`, `σ(Wf·xₜ+Uf·hₜ₋₁+Bf)`) // dot product made with ctrl+k . M
	set(`oₜ`, `σ(Wₒ·xₜ+Uₒ·hₜ₋₁+Bₒ)`)
	// ċₜis a vector of new candidates value
	set(`ĉₜ`, `tanh(Wc·xₜ+Uc·hₜ₋₁+Bc)`) // c made with ctrl+k c >
	ct := set(`cₜ`, `(fₜ*cₜ₋₁)+(iₜ*ĉₜ)`)
	ht := set(`hₜ`, `oₜ*tanh(cₜ)`)
	y := set(`yₜ`, `softmax(Wy·hₜ+By)`)

	dataSet.WriteComputedVector(y)
	return l.forwardStep(dataSet, ht, ct, step+1)
}
