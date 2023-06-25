package vocab

import (
    "sync"
    "fmt"
)

type Vocabulary[K string, V int] struct {
	S         sync.RWMutex
	Immutable bool
	Forward   map[K]V
	Inverse   map[V]K
}

type InferenceVocabulary[K string, V int] struct {
    Forward   map[K]V
    Inverse   map[V]K
}

func NewInferenceVocabFromExsting[K string, V int](v Vocabulary[K, V]) *InferenceVocabulary[K, V] {
	return &InferenceVocabulary[K, V]{Forward: v.Forward, Inverse: v.Inverse}
}

func NewVocabStructure[K string, V int]() *Vocabulary[K, V] {
	return &Vocabulary[K, V]{Forward: make(map[K]V), Inverse: make(map[V]K), Immutable: false}
}

func (b *Vocabulary[K, V]) Insert(k K, v V) {
	b.S.RLock()
	if b.Immutable {
		panic("Cannot modify immutable map")
	}
	b.S.RUnlock()

	b.S.Lock()
	defer b.S.Unlock()

	if _, ok := b.Forward[k]; ok {
		delete(b.Inverse, b.Forward[k])
	}

	b.Forward[k] = v
	b.Inverse[v] = k
}

func (b *Vocabulary[K, V]) Get(k K) (V, bool) {
	if !b.Exists(k) {
		return *new(V), false
	}
	b.S.RLock()
	defer b.S.RUnlock()
	return b.Forward[k], true
}

func (b *Vocabulary[K, V]) GetInverse(v V) (K, bool) {
	if !b.ExistsInverse(v) {
		return *new(K), false
	}
	b.S.RLock()
	defer b.S.RUnlock()
	return b.Inverse[v], true

}

func (b *Vocabulary[K, V]) Exists(k K) bool {
	b.S.RLock()
	defer b.S.RUnlock()
	_, ok := b.Forward[k]
	return ok
}

func (b *Vocabulary[K, V]) ExistsInverse(k V) bool {
	b.S.RLock()
	defer b.S.RUnlock()

	_, ok := b.Inverse[k]
	return ok
}

func (b *Vocabulary[K, V]) Size() int {
	b.S.RLock()
	defer b.S.RUnlock()
	return len(b.Forward)
}

func (v Vocabulary[string, int]) TokenToIdx(r string) (int, error) {
	val, exists := v.Get(r)

	if exists {
	    return val, nil
	}

	return 0, fmt.Errorf("Token %v is not part of the vocabulary", string(r))
}

func (v Vocabulary[string, int]) IdxToToken(i int) (string, error) {
	val, exists := v.GetInverse(i)

	if exists {
	    return val, nil
	}

	return "", fmt.Errorf("Invalid index: %v ", i)
}
