package main

import (
	"bufio"
	"context"
	"encoding/gob"
	"fmt"
	"io"
	_"io/ioutil"
	"flag"
	"log"
	"os"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/lstm"
	"github.com/owulveryck/lstm/datasetter/char"
)

type configuration struct {
	Dump string `envconfig:"dump" default:"checkpoint.bin"`
}

func newVocabulary(filename string) (vocabulary, error) {
	vocab := make(map[rune]struct{}, 0)
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	for {
		if c, _, err := r.ReadRune(); err != nil {
			if err == io.EOF {
				break
			}
			log.Fatal(err)
		} else {
			vocab[c] = struct{}{}
		}
	}
	output := make([]rune, len(vocab))
	i := 0
	for rne := range vocab {
		output[i] = rne
		i++
	}
	return output, nil

}

type vocabulary []rune

type backup struct {
	Model      lstm.Model
	Vocabulary vocabulary
}

func (v vocabulary) runeToIdx(r rune) (int, error) {
	for i := range v {
		if v[i] == r {
			return i, nil
		}
	}
	return 0, fmt.Errorf("Rune %v is not part of the vocabulary", r)
}

func (v vocabulary) idxToRune(i int) (rune, error) {
	var rn rune
	if i >= len(v) {
		return rn, fmt.Errorf("index invalid, no rune references")
	}
	return v[i], nil
}

func main() {
	
	// efore we brt we bus repetition. the superfluity say, he catunt thones not urfeits, er abe can bust ne

	var config configuration
	err := envconfig.Process("TRAIN", &config)
	if err != nil {
		log.Fatal(err)
	}

	var recovered backup

	f, err := os.Open(config.Dump)
	if err != nil {
		log.Println(err)
	}

	dec := gob.NewDecoder(f)
	err = dec.Decode(&recovered)
	if err != nil {
		log.Println(err)
	}

	model := recovered.Model
	vocab := recovered.Vocabulary

	prompt := flag.String("i", "halo", "the sentence you want to predict")
	flag.Parse()

	fmt.Println("---Vocabulary---")
	fmt.Println(string(vocab))
	fmt.Println("----------------")

	fmt.Println("Prompt:", *prompt)
	for _, r := range *prompt {
	    _, err := vocab.runeToIdx(r)

	    if err != nil {
		panic("Please use a prompt with known vocabulary characters")
	    }
	}

	vocabSize := len(vocab)

	prediction := char.NewPrediction(*prompt, vocab.runeToIdx, 100, vocabSize)
	err = model.Predict(context.TODO(), prediction)
	if err != nil {
		log.Println(err)
	}

	for _, output := range prediction.GetOutput() {
		var idx int
		for i, val := range output {
			if val == 1 {
				idx = i
			}
		}
		rne, err := vocab.idxToRune(idx)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf(string(rne))
	}
	fmt.Println("")
	
}
