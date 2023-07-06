package main

import (
	"context"
	"encoding/gob"
	"fmt"
	_"io/ioutil"
	"log"
	"os"
	"strings"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/lstm"
	"github.com/owulveryck/lstm/datasetter/char"

	."github.com/fahri-r/iteung-go/vocab"
)

type configuration struct {
	Dump string `envconfig:"dump" default:"checkpoint.bin"`
}

type backup struct {
	Model      lstm.Model
	Vocabulary Vocabulary[string, int]
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

	args := os.Args[1:]
	prompt := ""
	for _, arg := range args {
		prompt += arg + " "
	}

	prompt = strings.TrimRight(prompt, " ")

	fmt.Println("Prompt:", prompt)
	// fmt.Printf("Vocabulary: %v\n", vocab.Size())

	parts := strings.Fields(prompt)
	for _, r := range parts {
	    _, err := vocab.TokenToIdx(r)

	    if err != nil {
		panic("Please use a prompt with known vocabulary characters")
	    }
	}

	vocabSize := vocab.Size()

	prediction := char.NewPrediction(prompt, vocab.TokenToIdx, 100, vocabSize)

	err = model.Predict(context.TODO(), prediction)
	if err != nil {
		log.Println(err)
	}

	// fmt.Println("Prediction Size:", len(prediction.GetOutput()))

	for _, output := range prediction.GetOutput() {
		var idx int
		for i, val := range output {
			if val == 1 {
				idx = i
			}
		}
		rne, err := vocab.IdxToToken(idx)
		if err != nil {
			log.Fatal(err)
		}
		// fmt.Printf("%v\n", output)
		fmt.Printf(string(rne))
	}
	fmt.Println("")
	
}
